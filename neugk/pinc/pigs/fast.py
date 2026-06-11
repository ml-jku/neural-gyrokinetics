"""Fast primitives (self-contained): closed-form amplitude warm-start, exact separable grid
contraction (Gaussian + Gabor), and complex carrier-aware amplitude solve.

Math: block-diagonal Σ ⇒ each atom factorizes G_i = P_i(x_phys)·V_i(x_vel), so on the regular grid
f̂[vel,phys] = Σ_i c_i V_i[vel] P_i[phys] is an exact CP/Kronecker contraction — one full-grid step is a
GEMM (~17 ms) vs ~6 s dense, equal to the dense forward to machine precision.
"""
import torch
from neugk.pinc.pigs.model import flat_to_lower_tri, PHYS, VEL, YIDX
from neugk.pinc.neural_fields.nf_utils import to_complex


@torch.no_grad()
def solve_amplitudes(model, f_grid, f_df, n_sample: int = 1_000_000, lam: float = 1e-3,
                     vbatch: int = 100_000, generator=None):
    """c* = (ΦᵀΦ + λI)⁻¹ ΦᵀY over a uniform voxel subsample; streams the N×N normal equations."""
    dev = model.mu.device
    M, N = f_grid.shape[0], model.N
    idx = (torch.randint(0, M, (n_sample,), device=f_grid.device, generator=generator)
           if n_sample and n_sample < M else torch.arange(M, device=f_grid.device))
    GtG = torch.zeros(N, N, device=dev); GtY = torch.zeros(N, 2, device=dev)
    for s in range(0, idx.numel(), vbatch):
        ii = idx[s:s + vbatch]
        G = _basis(model, f_grid[ii].to(dev).float())
        GtG += G.T @ G; GtY += G.T @ f_df[:, ii].T.to(dev).float()
    GtG.diagonal().add_(lam * GtG.diagonal().mean().clamp(min=1e-12))
    model.amps.data.copy_(torch.linalg.solve(GtG, GtY).to(model.amps.dtype))


def _basis(model, coords, chunk=None):
    cp, cv = coords[:, PHYS], coords[:, VEL]
    Lp, Lv = model.L_phys(), model.L_vel()
    N = model.N; chunk = chunk or N; cols = []
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        dphys = cp.unsqueeze(1) - model.mu[sl, PHYS].unsqueeze(0)
        dvel = cv.unsqueeze(1) - model.mu[sl, VEL].unsqueeze(0)
        up = torch.linalg.solve_triangular(Lp[sl], dphys.permute(1, 2, 0), upper=False).permute(2, 0, 1)
        uv = torch.linalg.solve_triangular(Lv[sl], dvel.permute(1, 2, 0), upper=False).permute(2, 0, 1)
        cols.append(torch.exp(-0.5 * ((up ** 2).sum(-1) + (uv ** 2).sum(-1))))
    return torch.cat(cols, 1)


def _ff_impl(mu, Lp_raw, Lv_raw, amps, coords):
    Lpi = torch.linalg.inv(flat_to_lower_tri(Lp_raw, 3))
    Lvi = torch.linalg.inv(flat_to_lower_tri(Lv_raw, 2))
    dp = coords[:, PHYS].unsqueeze(1) - mu[:, PHYS].unsqueeze(0)
    dv = coords[:, VEL].unsqueeze(1) - mu[:, VEL].unsqueeze(0)
    up = torch.einsum('nij,bnj->bni', Lpi, dp)
    uv = torch.einsum('nij,bnj->bni', Lvi, dv)
    G = torch.exp(-0.5 * ((up * up).sum(-1) + (uv * uv).sum(-1)))
    return G @ amps


_compiled = None


def fast_forward(model, coords, compile: bool = True):
    """Drop-in for model(coords) on the density polish; inductor fuses the (B,N,3) intermediate."""
    global _compiled
    fn = _ff_impl
    if compile:
        if _compiled is None:
            _compiled = torch.compile(_ff_impl, dynamic=True)
        fn = _compiled
    return fn(model.mu, model.L_phys_raw, model.L_vel_raw, model.amps, coords)


def build_subgrids(data, device):
    G = data.grid.to(device)
    vel_grid = G[:, :, 0, 0, 0, 0:2].reshape(-1, 2).contiguous()
    phys_grid = G[0, 0, :, :, :, 2:5].reshape(-1, 3).contiguous()
    return vel_grid, phys_grid, vel_grid.shape[0], phys_grid.shape[0]


def grid_target(data, Nv, Np, device):
    return data.full_df.reshape(2, -1).t().reshape(Nv, Np, 2).contiguous().to(device)


def sep_field(model, vel_grid, phys_grid, pchunk: int = 8192):
    """Exact Gaussian f̂ on the full grid -> (Nv, Np, 2). Differentiable."""
    Lpi = torch.linalg.inv(flat_to_lower_tri(model.L_phys_raw, 3))
    Lvi = torch.linalg.inv(flat_to_lower_tri(model.L_vel_raw, 2))
    mup, muv = model.mu[:, PHYS], model.mu[:, VEL]
    dv = vel_grid.unsqueeze(0) - muv.unsqueeze(1)
    V = torch.exp(-0.5 * (torch.einsum('nij,nvj->nvi', Lvi, dv) ** 2).sum(-1))
    Pc = []
    for s in range(0, phys_grid.shape[0], pchunk):
        dp = phys_grid[s:s + pchunk].unsqueeze(0) - mup.unsqueeze(1)
        Pc.append(torch.exp(-0.5 * (torch.einsum('nij,ncj->nci', Lpi, dp) ** 2).sum(-1)))
    P = torch.cat(Pc, 1)
    return torch.stack([V.t() @ (P * model.amps[:, c:c + 1]) for c in range(2)], -1)


def sep_sample_field(model, data, vel_grid, phys_grid):
    """Separable Gaussian reconstruction in the DENORMALIZED (2,vpar,mu,s,x,y) layout."""
    f = sep_field(model, vel_grid, phys_grid)
    nvp, nmu, ns, nx, ny = data.df.shape[1:]
    f = f.reshape(nvp, nmu, ns, nx, ny, 2).permute(5, 0, 1, 2, 3, 4).contiguous()
    scale = data.scale["df"].reshape(2, 1, 1, 1, 1, 1).to(f.device)
    shift = data.shift["df"].reshape(2, 1, 1, 1, 1, 1).to(f.device)
    return f * scale + shift


def gabor_sep_field(model, vel_grid, phys_grid, pchunk: int = 8192):
    """Exact Gabor f̂ -> (Nv, Np, 2) = (Re, Im) of complex df. Reduces bit-identically to sep_field at ky=0."""
    Lpi = torch.linalg.inv(flat_to_lower_tri(model.L_phys_raw, 3))
    Lvi = torch.linalg.inv(flat_to_lower_tri(model.L_vel_raw, 2))
    mup, muv = model.mu[:, PHYS], model.mu[:, VEL]
    dv = vel_grid.unsqueeze(0) - muv.unsqueeze(1)
    V = torch.exp(-0.5 * (torch.einsum('nij,nvj->nvi', Lvi, dv) ** 2).sum(-1))
    c = torch.complex(model.amps[:, 0], model.amps[:, 1])
    muy = model.mu[:, YIDX]
    chunks = []
    for s in range(0, phys_grid.shape[0], pchunk):
        pg = phys_grid[s:s + pchunk]
        dp = pg.unsqueeze(0) - mup.unsqueeze(1)
        Pmag = torch.exp(-0.5 * (torch.einsum('nij,ncj->nci', Lpi, dp) ** 2).sum(-1))
        theta = model.ky.unsqueeze(1) * (pg[:, 2].unsqueeze(0) - muy.unsqueeze(1))
        chunks.append(Pmag * torch.polar(torch.ones_like(theta), theta))
    Pc = torch.cat(chunks, 1)
    df = V.t().to(Pc.dtype) @ (Pc * c.unsqueeze(1))
    return torch.stack([df.real, df.imag], -1)


def gabor_denorm(model, data, vel_grid, phys_grid):
    """Gabor separable reconstruction in the DENORMALIZED (2,vpar,mu,s,x,y) layout (for physics losses)."""
    f = gabor_sep_field(model, vel_grid, phys_grid)
    nvp, nmu, ns, nx, ny = data.df.shape[1:]
    f = f.reshape(nvp, nmu, ns, nx, ny, 2).permute(5, 0, 1, 2, 3, 4).contiguous()
    scale = data.scale["df"].reshape(2, 1, 1, 1, 1, 1).to(f.device)
    shift = data.shift["df"].reshape(2, 1, 1, 1, 1, 1).to(f.device)
    return f * scale + shift


@torch.no_grad()
def reconstruct(model, data, device):
    """Fast EXACT denormalized field (Gaussian or Gabor) for plotting/eval.
    Automatically converts Gaussian to Gabor (ky=0) for consistent interface."""
    from neugk.pinc.pigs.model import GaborSplat5D, from_gaussian
    vg, pg, _, _ = build_subgrids(data, device)
    # Ensure model is Gabor (converts Gaussian with ky=0 if needed)
    m = model if isinstance(model, GaborSplat5D) else from_gaussian(model)
    return gabor_denorm(m, data, vg, pg).to(device)


def ky_per_bin(data, device):
    """Angular wavenumber (normalized-y units) for each FFT bin m along the binormal y."""
    ny = data.df.shape[-1]
    _, pg, _, _ = build_subgrids(data, device)
    y = pg[:, 2]
    Ly = (y.max() - y.min()) * ny / (ny - 1)
    return 2 * torch.pi * torch.arange(ny, device=device) / Ly


@torch.no_grad()
def solve_new_amps_complex(model, mask, data, device, n_sample=500_000, lam=1e-3, vbatch=100_000, seed=0):
    """Closed-form COMPLEX (carrier-aware) amplitude warm-start for masked atoms, fit to the df
    RESIDUAL of the frozen atoms -- the carrier-aware analog of solve_amplitudes."""
    vg, pg, _, _ = build_subgrids(data, device)
    base = gabor_sep_field(model, vg, pg)
    Nv, Np = base.shape[:2]
    gt = data.full_df.reshape(2, -1).t().reshape(Nv, Np, 2).to(device)
    r = (gt - base).reshape(-1, 2); res_c = torch.complex(r[:, 0], r[:, 1])
    grid = data.grid.reshape(-1, 5).to(device); Nvox = grid.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    idx = (torch.randperm(Nvox, generator=g, device=device)[:n_sample] if n_sample < Nvox else torch.arange(Nvox, device=device))
    new = mask.nonzero().squeeze(1); Mn = new.numel()
    mu = model.mu[new]; Lp = model.L_phys()[new]; Lv = model.L_vel()[new]; ky = model.ky[new]; muy = mu[:, YIDX]
    BhB = torch.zeros(Mn, Mn, dtype=torch.complex64, device=device)
    Bhr = torch.zeros(Mn, dtype=torch.complex64, device=device)
    for s in range(0, idx.numel(), vbatch):
        ii = idx[s:s + vbatch]; co = grid[ii]
        up = torch.linalg.solve_triangular(Lp, (co[:, PHYS].unsqueeze(1) - mu[:, PHYS].unsqueeze(0)).permute(1, 2, 0), upper=False).permute(2, 0, 1)
        uv = torch.linalg.solve_triangular(Lv, (co[:, VEL].unsqueeze(1) - mu[:, VEL].unsqueeze(0)).permute(1, 2, 0), upper=False).permute(2, 0, 1)
        Gmag = torch.exp(-0.5 * ((up ** 2).sum(-1) + (uv ** 2).sum(-1)))
        theta = ky.unsqueeze(0) * (co[:, YIDX:YIDX + 1] - muy.unsqueeze(0))
        B = Gmag * torch.polar(torch.ones_like(theta), theta)
        BhB += B.conj().t() @ B; Bhr += B.conj().t() @ res_c[ii]
    BhB.diagonal().add_(lam * BhB.diagonal().real.mean().clamp_min(1e-12))
    c = torch.linalg.solve(BhB, Bhr)
    model.amps.data[new, 0] = c.real.to(model.amps.dtype); model.amps.data[new, 1] = c.imag.to(model.amps.dtype)
