"""Models + init + quantization (self-contained; no dependency on prior iterations).

Coord layout matches CycloneNFDataset.f_grid: (v_par, mu, s, x_phys, y_phys).
  spatial block = [2,3,4]=(s,x,y),  velocity block = [0,1]=(v_par,mu).
Two primitives:
  GaussianSplat5D : f_hat = Σ_i c_i · exp(-½‖L_phys⁻¹(x_phys-μ)‖² -½‖L_vel⁻¹(x_vel-μ)‖²)  (16 params/atom)
  GaborSplat5D    : + per-atom binormal carrier ky: atom_i · exp(i·ky_i·(y-μy_i))            (17 params/atom)
Block-diagonal Σ (Cholesky-parameterized) ⇒ the field is LINEAR in the complex amplitudes c=(Re,Im) and
SEPARABLE on the grid (powers the amp warm-start + the fast contraction in fast.py).
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

PHYS = slice(2, 5)
VEL = slice(0, 2)
YIDX = 4  # binormal y = last coord


def flat_to_lower_tri(flat: torch.Tensor, d: int) -> torch.Tensor:
    """(N, d*(d+1)/2) -> (N,d,d) lower-tri, softplus-positive diagonal."""
    N = flat.shape[0]
    L = flat.new_zeros(N, d, d)
    diag = torch.arange(d, device=flat.device)
    L[:, diag, diag] = F.softplus(flat[:, :d])
    if d > 1:
        row, col = torch.tril_indices(d, d, offset=-1, device=flat.device)
        L[:, row, col] = flat[:, d:]
    return L


def inv_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp(min=eps)
    return x + torch.log(-torch.expm1(-x))


class GaussianSplat5D(nn.Module):
    """16 params/Gaussian: mu(5) + L_phys(6) + L_vel(3) + amps(2). forward returns (...,2) channels-last."""
    D_PHYS, D_VEL = 3, 2

    def __init__(self, n_gaussians, init_mu, init_L_phys, init_L_vel, init_amps,
                 chunk_size: int = 512, use_checkpoint: bool = False, **_):
        super().__init__()
        self.N = n_gaussians
        self.chunk = chunk_size
        self.use_checkpoint = use_checkpoint
        self.embed_type = "none"
        self.grid_size = None
        self.mu = nn.Parameter(init_mu)
        self.L_phys_raw = nn.Parameter(init_L_phys)
        self.L_vel_raw = nn.Parameter(init_L_vel)
        self.amps = nn.Parameter(init_amps)

    def L_phys(self):
        return flat_to_lower_tri(self.L_phys_raw, self.D_PHYS)

    def L_vel(self):
        return flat_to_lower_tri(self.L_vel_raw, self.D_VEL)

    def _impl(self, coords):
        B = coords.shape[0]
        cp, cv = coords[:, PHYS], coords[:, VEL]
        Lp, Lv = self.L_phys(), self.L_vel()
        out = coords.new_zeros(B, 2)
        for s in range(0, self.N, self.chunk):
            sl = slice(s, s + self.chunk)
            dphys = cp.unsqueeze(1) - self.mu[sl, PHYS].unsqueeze(0)
            dvel = cv.unsqueeze(1) - self.mu[sl, VEL].unsqueeze(0)
            up = torch.linalg.solve_triangular(Lp[sl], dphys.permute(1, 2, 0), upper=False).permute(2, 0, 1)
            uv = torch.linalg.solve_triangular(Lv[sl], dvel.permute(1, 2, 0), upper=False).permute(2, 0, 1)
            G = torch.exp(-0.5 * ((up ** 2).sum(-1) + (uv ** 2).sum(-1)))
            out = out + G @ self.amps[sl]
        return out

    def forward(self, coords):
        # use_checkpoint matters for the DENSE physics path (train_pinc mode='dense'): without it every
        # voxel-block's (B, chunk, 3) intermediates stay in the autograd graph (~86 GB at full grid).
        orig = coords.shape[:-1]
        cf = coords.reshape(-1, 5)
        if self.use_checkpoint and torch.is_grad_enabled():
            out = torch.utils.checkpoint.checkpoint(self._impl, cf, use_reentrant=False)
        else:
            out = self._impl(cf)
        return out.reshape(*orig, 2)


class GaborSplat5D(GaussianSplat5D):
    """GaussianSplat5D + per-atom binormal carrier ky. ky=0 reduces EXACTLY to parent Gaussian.
    complex amplitude c_i = amps[:,0]+i·amps[:,1]."""

    def __init__(self, *args, init_ky=None, **kw):
        super().__init__(*args, **kw)
        ky = torch.zeros(self.N) if init_ky is None else init_ky
        self.ky = nn.Parameter(ky.to(self.mu.device).float())

    def _impl(self, coords):
        B = coords.shape[0]
        cp, cv = coords[:, PHYS], coords[:, VEL]
        Lp, Lv = self.L_phys(), self.L_vel()
        out = coords.new_zeros(B, 2)
        for s in range(0, self.N, self.chunk):
            sl = slice(s, s + self.chunk)
            up = torch.linalg.solve_triangular(Lp[sl], (cp.unsqueeze(1) - self.mu[sl, PHYS].unsqueeze(0)).permute(1, 2, 0), upper=False).permute(2, 0, 1)
            uv = torch.linalg.solve_triangular(Lv[sl], (cv.unsqueeze(1) - self.mu[sl, VEL].unsqueeze(0)).permute(1, 2, 0), upper=False).permute(2, 0, 1)
            Gmag = torch.exp(-0.5 * ((up ** 2).sum(-1) + (uv ** 2).sum(-1)))
            theta = self.ky[sl].unsqueeze(0) * (coords[:, YIDX:YIDX + 1] - self.mu[sl, YIDX].unsqueeze(0))
            a, b = self.amps[sl, 0], self.amps[sl, 1]
            out[:, 0] = out[:, 0] + (Gmag * (a * torch.cos(theta) - b * torch.sin(theta))).sum(1)
            out[:, 1] = out[:, 1] + (Gmag * (a * torch.sin(theta) + b * torch.cos(theta))).sum(1)
        return out


def from_gaussian(g: GaussianSplat5D, init_ky=None) -> GaborSplat5D:
    """Lift a trained GaussianSplat5D to a Gabor model (ky=0 => identical field)."""
    return GaborSplat5D(g.N, g.mu.data.clone(), g.L_phys_raw.data.clone(), g.L_vel_raw.data.clone(),
                        g.amps.data.clone(), chunk_size=g.chunk, init_ky=init_ky).to(g.mu.device)


def build_gs_5d(data, n_gaussians, device, sigma_grid_cells: float = 2.0, seed: int = 0,
                chunk_size: Optional[int] = None):
    """Importance-sample Gaussian centers from |f|², σ ~ grid spacing, amps = voxel values."""
    g = torch.Generator(device=data.f_df.device).manual_seed(seed)
    w = (data.f_df ** 2).sum(0).to(torch.float64) + 1e-12
    idx = torch.multinomial(w / w.sum(), n_gaussians, replacement=False, generator=g)
    mu = data.f_grid[idx].clone().to(device).float()
    amps = data.f_df[:, idx].T.clone().to(device).float()
    grid_shape = torch.tensor(data.df.shape[1:], dtype=torch.float32, device=device)
    sig = sigma_grid_cells / grid_shape
    Lp = torch.zeros(n_gaussians, 6, device=device); Lp[:, :3] = inv_softplus(sig[2:5])
    Lv = torch.zeros(n_gaussians, 3, device=device); Lv[:, :2] = inv_softplus(sig[0:2])
    return GaussianSplat5D(n_gaussians, mu, Lp, Lv, amps,
                           chunk_size=chunk_size or min(512, max(64, n_gaussians))).to(device)


BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "fp8": 1, "fp4": 0.5}


@torch.no_grad()
def quantize_(model, quant: str = "fp16") -> None:
    """Realize a storage scheme by rounding params in place (kept fp32-dtype for compute).
    fp16 -> native half (PSNR-lossless). fp8/fp4 -> per-attribute min-max round."""
    if quant == "fp32":
        return
    if quant == "fp16":
        for p in model.parameters():
            p.data = p.data.half().float()
        return
    bits = {"fp8": 8, "fp4": 4}.get(quant)
    if bits is None:
        raise ValueError(f"unknown quant: {quant}")
    levels = 2 ** bits - 1
    for p in model.parameters():
        t = p.data
        flat = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(-1, 1)
        mn = flat.amin(0, keepdim=True); s = (flat.amax(0, keepdim=True) - mn).clamp_min(1e-12) / levels
        p.data = (((flat - mn) / s).round().clamp(0, levels) * s + mn).reshape(t.shape)
