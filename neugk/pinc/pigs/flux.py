"""Flux graft + evaluation (self-contained).
  residual_centers : place new atoms WHERE THE FIT IS WORST (boosting placement)
  add_atoms        : base + M BROAD Gabor carriers at the flux tail (broad sigma_y -> concentrated spectral
                     peak that targets one ky precisely)
  evaluate         : PSNR(f)/PSNR(phi)/kyspec/qspec + heat-flux relL1 split into dominant (ky1-6) & tail (ky7-15)
"""
import torch

from neugk.pinc.neural_fields import integral_losses, spectra_losses
from neugk.physics.integrals import get_integrals

from neugk.pinc.pigs.model import GaborSplat5D, build_gs_5d, inv_softplus, YIDX
from neugk.pinc.pigs.fast import build_subgrids, gabor_sep_field, gabor_denorm, ky_per_bin

TAIL_BINS = list(range(7, 16))


@torch.no_grad()
def residual_centers(model, data, device, M, seed=2):
    """Importance-sample M centers (5D) from |df - reconstruction|^2 -- where the fit is worst."""
    vg, pg, _, _ = build_subgrids(data, device)
    pred = gabor_sep_field(model, vg, pg)
    Nv, Np = pred.shape[:2]
    gt = data.full_df.reshape(2, -1).t().reshape(Nv, Np, 2).to(device)
    res2 = ((pred - gt) ** 2).sum(-1).reshape(-1).double() + 1e-12
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.multinomial(res2 / res2.sum(), M, replacement=False, generator=g)
    grid = data.grid.reshape(-1, 5)
    return grid[idx.to(grid.device)].to(device).float()


@torch.no_grad()
def add_atoms(base: GaborSplat5D, data, device, M, bins=TAIL_BINS, sigma_y_cells=8.0, seed=1, shared_mu=None):
    """Return (model, new_mask): base atoms + M fresh BROAD Gabor carriers seeded at the flux `bins` (cycled).
    New amps init 0 (contribute nothing until trained). `shared_mu` (M,5) overrides placement."""
    tmpl = build_gs_5d(data, M, device, seed=seed)
    kyvals = ky_per_bin(data, device)
    init_ky = kyvals[torch.tensor([bins[i % len(bins)] for i in range(M)], device=device)]
    grid_shape = torch.tensor(data.df.shape[1:], dtype=torch.float32, device=device)
    Lp = tmpl.L_phys_raw.data.clone()
    Lp[:, 2] = inv_softplus((torch.tensor(sigma_y_cells, device=device) / grid_shape[YIDX]).clamp_min(1e-4))
    mu_new = shared_mu.clone() if shared_mu is not None else tmpl.mu.data
    N0 = base.N
    mu = torch.cat([base.mu.data, mu_new], 0)
    Lpr = torch.cat([base.L_phys_raw.data, Lp], 0)
    Lvr = torch.cat([base.L_vel_raw.data, tmpl.L_vel_raw.data], 0)
    amps = torch.cat([base.amps.data, torch.zeros(M, 2, device=device)], 0)
    ky = torch.cat([base.ky.data, init_ky], 0)
    m = GaborSplat5D(N0 + M, mu, Lpr, Lvr, amps, chunk_size=min(512, N0 + M), init_ky=ky).to(device)
    mask = torch.zeros(N0 + M, dtype=torch.bool, device=device); mask[N0:] = True
    return m, mask


@torch.no_grad()
def evaluate(model, data, device, band_dom=(1, 7), band_tail=(7, 16)):
    """Paper metrics + heat-flux relL1 split (dominant ky1-6 vs tail ky7-15). Evaluate the model you DEPLOY
    (i.e. quantize_ to fp16 first for on-disk-fair numbers). Automatically converts Gaussian to Gabor if needed."""
    from math import log10
    from neugk.pinc.pigs.model import GaborSplat5D, from_gaussian
    vg, pg, _, _ = build_subgrids(data, device)
    # Ensure model is Gabor (converts Gaussian with ky=0 if needed for consistent interface)
    m = model if isinstance(model, GaborSplat5D) else from_gaussian(model)
    pred = gabor_denorm(m, data, vg, pg).to(device)
    gt = data.full_df.to(device)
    il, (pphi, gphi), (pef, gef) = integral_losses(pred, gt, geom=data.geom, device=device,
                                                   use_flux_fields=False, timestep=None, return_fields=True)
    sl, _ = spectra_losses(pred_df=pred, pred_phi=pphi, pred_eflux=pef,
                           gt_df=gt, gt_phi=gphi, gt_eflux=gef, ds=data.ds)
    _, (_, ef, _) = get_integrals(pred, data.geom, flux_fields=True, spectral_df=False)
    _, (_, efg, _) = get_integrals(gt, data.geom, flux_fields=True, spectral_df=False)
    Qp = ef[0].sum((0, 1, 2, 3)); Qg = efg[0].sum((0, 1, 2, 3))
    def rl(bd): s = slice(*bd); return ((Qp[s] - Qg[s]).abs().sum() / Qg[s].abs().sum().clamp_min(1e-12)).item()
    # eflux_int_rel: relative error of the INTEGRATED (scalar, signed) heat flux Q_tot = sum_ky Q(ky) --
    # the physical heat-flux value. Report it alongside flux_relL1 (the per-ky spectrum error); note the
    # scalar can sign-cancel, so a small eflux_int_rel does NOT imply the spectrum is right (and vice versa).
    eflux_int_rel = ((Qp.sum() - Qg.sum()).abs() / Qg.sum().abs().clamp_min(1e-12)).item()
    return {"PSNR(f)": 10 * log10(gt.max().item() ** 2 / il["df loss"].item()),
            "PSNR(phi)": 10 * log10(gphi.max().item() ** 2 / il["phi mse"].item()),
            "kyspec L1": sl["kyspec loss"].item(), "qspec L1": sl["qspec loss"].item(),
            "flux_dom": round(rl(band_dom), 4), "flux_tail": round(rl(band_tail), 4),
            "flux_relL1": round(rl((1, 16)), 4), "eflux_int_rel": round(eflux_int_rel, 4)}
