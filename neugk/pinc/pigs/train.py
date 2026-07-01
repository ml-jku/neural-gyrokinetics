"""Training stages (self-contained):
  train_default : VANILLA Gaussian splatting baseline -- joint AdamW MSE on minibatches, no warm-start,
                  no separable trick (the "base" rung; slow, for reference)
  train_fast    : density fit = amp warm-start -> separable warmup (Huber) -> short fused polish
  train_pinc    : physics fine-tune (df/phi/spectra); mode='gpinc' (separable, ~10x faster) or 'dense'
  refine_flux   : frozen-base flux refine (the POST step): 'raw' (min flux) | 'wnorm' (phi-preserving tail)

PINC losses: df, flux, phi, kyspec, qspec, kyspec monotonicity, qspec monotonicity (all from neugk.pinc.neural_fields).
Only the shared neugk data + physics losses are external; nothing depends on prior iterations.
"""
import copy
import time
from math import log10

import numpy as np
import torch
import torch.nn.functional as F

from neugk.pinc.neural_fields import CycloneNFDataLoader, sample_field, integral_losses, spectra_losses
from neugk.physics.integrals import get_integrals
from neugk.pinc.neural_fields.nf_utils import to_complex

from neugk.pinc.pigs.model import build_gs_5d
from neugk.pinc.pigs.fast import (
    solve_amplitudes, fast_forward, build_subgrids, grid_target, sep_field, sep_sample_field, gabor_denorm,
    gabor_basis,
)

PINC_LW = {"df": 1.0, "flux": 1.0, "phi": 1.0, "kyspec": 1.0, "qspec": 1.0,
           "kyspec monotonicity": 1.0, "qspec monotonicity": 1.0}
_PARAM = {"amps": "amps", "ky": "ky", "mu": "mu", "L_phys": "L_phys_raw", "L_vel": "L_vel_raw"}


class _MAStop:
    """early stop when the moving average of the loss stops improving. quits once the windowed MA has not
    set a new minimum for `patience` checks. Takes the loss TENSOR (not .item()) and syncs only once per
    `window` steps -- a per-step .item() would serialize the otherwise-async GPU loop and cost more than
    the steps it saves. patience=0 disables (no sync at all)."""
    def __init__(self, window=25, patience=25, tol=0.0):
        self.window, self.patience, self.tol = window, patience, tol
        self.buf, self.best, self.since, self.n = [], float("inf"), 0, 0

    def step(self, loss):
        if self.patience <= 0:
            return False
        self.buf.append(loss.detach()); self.n += 1
        if self.n % self.window:           # only evaluate (and sync) once per window
            return False
        ma = torch.stack(self.buf).mean().item()
        self.buf = []
        if ma < self.best - self.tol:
            self.best, self.since = ma, 0
        else:
            self.since += 1
        return self.since * self.window >= self.patience  # `patience` is in steps (~last-N-epochs)


def _loss_fn(name):
    """'huber' (default) is robust to high-amplitude zonal outliers MSE over-weights, so it preserves
    the subdominant flux/spectrum modes (+~2-4 dB phi, better flux at ~-0.5 dB PSNR(f))."""
    if name == "huber":
        return lambda p, t: F.smooth_l1_loss(p, t, beta=0.1)
    if name == "mse":
        return F.mse_loss
    if name == "l1":
        return F.l1_loss
    raise ValueError(f"unknown loss: {name}")


def train_default(data, n, device, epochs=120, batch=200_000, seed=0):
    """VANILLA Gaussian splatting (the 'base' rung): importance-sampled init, then JOINT AdamW MSE on
    shuffled voxel minibatches. No amplitude warm-start, no separable contraction, no fused polish --
    this is the plain 5D-Gaussian fit we started from, kept as the slow reference baseline."""
    torch.set_float32_matmul_precision("high")
    model = build_gs_5d(data, n, device, seed=seed)
    loader = CycloneNFDataLoader(data, batch_size=batch, preload=True, shuffle=True)
    opt = torch.optim.AdamW([{"params": [model.mu], "lr": 5e-4},
                             {"params": [model.L_phys_raw, model.L_vel_raw], "lr": 3e-3},
                             {"params": [model.amps], "lr": 1e-2}], weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-7)
    fss = np.linspace(0.3, 1.0, epochs)
    t0 = time.time()
    for e in range(epochs):
        loader.subsample = fss[e]
        for f, coords in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(model(coords), f.to(torch.bfloat16))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    torch.cuda.synchronize()
    return model, time.time() - t0


def train_fast(data, n, device, warmup_steps=1000, polish_epochs=2, batch=200_000, seed=0,
               loss="huber", vel_grid=None, phys_grid=None, patience=25):
    """amp warm-start -> separable warmup (exact ~17ms steps) -> short fused polish.
    patience: early-stop the separable warmup when its loss MA plateaus for that many steps (0 disables)."""
    torch.set_float32_matmul_precision("high")
    lf = _loss_fn(loss)
    model = build_gs_5d(data, n, device, seed=seed)
    if vel_grid is None:
        vel_grid, phys_grid, Nv, Np = build_subgrids(data, device)
    else:
        Nv, Np = vel_grid.shape[0], phys_grid.shape[0]
    Y = grid_target(data, Nv, Np, device)
    t0 = time.time()
    solve_amplitudes(model, data.f_grid, data.f_df)
    opt = torch.optim.AdamW([{"params": [model.mu], "lr": 1e-3},
                             {"params": [model.L_phys_raw, model.L_vel_raw], "lr": 5e-3},
                             {"params": [model.amps], "lr": 1e-2}], weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, warmup_steps, 1e-6)
    stop = _MAStop(patience=patience)
    for _ in range(warmup_steps):
        loss_v = lf(sep_field(model, vel_grid, phys_grid), Y)
        opt.zero_grad(set_to_none=True); loss_v.backward(); opt.step(); sched.step()
        if stop.step(loss_v):
            break
    loader = CycloneNFDataLoader(data, batch_size=batch, preload=True, shuffle=True)
    opt = torch.optim.AdamW([{"params": [model.mu], "lr": 2e-4},
                             {"params": [model.L_phys_raw, model.L_vel_raw], "lr": 1e-3},
                             {"params": [model.amps], "lr": 5e-3}], weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(polish_epochs, 1), 1e-7)
    fss = np.linspace(0.5, 1.0, max(polish_epochs, 1))
    for e in range(polish_epochs):
        loader.subsample = fss[e]
        for f, coords in loader:
            loss_v = lf(fast_forward(model, coords, compile=True), f.to(torch.float32))
            opt.zero_grad(set_to_none=True); loss_v.backward(); opt.step()
        sched.step()
    torch.cuda.synchronize()
    return model, time.time() - t0


def train_pinc(model, data, device, epochs=40, mode="gpinc", vel_grid=None, phys_grid=None):
    """Physics fine-tune with ALL PINC losses: df, flux, phi, kyspec, qspec, monotonicity.
    mode='dense' uses sample_field; 'gpinc' uses the separable reconstruction (bit-identical, ~10x faster)."""
    torch.set_float32_matmul_precision("high")
    mp = copy.deepcopy(model)
    if mode == "dense":
        mp.use_checkpoint = True; mp.chunk = min(128, mp.chunk)
    elif mode == "gpinc" and vel_grid is None:
        vel_grid, phys_grid, _, _ = build_subgrids(data, device)
    opt = torch.optim.AdamW([{"params": [mp.mu], "lr": 5e-5},
                             {"params": [mp.L_phys_raw, mp.L_vel_raw], "lr": 3e-4},
                             {"params": [mp.amps], "lr": 1e-3}], weight_decay=1e-12)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-7)
    gt = data.full_df.to(device)
    best, best_model = -1e9, None
    t0 = time.time()
    for e in range(epochs):
        pred = (sep_sample_field(mp, data, vel_grid, phys_grid) if mode == "gpinc"
                else sample_field(mp, data, device, timestep=None)).to(device)
        il, (pphi, gphi), (pef, gef) = integral_losses(pred, gt, geom=data.geom, device=device,
                                                       use_flux_fields=False, timestep=None, return_fields=True)
        sl, _ = spectra_losses(pred_df=pred, pred_phi=pphi, pred_eflux=pef,
                               gt_df=gt, gt_phi=gphi, gt_eflux=gef, ds=data.ds)
        loss = sum(PINC_LW[k] * il[f"{k} loss"] for k in PINC_LW if f"{k} loss" in il) \
            + sum(PINC_LW[k] * sl[f"{k} loss"] for k in PINC_LW if f"{k} loss" in sl)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if e % 2 == 0 or e == epochs - 1:
            phi = 10 * log10(gphi.max().item() ** 2 / il["phi mse"].item())
            if phi > best:
                best, best_model = phi, copy.deepcopy(mp)
    torch.cuda.synchronize()
    return (best_model or mp), time.time() - t0


def refine_flux(model, data, device, mask=None, train=("amps", "ky"), flux_mode="raw",
                steps=500, lr=1e-2, flux_lambda=3.0, band=(1, 16), phi_floor=2e-2, patience=25):
    """Frozen-base flux refine. Objective = df-MSE anchor + flux_lambda*(flux term)/ft0.
      'raw'   -> match Q(ky) directly         (fixes BOTH bands, spends phi)
      'wnorm' -> match W(ky)=Q/|phi_k|^2       (quasilinear weight: phi-preserving, fixes the TAIL)
    `mask` freezes everything except the new atoms' `train` params."""
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    params = [getattr(m, _PARAM[k]) for k in train]
    for p in params:
        p.requires_grad_(True)
    vg, pg, _, _ = build_subgrids(data, device); gt = data.full_df.to(device); b = slice(*band)

    # amps-only POST freezes ALL geometry (mu/Sigma/ky), so the per-atom field basis is constant and the
    # render is linear in the complex amps: cache V,Pc once and replace 500 full exp/polar renders with a
    # single GEMM per step (bit-identical to gabor_denorm, ~10x cheaper -- the dominant cost of compress_pigs).
    amps_only = tuple(train) == ("amps",)
    if amps_only:
        with torch.no_grad():
            _V, _Pc = gabor_basis(m, vg, pg)
            _Vt = _V.t().to(torch.complex64).contiguous()
        _nvp, _nmu, _ns, _nx, _ny = data.df.shape[1:]
        _scale = data.scale["df"].reshape(2, 1, 1, 1, 1, 1).to(device)
        _shift = data.shift["df"].reshape(2, 1, 1, 1, 1, 1).to(device)

        def render():
            c = torch.complex(m.amps[:, 0], m.amps[:, 1])
            dfc = _Vt @ (_Pc * c.unsqueeze(1))
            f = torch.stack([dfc.real, dfc.imag], -1).reshape(_nvp, _nmu, _ns, _nx, _ny, 2)
            return f.permute(5, 0, 1, 2, 3, 4).contiguous() * _scale + _shift
    else:
        def render():
            return gabor_denorm(m, data, vg, pg)

    def spectra(field):
        phi, (_, ef, _) = get_integrals(field, data.geom, flux_fields=True, spectral_df=False)
        Q = ef[0].sum((0, 1, 2, 3))
        _z = torch.fft.fft(to_complex(phi), dim=-1, norm="forward")
        Pphi = (_z.real ** 2 + _z.imag ** 2).sum((0, 1))  # |z|^2 without complex abs (B300 nvrtc-safe)
        return Q, Pphi

    with torch.no_grad():
        Qg, Pphig = spectra(gt); pfl = phi_floor * Pphig.max(); Qsc = Qg.abs().max().clamp_min(1e-12)
        Wg = Qg / (Pphig + pfl)

    def fterm(Q, Pphi):
        return (((Q[b] - Qg[b]) / Qsc) ** 2).sum() if flux_mode == "raw" else ((Q[b] / (Pphi[b] + pfl) - Wg[b]) ** 2).sum()

    with torch.no_grad():
        ft0 = fterm(*spectra(render())).clamp_min(1e-12)
    opt = torch.optim.Adam(params, lr=lr)
    stop = _MAStop(patience=patience)
    for _ in range(steps):
        pred = render()
        loss = ((pred - gt) ** 2).mean() + flux_lambda * fterm(*spectra(pred)) / ft0
        opt.zero_grad(set_to_none=True); loss.backward()
        if mask is not None:
            for p in params:
                if p.grad is not None and p.shape[0] == mask.shape[0]:
                    p.grad.mul_(mask.view(-1, *([1] * (p.dim() - 1))))
        opt.step()
        if stop.step(loss):
            break
    torch.cuda.synchronize()
    return m
