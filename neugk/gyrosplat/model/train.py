"""The descent stages: fit the atom geometry and amplitudes.

`descend` is a plain AdamW loop with a cosine schedule on the block-wise Huber (or MSE)
loss between the rendered field and the normalised target.

Two stages use the backend's physics instead of the field: `physics_fit` descends on every
atom parameter against the backend's physics terms (field, potential, flux) evaluated on
the de-normalised field, and `routed_cofit` trains base and carriers concurrently with each
parameter row hearing one objective only: the physics terms move the base, the flux term
moves the carriers.  `gradient_ratio` is the flag that decides how long the routed co-fit
runs: on snapshots where the potential is placement-bound the geometry needs more steps.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from neugk.gyrosplat.model.atoms import AtomBank
from neugk.gyrosplat.config import CarrierConfig, FitConfig, PhysicsFitConfig
from neugk.gyrosplat.model.normalize import NormStats, denormalize
from neugk.gyrosplat.model.synth import blocks, _synth_block, to_channels


def _loss(pred, target, kind: str, delta: float):
    if kind == "mse":
        return F.mse_loss(pred, target)
    return F.huber_loss(pred, target, delta=delta)


def _groups(banks: Sequence[AtomBank], lrs, scale: float, with_ky: bool = False):
    groups = [
        {"params": [b.mu for b in banks], "lr": lrs[0] * scale},
        {
            "params": [p for b in banks for p in (b.L_phys_raw, b.L_vel_raw)],
            "lr": lrs[1] * scale,
        },
        {"params": [b.amps for b in banks], "lr": lrs[2] * scale},
    ]
    if with_ky:
        groups.append({"params": [b.ky for b in banks], "lr": lrs[3] * scale})
    return groups


def descend(
    bank: AtomBank,
    target: torch.Tensor,
    vel_grid: torch.Tensor,
    phys_grid: torch.Tensor,
    cfg: FitConfig,
    steps: Optional[int] = None,
    lr_scale: Optional[float] = None,
    log_every: int = 0,
) -> AtomBank:
    """AdamW on (mu, L, amps) against target (n_v, n_s, 2), then `cfg.polish_steps` at low rates."""
    steps = cfg.steps if steps is None else steps
    scale = cfg.lr_scale if lr_scale is None else lr_scale
    if cfg.schedule == "flat":
        params = [bank.mu, bank.L_phys_raw, bank.L_vel_raw, bank.amps]
        opt = torch.optim.Adam(params, lr=cfg.flat_lr)
        sched = None
    else:
        opt = torch.optim.AdamW(
            _groups([bank], (1e-3, 5e-3, 1e-2), scale), weight_decay=1e-8
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), 1e-6)
    n_s = phys_grid.shape[0]
    n_blocks = (n_s + cfg.chunk - 1) // cfg.chunk

    def step_once(optimizer):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for sl in blocks(n_s, cfg.chunk):
            V, Pre, Pim = bank.factors(vel_grid, phys_grid[sl])
            real, imag = _synth_block(V, Pre, Pim, bank.amps[:, 0], bank.amps[:, 1])
            loss = (
                _loss(
                    torch.stack([real, imag], -1),
                    target[:, sl],
                    cfg.loss,
                    cfg.huber_delta,
                )
                / n_blocks
            )
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        return total

    for step in range(1, steps + 1):
        total = step_once(opt)
        if sched is not None:
            sched.step()
        if log_every and step % log_every == 0:
            print(f"    step {step}/{steps} loss {total:.4e}", flush=True)
    if cfg.polish_steps > 0:
        polish = torch.optim.AdamW(
            _groups([bank], (2e-4, 1e-3, 5e-3), scale), weight_decay=1e-8
        )
        for _ in range(cfg.polish_steps):
            step_once(polish)
    return bank


# ------------------------------------------------------------------------------ physics stages
def rendered_fields(
    banks: Sequence[AtomBank],
    stats: Sequence[NormStats],
    vel_grids,
    phys_grids,
    chunk: int = 16384,
) -> List[torch.Tensor]:
    """Per species, the de-normalised (2, ...) field with gradients on every atom parameter."""
    out = []
    for bank, st, vel, phys in zip(banks, stats, vel_grids, phys_grids):
        V = bank.velocity_factor(vel)
        parts = []
        for sl in blocks(phys.shape[0], chunk):
            Pre, Pim = bank.spatial_factors(phys[sl])
            real, imag = _synth_block(V, Pre, Pim, bank.amps[:, 0], bank.amps[:, 1])
            parts.append(torch.stack([real, imag], -1))
        out.append(denormalize(to_channels(torch.cat(parts, dim=1), bank.shape), st))
    return out


def _physics_loss(terms: dict, w_field: float, w_potential: float, w_flux: float):
    loss = w_field * terms["field"] + w_potential * terms["potential"]
    if w_flux and "flux" in terms:
        loss = loss + w_flux * terms["flux"]
    return loss


def physics_fit(
    banks: Sequence[AtomBank],
    stats: Sequence[NormStats],
    vel_grids,
    phys_grids,
    backend,
    cfg: PhysicsFitConfig,
    chunk: int = 16384,
    log=None,
) -> None:
    """Descend on every atom parameter of every species against the backend's physics terms."""
    if cfg.epochs <= 0:
        return
    opt = torch.optim.AdamW(
        _groups(banks, (5e-5, 3e-4, 1e-3), cfg.lr_scale), weight_decay=1e-12
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, 1e-7)
    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)
        terms = backend.physics_terms(
            rendered_fields(banks, stats, vel_grids, phys_grids, chunk),
            with_flux=cfg.flux_weight > 0,
        )
        loss = _physics_loss(
            terms, cfg.field_weight, cfg.potential_weight, cfg.flux_weight
        )
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (
            epoch % max(1, cfg.epochs // 4) == 0 or epoch == cfg.epochs
        ):
            log(f"    physics fit {epoch}/{cfg.epochs}  loss {loss.item():.4e}")


def gradient_ratio(
    banks: Sequence[AtomBank],
    stats: Sequence[NormStats],
    vel_grids,
    phys_grids,
    backend,
    chunk: int = 16384,
) -> float:
    """|d potential / d geometry| over |d field / d geometry|: how much harder the potential is to
    move than the field on this snapshot (large on zonal-dominated snapshots)."""
    geom = [p for b in banks for p in (b.mu, b.L_phys_raw, b.L_vel_raw)]
    terms = backend.physics_terms(
        rendered_fields(banks, stats, vel_grids, phys_grids, chunk), with_flux=False
    )
    g_pot = torch.autograd.grad(
        terms["potential"], geom, retain_graph=True, allow_unused=True
    )
    g_field = torch.autograd.grad(terms["field"], geom, allow_unused=True)
    n_pot = sum(float(g.norm()) for g in g_pot if g is not None)
    n_field = sum(float(g.norm()) for g in g_field if g is not None)
    return n_pot / max(n_field, 1e-12)


def routed_cofit(
    banks: Sequence[AtomBank],
    stats: Sequence[NormStats],
    vel_grids,
    phys_grids,
    backend,
    cfg: CarrierConfig,
    epochs: int,
    chunk: int = 16384,
    log=None,
) -> None:
    """Concurrent base + carrier training with per-row gradient routing.

        physics terms (field + potential)  ->  base rows      (geometry, amplitudes)
        flux term                          ->  carrier rows   (geometry, amplitudes, ky)

    Two backward passes per epoch; each parameter row hears exactly one objective.  The
    base atoms keep ky = 0."""
    if epochs <= 0 or not any(b.n_carriers > 0 for b in banks):
        return
    for b in banks:
        b.ky.requires_grad_(True)
    params = [p for b in banks for p in (b.mu, b.L_phys_raw, b.L_vel_raw, b.amps, b.ky)]
    masks = [
        m
        for b in banks
        for m in (
            b.geometry_carrier_mask,
            b.geometry_carrier_mask,
            b.geometry_carrier_mask,
            b.carrier_mask,
            b.carrier_mask,
        )
    ]
    opt = torch.optim.AdamW(
        _groups(banks, (5e-5, 3e-4, 3e-3, 3e-3), 1.0, with_ky=True), weight_decay=1e-12
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1), 1e-7)
    try:
        for epoch in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            terms = backend.physics_terms(
                rendered_fields(banks, stats, vel_grids, phys_grids, chunk),
                with_flux=False,
            )
            (terms["field"] + terms["potential"]).backward()
            g_phys = [
                None if p.grad is None else p.grad.detach().clone() for p in params
            ]
            opt.zero_grad(set_to_none=True)
            terms = backend.physics_terms(
                rendered_fields(banks, stats, vel_grids, phys_grids, chunk),
                with_flux=True,
            )
            (cfg.routed_flux_weight * terms["flux"]).backward()
            g_flux = [
                None if p.grad is None else p.grad.detach().clone() for p in params
            ]
            opt.zero_grad(set_to_none=True)
            for i, (p, m) in enumerate(zip(params, masks)):
                gp, gf = g_phys[i], g_flux[i]
                if gp is None and gf is None:
                    continue
                g = torch.zeros_like(p)
                mm = m if g.dim() == 1 else m[:, None]
                if gp is not None:
                    g = torch.where(mm, g, gp)
                if gf is not None:
                    g = torch.where(mm, gf, g)
                if i % 5 == 4:  # ky: base rows stay at zero
                    g = torch.where(mm, g, torch.zeros_like(g))
                p.grad = g
            opt.step()
            sched.step()
            if log is not None and (
                epoch % max(1, epochs // 4) == 0 or epoch == epochs
            ):
                log(
                    f"    routed co-fit {epoch}/{epochs}  flux {terms['flux'].item():.4f}"
                )
    finally:
        for b in banks:
            b.ky.requires_grad_(False)
            b.ky.data[~b.carrier_mask] = 0.0
