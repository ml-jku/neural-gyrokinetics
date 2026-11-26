"""Grafting grouped Gabor carriers onto a fitted base bank.

A Gaussian atom has its power at ky = 0; the flux lives at ky > 0.  Carriers are atoms with
a plane-wave phase that put their power on one ky mode. Each group of `tied_k` carriers
starts with equal narrow envelopes, placed by stratified coverage at the flux tail.
By default these are independent parameters and can diverge during fitting. With
share_envelopes enabled, there is one trainable geometry row per group throughout
fitting, rendering and storage; individual amplitudes and frequencies remain free.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from neugk.gyrosplat.model.atoms import AtomBank, concat, inv_softplus
from neugk.gyrosplat.config import CarrierConfig


def _carrier_widths(n, shape, sx, sy, sv, device):
    """Cholesky parameters for diagonal envelopes of the given widths (in cells)."""
    cells = torch.tensor(shape, dtype=torch.float32, device=device)
    sig_p = torch.tensor([sv, sx, sy], device=device) / cells[2:5]  # (s1, x, y)
    sig_v = torch.tensor([sv, sv], device=device) / cells[0:2]
    Lp = torch.zeros(n, 6, device=device)
    Lp[:, :3] = inv_softplus(sig_p)
    Lv = torch.zeros(n, 3, device=device)
    Lv[:, :2] = inv_softplus(sig_v)
    return Lp, Lv


def stratified_centres(grid: torch.Tensor, m: int, seed: int = 2) -> torch.Tensor:
    """One coordinate per equal chunk of the flattened grid."""
    gen = torch.Generator(device=grid.device).manual_seed(seed)
    chunk = grid.shape[0] // m
    offs = torch.randint(0, max(chunk, 1), (m,), generator=gen, device=grid.device)
    idx = torch.arange(m, device=grid.device) * chunk + offs
    return grid[idx.clamp_max(grid.shape[0] - 1)].float()


def tied_budget(
    n_equiv: int, fraction: float, tied_k: int, reals_per_atom: int = 17
) -> Tuple[int, int]:
    """Historical allocation returning (n_base, n_groups), preserved for reproducibility.

    The 14+3k group cost is attained with share_envelopes enabled. Otherwise this
    remains the historical allocation rule, while each carrier costs 17 reals.
    Actual storage is computed separately from the fitted representation.
    """
    n_flux = int(round(fraction * n_equiv))
    per_group = 14 + 3 * tied_k
    m = (n_flux * reals_per_atom) // per_group
    n_base = (
        n_equiv - n_flux + (n_flux * reals_per_atom - m * per_group) // reals_per_atom
    )
    return n_base, m


def graft_tied_gabor(
    base: AtomBank,
    grid: torch.Tensor,
    m_groups: int,
    cfg: CarrierConfig,
    tail_bins: Sequence[int],
    seed: int = 0,
) -> AtomBank:
    """Append m_groups ky-carrier groups: each starts with equal envelopes (narrow wave packets along
    the carrier axis, placed by stratified coverage of the grid) with cfg.tied_k carriers
    cycling through `tail_bins`. share_envelopes controls whether geometry remains shared."""
    device = base.mu.device
    shape = base.shape
    k = cfg.tied_k
    m = int(m_groups)
    n_carr = m * k
    centres = stratified_centres(grid.to(device), m, seed=seed + 1).repeat_interleave(
        k, dim=0
    )
    cycle = torch.tensor(
        [tail_bins[i % len(tail_bins)] for i in range(k)], device=device
    ).float()
    ky = 2.0 * np.pi * cycle.repeat(m)
    Lp, Lv = _carrier_widths(n_carr, shape, 2.0, cfg.sigma_y_cells, 2.0, device)
    extra = AtomBank(
        centres, Lp, Lv, torch.zeros(n_carr, 2, device=device), ky, shape=shape
    )
    bank = concat(base, extra)
    return bank.share_carrier_envelopes(k) if cfg.share_envelopes else bank
