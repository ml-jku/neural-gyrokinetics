"""Placing the base atoms.

`random_init` scatters the centres uniformly (the textbook start).  `field_init` draws
centres with probability proportional to |f|^2, seeds each amplitude with the field value
at its centre and gives every atom the same width.
"""
from __future__ import annotations

import torch

from neugk.gyrosplat.model.atoms import AtomBank, inv_softplus


def weighted_sample(p: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """k distinct indices drawn with probability proportional to p (inverse CDF with de-duplication;
    torch.multinomial is capped at 2^24 categories and grids are larger)."""
    c = torch.cumsum(p.double(), 0)
    tot = c[-1]
    picked = torch.empty(0, dtype=torch.long, device=p.device)
    for _ in range(40):
        if picked.numel() >= k:
            break
        m = int((k - picked.numel()) * 1.3) + 32
        u = (torch.rand(m, generator=gen) * float(tot)).to(p.device).double()
        idx = torch.searchsorted(c, u).clamp_(max=p.numel() - 1)
        picked = torch.unique(torch.cat([picked, idx]))
    perm = torch.randperm(picked.numel(), generator=gen).to(p.device)
    return picked[perm[:k]]


def _constant_widths(n, shape, sigma_cells, device):
    sig = sigma_cells / torch.tensor(shape, dtype=torch.float32, device=device)
    Lp = torch.zeros(n, 6, device=device)
    Lp[:, :3] = inv_softplus(sig[2:5])
    Lv = torch.zeros(n, 3, device=device)
    Lv[:, :2] = inv_softplus(sig[0:2])
    return Lp, Lv


def random_init(
    n: int, shape, device, seed: int = 0, sigma_cells: float = 2.0
) -> AtomBank:
    """Centres uniform over the domain, constant widths, small random amplitudes."""
    gen = torch.Generator().manual_seed(seed)
    mu = torch.rand(n, 5, generator=gen).to(device)
    amps = (0.01 * torch.randn(n, 2, generator=gen)).to(device)
    Lp, Lv = _constant_widths(n, shape, sigma_cells, device)
    return AtomBank(mu, Lp, Lv, amps, shape=shape).to(device)


def field_init(
    n: int,
    target: torch.Tensor,
    grid: torch.Tensor,
    shape,
    seed: int = 0,
    sigma_cells: float = 2.0,
) -> AtomBank:
    """target (n_v, n_s, 2) on the tensor-product grid (n_voxel, 5)."""
    device = target.device
    gen = torch.Generator().manual_seed(seed)
    p = (target**2).sum(-1).reshape(-1)
    idx = weighted_sample(p, n, gen)
    mu = grid[idx].clone()
    amps = target.reshape(-1, 2)[idx].clone()
    Lp, Lv = _constant_widths(n, shape, sigma_cells, device)
    return AtomBank(mu, Lp, Lv, amps, shape=shape).to(device)
