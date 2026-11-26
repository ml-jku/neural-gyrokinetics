"""A small synthetic snapshot: a few Gaussian blobs plus a couple of plane-wave modes, with
the same array layout as the real datasets.  Used by the tests and the quick-start."""
from __future__ import annotations

import torch

from neugk.gyrosplat.model.atoms import AtomBank, make_grid, subgrids, inv_softplus
from neugk.gyrosplat.data import Snapshot
from neugk.gyrosplat.model.synth import render, to_channels


def make_synthetic_snapshot(
    shape=(8, 4, 6, 16, 16),
    n_species: int = 1,
    n_blobs: int = 12,
    n_modes: int = 4,
    seed: int = 0,
    device="cpu",
    noise: float = 0.0,
) -> Snapshot:
    gen = torch.Generator().manual_seed(seed)
    grid = make_grid(shape, device)
    vel, phys = subgrids(grid, shape)
    fields = []
    for s in range(n_species):
        n = n_blobs + n_modes
        mu = torch.rand(n, 5, generator=gen).to(device)
        widths = torch.tensor(shape, dtype=torch.float32, device=device)
        Lp = torch.zeros(n, 6, device=device)
        Lv = torch.zeros(n, 3, device=device)
        Lp[:, :3] = inv_softplus(3.0 / widths[2:5])
        Lv[:, :2] = inv_softplus(2.5 / widths[0:2])
        Lp[n_blobs:, 1] = inv_softplus(12.0 / widths[3])  # wide envelopes for the modes
        Lp[n_blobs:, 2] = inv_softplus(12.0 / widths[4])
        amps = torch.randn(n, 2, generator=gen).to(device)
        ky = torch.zeros(n, device=device)
        ky[n_blobs:] = (
            2
            * torch.pi
            * torch.randint(1, 5, (n_modes,), generator=gen).float().to(device)
        )
        # the modes also carry a radial wavenumber, applied as a phase on the rendered mode
        kx = (
            2
            * torch.pi
            * torch.randint(-3, 4, (n_modes,), generator=gen).float().to(device)
        )
        bank = AtomBank(mu, Lp, Lv, amps, ky, shape=shape).to(device)
        f = to_channels(render(bank.subset(torch.arange(n_blobs)), vel, phys), shape)
        x = grid[:, 3].reshape(shape)
        for j in range(n_modes):
            one = bank.subset(torch.tensor([n_blobs + j]))
            g = to_channels(render(one, vel, phys), shape)
            phase = kx[j] * (x - one.mu.data[0, 3])
            c, s_ = torch.cos(phase), torch.sin(phase)
            f = f + torch.stack([g[0] * c - g[1] * s_, g[0] * s_ + g[1] * c], 0)
        if noise > 0:
            f = f + noise * torch.randn(f.shape, generator=gen).to(device)
        fields.append(f.contiguous())
    return Snapshot(fields, name=f"synthetic_{seed}", meta={"shape": tuple(shape)})
