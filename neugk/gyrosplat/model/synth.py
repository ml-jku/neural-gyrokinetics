"""Rendering a bank on the grid.

The rendered field is  f(v, s) = sum_n V_n(v) [ a_n Pre_n(s) - b_n Pim_n(s) ]  for the real
channel and  sum_n V_n(v) [ a_n Pim_n(s) + b_n Pre_n(s) ]  for the imaginary one, with
c_n = a_n + i b_n.  Two GEMMs per block of spatial rows, never a per-atom loop.  Because the
map amplitudes -> field is linear at fixed geometry, `render_linear` is an autograd
function whose backward is the adjoint of those GEMMs, recomputed block by block: the
render stores no activations, so the physics stage can differentiate through it on the
full grid.
"""
from __future__ import annotations

import torch

from neugk.gyrosplat.model.atoms import AtomBank


def blocks(n_rows: int, chunk: int):
    for i in range(0, n_rows, chunk):
        yield slice(i, min(i + chunk, n_rows))


def _synth_block(V, Pre, Pim, a, b):
    real = torch.einsum("vi,pi->vp", V, Pre * a - Pim * b)
    imag = torch.einsum("vi,pi->vp", V, Pim * a + Pre * b)
    return real, imag


@torch.no_grad()
def render(
    bank: AtomBank, vel_grid: torch.Tensor, phys_grid: torch.Tensor, chunk: int = 16384
) -> torch.Tensor:
    """The field on the separable grid, (n_v, n_s, 2)."""
    n_s = phys_grid.shape[0]
    out = vel_grid.new_zeros(vel_grid.shape[0], n_s, 2)
    a, b = bank.amps[:, 0], bank.amps[:, 1]
    V = bank.velocity_factor(vel_grid)
    for sl in blocks(n_s, chunk):
        Pre, Pim = bank.spatial_factors(phys_grid[sl])
        out[:, sl, 0], out[:, sl, 1] = _synth_block(V, Pre, Pim, a, b)
    return out


class _RenderLinear(torch.autograd.Function):
    """amps (N, 2) -> field (n_v, n_s, 2) at frozen geometry; backward = the exact adjoint."""

    @staticmethod
    def forward(ctx, amps, bank, vel_grid, phys_grid, chunk):
        ctx.bank, ctx.vel, ctx.phys, ctx.chunk = bank, vel_grid, phys_grid, chunk
        n_s = phys_grid.shape[0]
        out = amps.new_zeros(vel_grid.shape[0], n_s, 2)
        a, b = amps[:, 0], amps[:, 1]
        with torch.no_grad():
            V = bank.velocity_factor(vel_grid)
            for sl in blocks(n_s, chunk):
                Pre, Pim = bank.spatial_factors(phys_grid[sl])
                out[:, sl, 0], out[:, sl, 1] = _synth_block(V, Pre, Pim, a, b)
        return out

    @staticmethod
    def backward(ctx, g):
        bank, vel, phys, chunk = ctx.bank, ctx.vel, ctx.phys, ctx.chunk
        ga = g.new_zeros(bank.N)
        gb = g.new_zeros(bank.N)
        with torch.no_grad():
            V = bank.velocity_factor(vel)
            for sl in blocks(phys.shape[0], chunk):
                Pre, Pim = bank.spatial_factors(phys[sl])
                Y0 = V.T @ g[:, sl, 0].contiguous()
                Y1 = V.T @ g[:, sl, 1].contiguous()
                ga += (Y0 * Pre.T).sum(1) + (Y1 * Pim.T).sum(1)
                gb += (Y1 * Pre.T).sum(1) - (Y0 * Pim.T).sum(1)
        return torch.stack([ga, gb], dim=1), None, None, None, None


def render_linear(
    amps: torch.Tensor,
    bank: AtomBank,
    vel_grid: torch.Tensor,
    phys_grid: torch.Tensor,
    chunk: int = 16384,
) -> torch.Tensor:
    """Differentiable in `amps` only (geometry frozen)."""
    return _RenderLinear.apply(amps, bank, vel_grid, phys_grid, chunk)


def to_channels(field_sep: torch.Tensor, shape) -> torch.Tensor:
    """(n_v, n_s, 2) -> (2, *shape)."""
    return field_sep.permute(2, 0, 1).reshape((2,) + tuple(shape))


def to_separable(field: torch.Tensor) -> torch.Tensor:
    """(2, n_v1, n_v2, n_s1, n_s2, n_s3) -> (n_v, n_s, 2)."""
    shape = field.shape[1:]
    n_v = shape[0] * shape[1]
    n_s = shape[2] * shape[3] * shape[4]
    return field.reshape(2, n_v, n_s).permute(1, 2, 0).contiguous()


@torch.no_grad()
def residual_power(
    bank: AtomBank, target: torch.Tensor, vel_grid, phys_grid, chunk: int = 16384
) -> float:
    """Mean squared residual over both channels and all grid points, without materialising the field."""
    n_s = phys_grid.shape[0]
    se = torch.zeros((), device=phys_grid.device, dtype=torch.float64)
    a, b = bank.amps[:, 0], bank.amps[:, 1]
    V = bank.velocity_factor(vel_grid)
    for sl in blocks(n_s, chunk):
        Pre, Pim = bank.spatial_factors(phys_grid[sl])
        real, imag = _synth_block(V, Pre, Pim, a, b)
        se += ((real - target[:, sl, 0]) ** 2 + (imag - target[:, sl, 1]) ** 2).sum(
            dtype=torch.float64
        )
    return float(se / target[..., 0].numel())


def target_power(target: torch.Tensor) -> float:
    """The same normalisation as `residual_power`, for the field-fit SNR."""
    return float(
        (target**2).sum(-1).sum(dtype=torch.float64) / target[..., 0].numel()
    )


def fit_snr_db(
    bank: AtomBank, target: torch.Tensor, vel_grid, phys_grid, chunk: int = 16384
) -> float:
    import math

    return 10 * math.log10(
        target_power(target)
        / max(residual_power(bank, target, vel_grid, phys_grid, chunk), 1e-300)
    )
