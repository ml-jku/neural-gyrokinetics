"""The amplitude solve (variable projection).

At fixed geometry the field is linear in the complex amplitudes, so the least-squares
amplitudes solve a 2N x 2N normal system whose Gram matrix is assembled from small per-block
Grams of the separable factors, never from the voxel-sized design matrix.  The Gram and the
solve run in float64: on large grids with heavily overlapping atoms a float32 Gram gives
amplitudes worse than zero.  With a per-column ridge the regularisation scales with each
atom's own diagonal, which keeps wide carriers from crushing narrow base atoms.

Two extensions of the same system: a solve restricted to a subset of atoms (against the
residual the others leave, used for freshly grafted carriers), and extra linear rows, e.g.
a potential summary that is itself linear in the amplitudes, so that the field fit and
the potential are solved for jointly in closed form.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch

from neugk.gyrosplat.model.atoms import AtomBank
from neugk.gyrosplat.model.synth import blocks, render


def _normal_equations(
    bank: AtomBank,
    target: torch.Tensor,
    vel_grid: torch.Tensor,
    phys_grid: torch.Tensor,
    chunk: int,
    dtype,
):
    """(A, rhs) of the least-squares system in the [a; b] ordering of the real amplitudes."""
    N, n_s = bank.N, phys_grid.shape[0]
    dev = phys_grid.device
    Gpp = torch.zeros(N, N, device=dev, dtype=dtype)
    Gmm = torch.zeros(N, N, device=dev, dtype=dtype)
    Gpm = torch.zeros(N, N, device=dev, dtype=dtype)
    ra = torch.zeros(N, device=dev, dtype=dtype)
    rb = torch.zeros(N, device=dev, dtype=dtype)
    V = bank.velocity_factor(vel_grid).to(dtype)
    for sl in blocks(n_s, chunk):
        Pre32, Pim32 = bank.spatial_factors(phys_grid[sl])
        Pre, Pim = Pre32.to(dtype), Pim32.to(dtype)
        Gpp += Pre.T @ Pre
        Gmm += Pim.T @ Pim
        Gpm += Pre.T @ Pim
        Y0 = V.T @ target[:, sl, 0].to(dtype)
        Y1 = V.T @ target[:, sl, 1].to(dtype)
        ra += (Y0 * Pre.T).sum(1) + (Y1 * Pim.T).sum(1)
        rb += (Y1 * Pre.T).sum(1) - (Y0 * Pim.T).sum(1)
    Gv = V.T @ V
    A = torch.zeros(2 * N, 2 * N, device=dev, dtype=dtype)
    blk = Gv * (Gpp + Gmm)
    A[:N, :N] = blk
    A[N:, N:] = blk
    A[:N, N:] = Gv * (Gpm.T - Gpm)
    A[N:, :N] = A[:N, N:].T
    return A, torch.cat([ra, rb])


def _add_ridge(A: torch.Tensor, ridge: float, per_column: bool):
    d = A.diagonal()
    if per_column:
        floor = 1e-6 * d.abs().mean().clamp_min(1e-30)
        d.add_(ridge * d.abs().clamp_min(floor))
    else:
        d.add_(ridge * d.abs().mean().clamp_min(1e-30))


def _solve(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(A, rhs)
    except Exception:
        return torch.linalg.lstsq(A, rhs.unsqueeze(1)).solution.squeeze(1)


@torch.no_grad()
def solve_amplitudes(
    bank: AtomBank,
    target: torch.Tensor,
    vel_grid: torch.Tensor,
    phys_grid: torch.Tensor,
    ridge: float = 1e-3,
    per_column: bool = True,
    chunk: int = 4096,
    dtype=torch.float64,
    columns: Optional[torch.Tensor] = None,
    linear_terms: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
) -> AtomBank:
    """Overwrite bank.amps with the ridge least-squares amplitudes for `target` (n_v, n_s, 2).

    columns      : solve only these atoms, against the residual the other atoms leave at their
                   current amplitudes.
    linear_terms : [(M, t, weight)] adds weight * |M c - t|^2 for a summary M (m, 2N) in the
                   [a; b] ordering (see `summary_jacobian`); each term is balanced against the
                   field term by the ratio of the Gram diagonals, so weight 1 means "as much
                   as the field".
    """
    if columns is not None:
        keep = bank.amps.data.clone()
        bank.amps.data[columns] = 0.0
        rest = render(bank, vel_grid, phys_grid, chunk=max(chunk, 4096))
        bank.amps.data.copy_(keep)
        sub = bank.subset(columns)
        solve_amplitudes(
            sub,
            target - rest,
            vel_grid,
            phys_grid,
            ridge,
            per_column,
            chunk,
            dtype,
            None,
            linear_terms,
        )
        bank.amps.data[columns] = sub.amps.data
        return bank
    N = bank.N
    A, rhs = _normal_equations(bank, target, vel_grid, phys_grid, chunk, dtype)
    for M, t, weight in linear_terms or ():
        M = M.to(device=A.device, dtype=dtype)
        MtM = M.T @ M
        scale = (
            weight
            * A.diagonal().mean().clamp_min(1e-30)
            / MtM.diagonal().mean().clamp_min(1e-30)
        )
        A += scale * MtM
        rhs += scale * (M.T @ t.to(device=A.device, dtype=dtype))
    _add_ridge(A, ridge, per_column)
    c = _solve(A, rhs)
    bank.amps.data[:, 0] = c[:N].to(bank.amps.dtype)
    bank.amps.data[:, 1] = c[N:].to(bank.amps.dtype)
    return bank


def summary_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor], bank: AtomBank
) -> torch.Tensor:
    """The (m, 2N) Jacobian of a summary fn(amps (N, 2)) -> (m,) that is linear in the
    amplitudes, in the [a; b] column ordering of `solve_amplitudes`.  One backward pass per
    summary entry, so keep summaries small (a few dozen numbers)."""
    c0 = bank.amps.detach().clone()
    J = torch.autograd.functional.jacobian(fn, c0)  # (m, N, 2)
    return torch.cat([J[..., 0], J[..., 1]], dim=1).double()
