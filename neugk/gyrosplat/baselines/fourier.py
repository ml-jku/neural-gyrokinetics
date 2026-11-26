"""Fourier top-K: keep the K largest Fourier coefficients of each species' object.

The object is already spectral along the binormal axis.  With the object's remaining real
axes listed in `axes` the code keeps the largest coefficients of the full Fourier
transform, the paper's control; with `axes` empty it keeps the largest cells of the
spectral object as stored.
"""
from __future__ import annotations

from typing import Sequence

import torch

from neugk.gyrosplat.baselines.gn_on_code import LinearCode


def forward(X: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    """Orthonormal FFT over `axes` of one species' object."""
    return torch.fft.fftn(X, dim=list(axes), norm="ortho") if axes else X


def inverse(C: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    return torch.fft.ifftn(C, dim=list(axes), norm="ortho") if axes else C


def fourier_topk(H: torch.Tensor, budget_reals: int, axes: Sequence[int] = ()):
    """H (n_species, ...) complex.  Keeps budget_reals / 2 complex coefficients per species (the
    positions are not charged: the value-count convention of the top-K codes).
    Returns (H_hat, achieved reals per species, the code)."""
    k = budget_reals // 2
    shape = H.shape[1:]
    index, values = [], []
    for s in range(H.shape[0]):
        flat = forward(H[s], axes).reshape(-1)
        idx = torch.topk(flat.abs(), min(k, flat.numel())).indices
        index.append(idx)
        values.append(flat[idx].clone())

    def decode(vals):
        planes = []
        for idx, v in zip(index, vals):
            C = torch.zeros(shape.numel(), dtype=v.dtype, device=v.device)
            planes.append(inverse(C.index_put((idx,), v).reshape(shape), axes))
        return torch.stack(planes, 0)

    with torch.no_grad():
        H_hat = decode(values)
    return H_hat, [2 * int(v.numel()) for v in values], LinearCode(values, decode)
