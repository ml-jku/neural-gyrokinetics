"""Shared pieces of a backend: the spectral transform conventions and the row bookkeeping.

The stored real-space channels of a species have layout (2, n_v1, n_v2, n_s1, x, y); the
spectral object used by the physics is complex with layout
(n_species, n_v2, n_v1, n_s1, kx, ky), kx centred (fftshift) and ky one-sided as the
simulation stores it.  Rows of the flux objective are local to one ky mode; a backend
returns k rows per supervised mode and the loop below flattens them mode-major so that
`row_index(b, k)` is the position of row k of the b-th supervised mode.
"""
from __future__ import annotations

from typing import Sequence

import torch


def to_spectral_species(field: torch.Tensor) -> torch.Tensor:
    """(2, n_v1, n_v2, n_s1, x, y) real channels -> (n_v2, n_v1, n_s1, kx, ky) complex."""
    c = torch.complex(field[0], field[1])
    spec = torch.fft.fftshift(
        torch.fft.fftn(c, dim=(-2, -1), norm="forward"), dim=(-2,)
    )
    return spec.permute(1, 0, 2, 3, 4)


def to_real_species(spec: torch.Tensor) -> torch.Tensor:
    """The inverse of `to_spectral_species`: (n_v2, n_v1, n_s1, kx, ky) -> (2, n_v1, n_v2, n_s1, x, y)."""
    c = torch.fft.ifftn(
        torch.fft.ifftshift(spec.permute(1, 0, 2, 3, 4), dim=(-2,)),
        dim=(-2, -1),
        norm="forward",
    )
    return torch.stack([c.real, c.imag], 0)


def spatial_spectrum(P: torch.Tensor) -> torch.Tensor:
    """(n_s1, x, y, m) complex footprints -> (n_s1, kx, ky, m), the same transform as the object."""
    return torch.fft.fftshift(torch.fft.fftn(P, dim=(1, 2), norm="forward"), dim=(1,))


class RowLayout:
    """Mode-major flattening of `k_per_mode` rows over the supervised modes `band`."""

    def __init__(self, band: Sequence[int], k_per_mode: int):
        self.band = list(band)
        self.k = k_per_mode

    @property
    def n_rows(self) -> int:
        return len(self.band) * self.k

    def row_index(self, b: int, k: int) -> int:
        return b * self.k + k

    def row_mode(self, row: int) -> int:
        return self.band[row // self.k]
