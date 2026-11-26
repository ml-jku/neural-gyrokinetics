"""The two contracts the codec is written against: a `Snapshot` and a `Backend`.

A Snapshot is one time step of one simulation as the codec sees it: per species, a complex
field stored as two real channels on a five-dimensional grid (two velocity axes, then
three spatial axes with the axis the carriers modulate last).  The codec never asks what the
axes mean; it fits the array and hands reconstructions to the backend.

A Backend knows the physics.  It converts the stored array to the spectral layout the field
equations use, solves the potential, integrates the flux, and exposes the flux rows the
Gauss-Newton matches.  The GKW backend (through the host `neugk` package) ships with the
package, plus a synthetic one for tests and for trying the codec on other data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence

import torch


@dataclass
class Snapshot:
    """One snapshot.

    fields : per species, a real tensor (2, n_v1, n_v2, n_s1, n_s2, n_s3); channel 0 / 1 are the
             real and imaginary parts of the complex fluctuation.  The last axis is the carrier
             axis (binormal).
    name   : an identifier used for file names and logs.
    meta   : whatever the backend needs (geometry, run paths, spectra); opaque to the codec.
    """

    fields: List[torch.Tensor]
    name: str = "snapshot"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_species(self) -> int:
        return len(self.fields)

    @property
    def shape(self) -> Sequence[int]:
        """The per-channel grid shape (n_v1, n_v2, n_s1, n_s2, n_s3)."""
        return tuple(self.fields[0].shape[1:])

    @property
    def raw_reals(self) -> int:
        """Number of stored reals in the uncompressed snapshot (all species, both channels)."""
        return sum(int(f.numel()) for f in self.fields)


class Backend(Protocol):
    """Everything the codec asks a dataset's physics for.

    The spectral object H (the quantity the atoms represent) has layout
    (n_species, n_v2, n_v1, n_s1, n_s2 modes, n_s3 modes); `to_spectral` produces it from the
    per-species real-space channels the atoms render.
    """

    band: Sequence[int]  # the supervised modes along the last axis
    gt: torch.Tensor  # the ground-truth spectral object the atoms represent
    raw_reals: int  # stored reals of the uncompressed snapshot (for the compression ratio)
    device: str

    def target_fields(self) -> List[torch.Tensor]:
        """Per species, the (2, ...) real-space channels the atoms fit."""
        ...

    def to_spectral(self, fields: List[torch.Tensor]) -> torch.Tensor:
        """Per-species (2, ...) real channels -> the spectral object (n_species, ...)."""
        ...

    def flux_rows(self, spectral: torch.Tensor) -> torch.Tensor:
        """The normalised flux rows (float64 vector) the Gauss-Newton matches."""
        ...

    def flux_rows_target(self) -> torch.Tensor:
        """The same rows evaluated on the ground truth."""
        ...

    def row_mode(self, k: int) -> int:
        """Which mode (index along the last axis) row k reads; rows are local to one mode."""
        ...

    def decode(self, spectral: torch.Tensor) -> torch.Tensor:
        """The stored object -> the distribution function the physics is scored on."""
        ...

    def score(self, decoded: torch.Tensor) -> Dict[str, Any]:
        """Field PSNR, potential PSNR, flux error and the cancellation ratio of one decoded snapshot."""
        ...
