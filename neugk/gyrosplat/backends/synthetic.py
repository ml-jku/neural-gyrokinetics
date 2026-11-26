"""A toy backend: the smallest physics with the same structure as the real ones.

The "potential" of a mode is one weighted velocity moment of the object,
phi(kx, ky) = sum_v w_v H_v(kx, ky), and the "flux" of a ky mode is the signed quadratic
form  Q(ky) = sum_kx sum_v Im( H_v conj(w_v phi) ) e_v  with an energy weight e_v.  That is
exactly the shape of the gyrokinetic readouts (fields linear, fluxes quadratic and local
per mode), so every stage of the codec runs on it unchanged.  Use it as the template for
wiring a new dataset.
"""
from __future__ import annotations

import math

from typing import Dict, List, Sequence

import torch

from neugk.gyrosplat.backends.base import (
    RowLayout,
    spatial_spectrum,
    to_real_species,
    to_spectral_species,
)
from neugk.gyrosplat.data import Snapshot
from neugk.gyrosplat.eval.metrics import (
    cancellation_ratio,
    field_psnr_conventions,
    peak_psnr,
    rel_l1,
)


class SyntheticBackend:
    def __init__(
        self,
        snapshot: Snapshot,
        band: Sequence[int] = (1, 2, 3, 4),
        device="cpu",
        seed: int = 0,
    ):
        self.device = device
        self.band = list(band)
        self.snapshot = snapshot
        self.raw_reals = snapshot.raw_reals
        shape = snapshot.shape
        gen = torch.Generator().manual_seed(seed)
        n_v1, n_v2 = shape[0], shape[1]
        # velocity moment weights and an energy weight, fixed by the seed
        self.w = torch.complex(
            torch.randn(n_v2, n_v1, generator=gen),
            torch.randn(n_v2, n_v1, generator=gen),
        ).to(device)
        self.e = torch.rand(n_v2, n_v1, generator=gen).to(device) + 0.5
        self.gt = self.to_spectral(
            [f.to(device) for f in snapshot.fields]
        )  # (n_species, ...)
        self.layout = RowLayout(self.band, 1 + snapshot.n_species)
        self.real_axes = (
            0,
            1,
            2,
        )  # the velocity and parallel axes of the object are real
        self.spectral_unit = math.sqrt(
            self.gt.shape[-1] * self.gt.shape[-2]
        )  # forward-normalised -> orthonormal
        with torch.no_grad():
            gross = torch.stack(
                [
                    self._integrand(self.gt[..., i]).abs().sum(dim=(1, 2, 3, 4))
                    for i in range(self.gt.shape[-1])
                ]
            )
            self.gross = gross.clamp_min(1e-30)
            self._target = self.flux_rows(self.gt)

    # -- transforms -----------------------------------------------------------------------
    def to_spectral(self, fields: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([to_spectral_species(f) for f in fields], 0)

    def to_real(self, spectral: torch.Tensor) -> List[torch.Tensor]:
        return [to_real_species(spectral[s]) for s in range(spectral.shape[0])]

    spatial_spectrum = staticmethod(spatial_spectrum)

    def target_fields(self) -> List[torch.Tensor]:
        """What the atoms represent: here the stored field itself."""
        return [f.to(self.device) for f in self.snapshot.fields]

    # -- readouts ---------------------------------------------------------------------------
    def potential(self, H_mode: torch.Tensor) -> torch.Tensor:
        """(n_species, n_v2, n_v1, n_s1, kx) -> (n_s1, kx): the species-summed moment."""
        return torch.einsum("ev,sevcr->cr", self.w, H_mode)

    def _integrand(self, H_mode: torch.Tensor) -> torch.Tensor:
        phi = self.potential(H_mode)
        return (
            torch.imag(
                H_mode
                * torch.conj(self.w[None, :, :, None, None] * phi[None, None, None])
            )
            * self.e[None, :, :, None, None]
        )

    def mode_rows(self, H_mode: torch.Tensor, i: int) -> torch.Tensor:
        """Rows of one mode: total flux, then the per-species flux, each over its own gross."""
        q = self._integrand(H_mode).sum(dim=(1, 2, 3, 4))  # (n_species,)
        g = self.gross[i]
        return torch.cat([(q.sum() / g.sum()).reshape(1), q / g]).double()

    def flux_rows(self, H: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mode_rows(H[..., i], i) for i in self.band])

    def flux_rows_target(self) -> torch.Tensor:
        return self._target

    def row_index(self, b: int, k: int) -> int:
        return self.layout.row_index(b, k)

    def row_mode(self, row: int) -> int:
        return self.layout.row_mode(row)

    def fields_of(self, H: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.potential(H[..., i]) for i in range(H.shape[-1])], -1)[
            None
        ]

    def decode(self, H: torch.Tensor) -> torch.Tensor:
        """The stored object is the scored distribution itself."""
        return H

    # -- the physics terms of the descent stages --------------------------------------------
    def _band_flux(self, H: torch.Tensor) -> torch.Tensor:
        """(n_band, n_species) signed flux per supervised mode."""
        return torch.stack(
            [self._integrand(H[..., i]).sum(dim=(1, 2, 3, 4)) for i in self.band]
        )

    def flux_magnitude(self) -> float:
        return float(self._band_flux(self.gt).abs().sum())

    def physics_terms(
        self, fields: List[torch.Tensor], with_flux: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Relative squared errors of the field and of the potential, and the relative L1 error of
        the band flux, all differentiable in the fields."""
        H = self.to_spectral(fields)
        gt_fields = self.target_fields()
        field = sum(((f - g) ** 2).sum() for f, g in zip(fields, gt_fields)) / sum(
            (g**2).sum() for g in gt_fields
        )
        phi, phi_gt = self.fields_of(H)[0], self.fields_of(self.gt)[0]
        terms = {
            "field": field,
            "potential": ((phi - phi_gt).abs() ** 2).sum() / (phi_gt.abs() ** 2).sum(),
        }
        if with_flux:
            q, q_gt = self._band_flux(H), self._band_flux(self.gt)
            terms["flux"] = (q - q_gt).abs().sum() / q_gt.abs().sum().clamp_min(1e-30)
        return terms

    def _summary_raw(self, H: torch.Tensor) -> torch.Tensor:
        phi = self.fields_of(H)[0].sum(0).reshape(-1)  # (kx * ky,) complex
        return torch.cat([phi.real, phi.imag])

    @property
    def summary_scale(self) -> float:
        with torch.no_grad():
            return float(self._summary_raw(self.gt).abs().max().clamp_min(1e-30))

    @property
    def summary_target(self) -> torch.Tensor:
        with torch.no_grad():
            return self._summary_raw(self.gt) / self.summary_scale

    def potential_summary(self, fields: List[torch.Tensor]) -> torch.Tensor:
        """The potential on every mode, summed over the parallel axis, scaled by the truth's peak."""
        return self._summary_raw(self.to_spectral(fields)) / self.summary_scale

    # -- scoring ----------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, decoded: torch.Tensor) -> Dict[str, object]:
        gt = self.gt
        phi, phi_gt = self.fields_of(decoded)[0], self.fields_of(gt)[0]
        q = torch.stack(
            [
                self._integrand(decoded[..., i]).sum(dim=(1, 2, 3, 4))
                for i in range(gt.shape[-1])
            ]
        )
        q_gt = torch.stack(
            [
                self._integrand(gt[..., i]).sum(dim=(1, 2, 3, 4))
                for i in range(gt.shape[-1])
            ]
        )
        integ = torch.stack([self._integrand(decoded[..., i]) for i in self.band], -1)
        integ_gt = torch.stack([self._integrand(gt[..., i]) for i in self.band], -1)
        row = {
            "phi_psnr": peak_psnr(phi, phi_gt),
            "flux_relL1_band": [
                rel_l1(q[self.band, s], q_gt[self.band, s]) for s in range(gt.shape[0])
            ],
            "kappa": cancellation_ratio(integ, dims=(1, 2, 3, 4, 5)),
            "kappa_gt": cancellation_ratio(integ_gt, dims=(1, 2, 3, 4, 5)),
        }
        row.update(field_psnr_conventions(decoded, gt))
        return row
