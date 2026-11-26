"""The GKW backend: one species, electrostatic, the stored fluctuation itself as the object.

The physics comes from the host `neugk` package: the quasineutral potential and the
heat-flux spectrum from `get_integrals`, the field and potential losses from
`integral_losses`.  Both operate on the real-space field and transform the binormal axis
themselves, so the spectral object of this backend keeps the radial axis real and
transforms the binormal axis only.  The flux rows are the banded heat-flux spectrum over
its peak; the Gauss-Newton runs with the autograd Jacobian through the host readout.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch

from neugk.gyrosplat.backends.base import RowLayout
from neugk.gyrosplat.data import Snapshot
from neugk.gyrosplat.eval.metrics import peak_psnr, rel_l1


class GkwBackend:
    def __init__(
        self,
        snapshot: Snapshot,
        band: Sequence[int] = range(1, 16),
        device: str = "cuda",
        dominant: Sequence[int] = range(1, 4),
        tail: Sequence[int] = range(7, 16),
    ):
        self.device = device
        self.band = list(band)
        self.dominant, self.tail = list(dominant), list(tail)
        self.snapshot = snapshot
        self.raw_reals = snapshot.raw_reals
        self.geom = {k: v.to(device) for k, v in snapshot.meta["geom"].items()}
        self.ds = snapshot.meta["ds"]
        self.field_gt = snapshot.fields[0].to(device)
        self.gt = self.to_spectral([self.field_gt])
        self.layout = RowLayout(self.band, 1)
        self.real_axes = (
            0,
            1,
            2,
            3,
        )  # velocity, parallel and radial axes of the object are real
        self.spectral_unit = math.sqrt(
            self.gt.shape[-1]
        )  # forward-normalised -> orthonormal
        with torch.no_grad():
            self.phi_gt, self.Q_gt, self.integrand_gt = self.readouts(self.field_gt)
            self.scale = float(self.Q_gt[self.band].abs().max().clamp_min(1e-12))
            self._target_rows = (self.Q_gt[self.band] / self.scale).double()
            summary = self._potential_summary_raw(self.field_gt)
            self.summary_scale = float(summary.abs().max().clamp_min(1e-30))
            self.summary_target = (summary / self.summary_scale).detach()

    # -- transforms: binormal axis only ----------------------------------------------------
    def to_spectral(self, fields: List[torch.Tensor]) -> torch.Tensor:
        c = torch.complex(fields[0][0], fields[0][1])
        return torch.fft.fft(c, dim=-1, norm="forward").permute(1, 0, 2, 3, 4)[None]

    def to_real(self, spectral: torch.Tensor) -> List[torch.Tensor]:
        c = torch.fft.ifft(spectral[0].permute(1, 0, 2, 3, 4), dim=-1, norm="forward")
        return [torch.stack([c.real, c.imag], 0)]

    @staticmethod
    def spatial_spectrum(P: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(P, dim=2, norm="forward")

    def target_fields(self) -> List[torch.Tensor]:
        return [self.field_gt]

    # -- host readouts ----------------------------------------------------------------------
    def readouts(self, field: torch.Tensor):
        """(potential, heat-flux spectrum Q(ky), the signed flux integrand) of a real-space field."""
        from neugk.physics import get_integrals

        phi, (_, ef, _) = get_integrals(
            field, self.geom, flux_fields=True, spectral_df=False
        )
        integrand = ef[0]
        return phi, integrand.sum((0, 1, 2, 3)), integrand

    def flux_magnitude(self) -> float:
        """sum |Q(ky)| over the band of the truth (the carrier gate reads this)."""
        return float(self.Q_gt[self.band].abs().sum())

    def physics_terms(
        self, fields: List[torch.Tensor], with_flux: bool = True
    ) -> Dict[str, torch.Tensor]:
        """The differentiable terms the physics fit and the routed co-fit use, on de-normalised fields."""
        from neugk.physics import integral_losses

        losses = integral_losses(
            fields[0], self.field_gt, geom=self.geom, device=self.device
        )
        terms = {"field": losses["df loss"], "potential": losses["phi loss"]}
        if with_flux:
            _, Q, _ = self.readouts(fields[0])
            b = self.band
            terms["flux"] = (Q[b] - self.Q_gt[b]).abs().sum() / self.Q_gt[
                b
            ].abs().sum().clamp_min(1e-12)
        return terms

    def _potential_summary_raw(self, field: torch.Tensor) -> torch.Tensor:
        phi, _, _ = self.readouts(field)
        F = torch.fft.rfft(phi.float(), dim=-1)
        v = F.sum(dim=tuple(range(F.dim() - 1)))
        return torch.cat([v.real, v.imag])

    def potential_summary(self, fields: List[torch.Tensor]) -> torch.Tensor:
        """A small linear summary of the potential (its binormal spectrum summed over the rest),
        scaled by the truth's peak; the amplitude solve matches it to `summary_target`."""
        return self._potential_summary_raw(fields[0]) / self.summary_scale

    # -- rows -------------------------------------------------------------------------------
    def mode_rows(self, H_mode: torch.Tensor, i: int) -> torch.Tensor:
        full = torch.zeros(self.gt.shape, dtype=H_mode.dtype, device=H_mode.device)
        full[..., i] = H_mode
        _, Q, _ = self.readouts(self.to_real(full)[0])
        return (Q[i] / self.scale).reshape(1).double()

    def flux_rows(self, H: torch.Tensor) -> torch.Tensor:
        _, Q, _ = self.readouts(self.to_real(H)[0])
        return (Q[self.band] / self.scale).double()

    def flux_rows_target(self) -> torch.Tensor:
        return self._target_rows

    def row_index(self, b: int, k: int) -> int:
        return self.layout.row_index(b, k)

    def row_mode(self, row: int) -> int:
        return self.layout.row_mode(row)

    def decode(self, H: torch.Tensor) -> torch.Tensor:
        """The stored fluctuation is the scored distribution itself."""
        return H

    # -- scoring ----------------------------------------------------------------------------
    @torch.no_grad()
    def score(self, decoded: torch.Tensor) -> Dict[str, object]:
        """The paper's scores of a decoded spectral object.

        `psnr_f`, `psnr_phi`: peak PSNR (the reference's largest modulus squared over the mean
        squared error) of the distribution on its stored real grid and of the potential.
        `flux_rel_l1`: relative L1 error of the heat-flux spectrum over every nonzero binormal
        mode; `kappa`, `kappa_truth`, `kappa_rel`: the gross-to-net cancellation ratio of the
        flux integrand over the same modes, the reference's, and their ratio.  The older keys
        (`psnr_paper_all`, `phi_psnr` with the signed peak, `flux_relL1_band` over the
        supervised band, `kappa` per band) stay for the fit logs."""
        field = self.to_real(decoded)[0]
        phi, Q, integrand = self.readouts(field)
        b = self.band
        peak = float(self.phi_gt.max()) ** 2
        mse = float(((phi - self.phi_gt) ** 2).mean())
        nz = slice(1, None)
        kappa, kappa_truth = _kappa(integrand[..., nz]), _kappa(
            self.integrand_gt[..., nz]
        )
        return {
            "psnr_paper": [peak_psnr(field, self.field_gt)],
            "psnr_paper_all": peak_psnr(field, self.field_gt),
            "phi_psnr": 10 * math.log10(max(peak, 1e-300) / max(mse, 1e-300)),
            "flux_relL1_band": [rel_l1(Q[b], self.Q_gt[b])],
            "flux_relL1_dominant": [rel_l1(Q[self.dominant], self.Q_gt[self.dominant])],
            "flux_relL1_tail": [rel_l1(Q[self.tail], self.Q_gt[self.tail])],
            "kappa": [_kappa(integrand[..., b])],
            "kappa_gt": [_kappa(self.integrand_gt[..., b])],
            "psnr_f": peak_psnr(field, self.field_gt),
            "psnr_phi": peak_psnr(phi, self.phi_gt),
            "psnr_phi_signed": 10 * math.log10(max(peak, 1e-300) / max(mse, 1e-300)),
            "f_rel_l2": _rel_l2(field, self.field_gt),
            "phi_rel_l2": _rel_l2(phi, self.phi_gt),
            "flux_rel_l1": rel_l1(Q[nz].double(), self.Q_gt[nz].double()),
            "flux_rel_l2": _rel_l2(Q[nz], self.Q_gt[nz]),
            "kappa_nonzero": kappa,
            "kappa_truth_nonzero": kappa_truth,
            "kappa_rel": kappa / max(kappa_truth, 1e-150),
            "flux_pred": Q.detach().cpu().tolist(),
            "flux_truth": self.Q_gt.detach().cpu().tolist(),
            "flux_evaluation_modes": list(range(1, Q.numel())),
            "flux_training_modes": list(b),
        }


def _rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    err = (pred.double() - ref.double()).abs().square().sum().sqrt()
    return float(err / ref.double().abs().square().sum().sqrt().clamp_min(1e-150))


def _kappa(integrand: torch.Tensor) -> float:
    gross = integrand.abs().sum(dtype=torch.float64)
    net = integrand.sum(dtype=torch.float64).abs().clamp_min(1e-30)
    return float(gross / net)
