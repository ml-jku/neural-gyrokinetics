"""Scalar metrics shared by every codec and baseline.

`peak_psnr` is the paper's PSNR: the reference's maximum modulus squared over the mean
squared error.  `field_psnr_conventions` reports it in three layouts of the same spectral
object: the stored spectral layout, the paper layout (radial axis real, binormal spectral)
and the fully real grid.  PSNR is scale invariant, so the transform normalisations cancel
as long as prediction and reference share them.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch


def peak_psnr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    peak = float((ref.abs() ** 2).max())
    mse = float(((pred - ref).abs() ** 2).mean())
    return 10 * math.log10(max(peak, 1e-300) / max(mse, 1e-300))


def rel_l1(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float((pred - ref).abs().sum() / ref.abs().sum().clamp_min(1e-30))


def power_snr_db(ref: torch.Tensor, err: torch.Tensor, dims) -> List[float]:
    """10 log10 (sum |ref|^2 / sum |err|^2) over `dims`, one value per remaining leading index."""
    num = (ref.abs() ** 2).sum(dim=dims, dtype=torch.float64)
    den = (err.abs() ** 2).sum(dim=dims, dtype=torch.float64).clamp_min(1e-300)
    return (10 * torch.log10(num / den)).cpu().tolist()


def _to_x_real(spec: torch.Tensor) -> torch.Tensor:
    """(n_v2, n_v1, n_s1, n_x modes, n_y modes) -> radial axis real (inverse over the centred radial modes)."""
    return torch.fft.ifft(torch.fft.ifftshift(spec, dim=(-2,)), dim=-2, norm="forward")


def _to_xy_real(spec_x: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft(spec_x, n=2 * spec_x.shape[-1], dim=-1, norm="forward")


def _accumulate(pred, ref):
    return (
        float((ref.abs() ** 2).max()),
        float(((pred - ref).abs() ** 2).sum(dtype=torch.float64)),
        ref.numel(),
    )


def _psnr(peak, sse, n):
    return 10 * math.log10(max(peak, 1e-300) / max(sse / max(n, 1), 1e-300))


@torch.no_grad()
def field_psnr_conventions(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, object]:
    """pred, ref: (n_species, n_v2, n_v1, n_s1, n_x modes, n_y modes) complex.  Per species and pooled."""
    out = {"psnr_spectral": [], "psnr_paper": [], "psnr_realgrid": []}
    pooled = {k: [0.0, 0.0, 0] for k in out}

    def add(key, p, r):
        pk, sse, n = _accumulate(p, r)
        out[key].append(_psnr(pk, sse, n))
        acc = pooled[key]
        acc[0] = max(acc[0], pk)
        acc[1] += sse
        acc[2] += n

    for s in range(pred.shape[0]):
        add("psnr_spectral", pred[s], ref[s])
        xp, xr = _to_x_real(pred[s]), _to_x_real(ref[s])
        add("psnr_paper", xp, xr)
        add("psnr_realgrid", _to_xy_real(xp), _to_xy_real(xr))
        del xp, xr
    for k, acc in pooled.items():
        out[k + "_all"] = _psnr(*acc)
    return out


def _to_stored_grid(spec: torch.Tensor) -> torch.Tensor:
    """(..., kx modes, ky modes) -> the complex real-space field the snapshots are stored as:
    the inverse transform over both spatial axes."""
    return torch.fft.ifftn(
        torch.fft.ifftshift(spec, dim=(-2,)), dim=(-2, -1), norm="forward"
    )


def field_psnr_real(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Peak-PSNR of a spectral field on the stored real-space grid (the paper's convention for
    the potentials).  By Parseval it differs from the spectral value by an offset that depends
    on the reference only."""
    return peak_psnr(_to_stored_grid(pred), _to_stored_grid(ref))


def cancellation_ratio(integrand: torch.Tensor, dims) -> List[float]:
    """kappa = (Q+ + Q-) / |Q+ - Q-| of a signed flux integrand, per remaining leading index."""
    qp = integrand.clamp(min=0).sum(dim=dims, dtype=torch.float64)
    qn = (-integrand.clamp(max=0)).sum(dim=dims, dtype=torch.float64)
    return ((qp + qn) / (qp - qn).abs().clamp_min(1e-30)).cpu().tolist()
