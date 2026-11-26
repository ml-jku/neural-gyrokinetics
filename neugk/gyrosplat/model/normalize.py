"""Zonal-split z-scoring of the target field.

The ky = 0 part of a gyrokinetic fluctuation (the zonal flow) is smooth and carries most of
the power; z-scoring it together with the rest would let it dominate the fit.  The zonal
mean over the carrier axis and the remainder therefore get separate statistics.  The
transform is affine and is undone before anything physical is evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormStats:
    zonal_mean: float
    zonal_std: float
    fluct_mean: float
    fluct_std: float
    kind: str = "zonal"


def normalize(field: torch.Tensor, kind: str = "zonal", eps: float = 1e-8):
    """field (2, ..., n_y) -> (normalised field, stats).  kind "global" is a plain z-score."""
    if kind == "global":
        m, s = field.mean(), field.std().clamp_min(eps)
        return (field - m) / s, NormStats(
            float(m), float(s), float(m), float(s), "global"
        )
    zonal = field.mean(dim=-1, keepdim=True).expand_as(field)
    fluct = field - zonal
    zm, zs = zonal.mean(), zonal.std().clamp_min(eps)
    fm, fs = fluct.mean(), fluct.std().clamp_min(eps)
    out = (zonal - zm) / zs + (fluct - fm) / fs
    return out, NormStats(float(zm), float(zs), float(fm), float(fs), "zonal")


def denormalize(field: torch.Tensor, stats: NormStats) -> torch.Tensor:
    if stats.kind == "global":
        return field * stats.zonal_std + stats.zonal_mean
    zonal = field.mean(dim=-1, keepdim=True).expand_as(field)
    fluct = field - zonal
    return (zonal * stats.zonal_std + stats.zonal_mean) + (
        fluct * stats.fluct_std + stats.fluct_mean
    )


def denormalize_linear(field: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """The linear part of `denormalize` (no offsets): what a derivative with respect to the
    amplitudes sees."""
    if stats.kind == "global":
        return field * stats.zonal_std
    zonal = field.mean(dim=-1, keepdim=True).expand_as(field)
    return zonal * stats.zonal_std + (field - zonal) * stats.fluct_std
