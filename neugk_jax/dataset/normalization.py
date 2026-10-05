"""Normalization statistics and the per-field scale/shift table built from them."""

from __future__ import annotations

import pickle
from typing import Mapping, Optional, Sequence

import equinox as eqx
import numpy as np

from neugk_jax.utils import RunningStats


class _StatsUnpickler(pickle.Unpickler):
    """Unpickles stats pickles that reference a ``RunningMeanStd`` class."""

    def find_class(self, module, name):
        if name == "RunningMeanStd" and module.startswith(("neugk.", "neugk_jax.")):
            return RunningStats
        return super().find_class(module, name)


def load_stats(path, normalization: Optional[Mapping] = None) -> dict[str, dict]:
    """``stats[field]["full"]`` from a stats pickle.

    The pickle holds one ``RunningMeanStd`` per field, reduced over the field's
    ``normalization[field]["agg_axes"]`` (keepdims), or a dict already in the
    ``stats[field][key]`` form (returned as is).
    """
    with open(path, "rb") as f:
        raw = _StatsUnpickler(f).load()
    if all(isinstance(v, dict) for v in raw.values()):
        return {str(k): v for k, v in raw.items()}
    out = {}
    for k, rms in raw.items():
        agg = ((normalization or {}).get(k) or {}).get("agg_axes")
        out[k] = {"full": rms.moments(np.float32, axes=tuple(int(a) for a in agg) if agg else None)}
    return out


def metadata_stats(metadata: Mapping[int, dict], fields: Sequence[str]) -> dict[str, dict]:
    """``stats[field][fid]`` and their pooled ``stats[field]["full"]`` from metadata moments."""
    out: dict[str, dict] = {}
    for k in fields:
        out[k] = {}
        pooled = RunningStats(prior_count=0.0)
        for fid, meta in metadata.items():
            if f"{k}_mean" not in meta:
                continue
            mean, std = meta[f"{k}_mean"], meta[f"{k}_std"]
            mn, mx = meta.get(f"{k}_min", mean), meta.get(f"{k}_max", mean)
            out[k][fid] = {"mean": mean, "std": std, "min": mn, "max": mx}
            pooled.merge(mean, std**2, mn, mx, count=len(meta["timesteps"]))
        if pooled.count:
            out[k]["full"] = pooled.moments(np.float32)
    return out


def _scale_shift(spec: Mapping, stats: Mapping) -> tuple[np.ndarray, np.ndarray]:
    if spec["type"] == "zscore":
        return np.asarray(stats["std"], np.float32), np.asarray(stats["mean"], np.float32)
    if spec["type"] == "minmax":
        lo, hi = np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)
        scale = (hi - lo) / spec.get("minmax_beta1", 8)
        return scale, lo + scale * spec.get("minmax_beta2", 4)
    raise ValueError(f"unknown normalization type {spec['type']!r}")


def _to_rank(a: np.ndarray, ndim: int) -> np.ndarray:
    # drop or prepend unit axes so a per-trajectory table row broadcasts against one sample
    while a.ndim > ndim and a.shape[0] == 1:
        a = a[0]
    return a.reshape((1,) * (ndim - a.ndim) + a.shape)


class NormTable(eqx.Module):
    """Per-field ``(scale, shift)`` of the normalization ``x_norm = (x - shift) / scale``.

    Dataset-scope tables broadcast against one sample or a batch; per-trajectory tables
    (``per_file``) carry a leading trajectory axis and are gathered by file index (a scalar
    for one sample, an array for a batch). Fields without a table are identity; fields
    listed in ``missing`` were configured but have no statistics and raise. The arrays are
    numpy on the host and device arrays once placed; the methods also run inside jit.
    """

    scale: dict
    shift: dict
    per_file: bool = eqx.field(static=True)
    missing: tuple = eqx.field(static=True)

    @classmethod
    def from_stats(
        cls,
        stats: Mapping[str, Mapping],
        normalization: Optional[Mapping],
        scope: str,
        n_files: int,
        ndims: Mapping[str, int],
    ) -> "NormTable":
        """Table of every ``normalization`` field with statistics in ``stats``.

        ``scope`` is ``"dataset"`` (the ``"full"`` statistics) or ``"trajectory"`` (one row
        per file id); ``ndims`` is the per-sample rank of each field.
        """
        if scope not in ("dataset", "trajectory"):
            raise ValueError(
                f"normalization_scope must be 'dataset' or 'trajectory', got {scope!r}"
            )
        per_file = scope == "trajectory"
        scale, shift, missing = {}, {}, []
        for field, spec in (normalization or {}).items():
            by_key = stats.get(field) or {}
            keys = list(range(n_files)) if per_file else ["full"]
            if not all(k in by_key and by_key[k] for k in keys):
                missing.append(field)
                continue
            pairs = [_scale_shift(spec, by_key[k]) for k in keys]
            if per_file:
                scale[field] = np.stack([_to_rank(p[0], ndims[field]) for p in pairs])
                shift[field] = np.stack([_to_rank(p[1], ndims[field]) for p in pairs])
            else:
                scale[field], shift[field] = pairs[0]
        return cls(scale, shift, per_file, tuple(missing))

    def scale_shift(self, field: str, fid):
        """``(scale, shift)`` of ``field`` for file index (or indices) ``fid``."""
        if field in self.missing:
            raise KeyError(
                f"normalization lists {field!r} but there are no {field!r} statistics; "
                "add them to normalization_stats or drop the field"
            )
        if field not in self.scale:
            return np.float32(1.0), np.float32(0.0)
        scale, shift = self.scale[field], self.shift[field]
        if self.per_file:
            scale, shift = scale[fid], shift[fid]
        return scale, shift

    def normalize(self, field: str, x, fid):
        scale, shift = self.scale_shift(field, fid)
        return (x - shift) / scale

    def denormalize(self, field: str, x, fid):
        scale, shift = self.scale_shift(field, fid)
        return x * scale + shift
