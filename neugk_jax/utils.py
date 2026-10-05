"""Generic utilities: config conversion, trace counting, atomic writes, progress bars,
separate/recombine zonal flow, running stats."""

from __future__ import annotations

import functools
import os
from collections import Counter
from typing import Callable, Mapping

import equinox as eqx
import numpy as np
import yaml
from omegaconf import OmegaConf

# number of times each named jitted function has been traced
TRACE_COUNTS: Counter = Counter()


def count_trace(name: str) -> None:
    TRACE_COUNTS[name] += 1


def traced_jit(name: str, **jit_kwargs):
    """``eqx.filter_jit`` that counts every trace of the function under ``name``."""

    def deco(fn):
        @functools.wraps(fn)
        def traced(*args, **kwargs):
            count_trace(name)
            return fn(*args, **kwargs)

        return eqx.filter_jit(traced, **jit_kwargs)

    return deco


def to_dict(src) -> dict:
    """A plain resolved dict from an OmegaConf node, a mapping or a YAML path (``{}`` for None)."""
    if src is None:
        return {}
    if OmegaConf.is_config(src):
        return OmegaConf.to_container(src, resolve=True)
    if isinstance(src, Mapping):
        return dict(src)
    with open(src) as f:
        return yaml.safe_load(f)


def atomic_write(path, write: Callable, mode: str = "wb") -> None:
    """Write ``path`` through ``write(file)`` into a temporary sibling, then rename it into place."""
    path = os.fspath(path)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, mode) as f:
        write(f)
    os.replace(tmp, path)


def progress(it, show: bool, **kwargs):
    """``it`` wrapped in a tqdm bar when ``show``."""
    if not show:
        return it
    from tqdm import tqdm

    return tqdm(it, **kwargs)


def separate_zf(x, axis: int = 0):
    """Separate Zonal Flow (ZF) and non-ZF components of a numpy or jax array.

    Layout: ``[zf, x - zf]`` along ``axis``. ZF is the **mean** over the
    last axis (ky), broadcast back; the "rest" is ``x - zf`` (so the
    decomposition is exact, ``zf + rest == x``).
    """
    xp = x.__array_namespace__()
    zf = xp.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
    return xp.concatenate([zf, x - zf], axis=axis)


def recombine_zf(x, axis: int = 0):
    """Sum the (real, imag) channel pairs along ``axis`` back to 2 channels.

    Inverts ``separate_zf`` (``[zf, rest]``) and the ky-band split of the preprocessing;
    a 2-channel ``x`` is returned as is.
    """
    xp = x.__array_namespace__()
    x = xp.moveaxis(x, axis, 0)
    return xp.moveaxis(x.reshape(-1, 2, *x.shape[1:]).sum(0), 0, axis)


class RunningStats:
    """Elementwise running mean/var/min/max in float64, seeded with ``prior_count`` (variance 1).

    ``push`` adds one sample in place (Welford), ``merge`` a summary of ``count`` samples
    (Chan et al.); buffers take the shape of the first input. Unpickled ``RunningMeanStd``
    objects of the stored dataset statistics carry the same attributes.
    """

    def __init__(self, prior_count: float):
        self.count = prior_count
        self.mean = self.var = self.min = self.max = None

    def _start(self, shape) -> None:
        self.mean, self.var = np.zeros(shape), np.ones(shape)
        self.min, self.max = np.full(shape, np.inf), np.full(shape, -np.inf)

    def push(self, x) -> None:
        x = np.asarray(x)
        if self.mean is None:
            self._start(x.shape)
        c, n1 = self.count, self.count + 1.0
        d = np.asarray(x - self.mean)
        self.mean += d * (1.0 / n1)
        self.var *= c / n1
        np.multiply(d, d, out=d)
        d *= c / (n1 * n1)
        self.var += d
        np.minimum(self.min, x, out=self.min)
        np.maximum(self.max, x, out=self.max)
        self.count = n1

    def merge(self, mean, var, mn, mx, count: float = 1) -> None:
        mean, var = np.asarray(mean, np.float64), np.asarray(var, np.float64)
        if self.mean is None:
            self._start(mean.shape)
        new_count = self.count + count
        delta = mean - self.mean
        m2 = self.var * self.count + var * count + delta**2 * (self.count * count / new_count)
        self.mean = self.mean + delta * (count / new_count)
        self.var = m2 / new_count
        self.min = np.minimum(self.min, np.asarray(mn, np.float64))
        self.max = np.maximum(self.max, np.asarray(mx, np.float64))
        self.count = new_count

    def moments(self, dtype=None, axes=None) -> dict:
        """``mean``/``var``/``std``/``min``/``max``, optionally pooled over ``axes`` (keepdims).

        Pooling over ``axes`` runs in the stored precision and takes the mean of the means, the
        mean variance plus the variance of the means, and the extreme min/max.
        """
        mean, var = np.asarray(self.mean), np.asarray(self.var)
        mn, mx = np.asarray(self.min), np.asarray(self.max)
        if axes:
            pooled = mean.mean(axis=axes, keepdims=True)
            spread = ((mean - pooled) ** 2).mean(axis=axes, keepdims=True)
            mean, var = pooled, var.mean(axis=axes, keepdims=True) + spread
            mn, mx = mn.min(axis=axes, keepdims=True), mx.max(axis=axes, keepdims=True)
        out = {"mean": mean, "var": var, "std": np.sqrt(var), "min": mn, "max": mx}
        return out if dtype is None else {k: v.astype(dtype) for k, v in out.items()}
