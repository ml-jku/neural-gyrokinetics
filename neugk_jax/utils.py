"""Generic utilities: seeding, separate/recombine zonal flow, running stats."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy. JAX uses explicit keys threaded through ops."""
    random.seed(seed)
    np.random.seed(seed)


def separate_zf(x, axis: int = 0):
    """Separate Zonal Flow (ZF) and non-ZF components — matches upstream.

    Layout: ``[zf, x - zf]`` along ``axis``. ZF is the **mean** over the
    last axis (ky), broadcast back; the "rest" is ``x - zf`` (so the
    decomposition is exact, ``zf + rest == x``).

    Works on both numpy and jax arrays — picks the right namespace via
    duck typing.
    """
    nky = x.shape[-1]
    if isinstance(x, jnp.ndarray):
        zf = jnp.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
        return jnp.concatenate([zf, x - zf], axis=axis)
    zf = np.broadcast_to(x.mean(axis=-1, keepdims=True), x.shape)
    return np.concatenate([zf, x - zf], axis=axis)


def recombine_zf(x, axis: int = 0):
    """Inverse of ``separate_zf``: ``[zf, non_zf]`` → ``zf + non_zf``."""
    if x.shape[axis] <= 2 or x.shape[axis] % 2 != 0:
        return x
    half = x.shape[axis] // 2
    if isinstance(x, jnp.ndarray):
        zf, non_zf = jnp.split(x, 2, axis=axis)
    else:
        zf, non_zf = np.split(x, 2, axis=axis)
    return zf + non_zf


def remaining_progress(step: float, total: float) -> float:
    return min(max(step / max(total, 1.0), 0.0), 1.0)


@dataclass
class RunningMeanStd:
    """Numerically stable running mean/var over batches (numpy buffers)."""

    mean: np.ndarray | float = 0.0
    var: np.ndarray | float = 1.0
    min: np.ndarray | float = math.inf
    max: np.ndarray | float = -math.inf
    count: float = 0.0

    def __init__(self, shape: tuple[int, ...] | None = None):
        if shape is None:
            self.mean = 0.0
            self.var = 1.0
            self.min = math.inf
            self.max = -math.inf
        else:
            self.mean = np.zeros(shape, dtype=np.float64)
            self.var = np.ones(shape, dtype=np.float64)
            self.min = np.full(shape, math.inf, dtype=np.float64)
            self.max = np.full(shape, -math.inf, dtype=np.float64)
        self.count = 0.0

    def update(self, mean, var, mn, mx, count: int = 1) -> None:
        mean = np.asarray(mean, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)
        mn = np.asarray(mn, dtype=np.float64)
        mx = np.asarray(mx, dtype=np.float64)
        new_count = self.count + count
        delta = mean - np.asarray(self.mean)
        new_mean = np.asarray(self.mean) + delta * (count / new_count)
        m_a = np.asarray(self.var) * self.count
        m_b = var * count
        m2 = m_a + m_b + (delta**2) * (self.count * count / new_count)
        new_var = m2 / new_count
        self.mean = new_mean
        self.var = new_var
        self.min = np.minimum(self.min, mn)
        self.max = np.maximum(self.max, mx)
        self.count = new_count

    def combine(self, other: "RunningMeanStd") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.mean = other.mean
            self.var = other.var
            self.min = other.min
            self.max = other.max
            self.count = other.count
            return
        self.update(other.mean, other.var, other.min, other.max, int(other.count))


def expand_as(x: np.ndarray | jnp.ndarray, ref: np.ndarray | jnp.ndarray):
    """Broadcast x to the shape of ref by inserting leading singleton axes."""
    x = jnp.asarray(x) if isinstance(ref, jnp.ndarray) else np.asarray(x)
    while x.ndim < ref.ndim:
        x = x[None, ...] if isinstance(x, np.ndarray) else jnp.expand_dims(x, 0)
    return jnp.broadcast_to(x, ref.shape) if isinstance(ref, jnp.ndarray) else np.broadcast_to(x, ref.shape)


def split_keys(key: jax.Array, n: int) -> list[jax.Array]:
    """Convenience: split into a Python list of subkeys."""
    return list(jax.random.split(key, n))


def stop_grad(x):
    """Alias for jax.lax.stop_gradient (used for frozen buffers stored as leaves)."""
    return jax.lax.stop_gradient(x)
