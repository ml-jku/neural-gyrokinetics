"""Positional and conditioning embeddings."""

from __future__ import annotations

import math
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.utils import MLP, Linear, relu, silu


def _sincos_1d(length: int, dim: int, base: float = 10000.0) -> jnp.ndarray:
    """Standard sinusoidal embedding of shape (length, dim). dim must be even."""
    assert dim % 2 == 0, "sincos dim must be even"
    pos = jnp.arange(length, dtype=jnp.float32)
    # force f32: with x64 enabled (gyaradax import) ``math.log(base)`` is a
    # Python float which promotes freqs/angles to f64; the resulting buffer
    # then contaminates every downstream call (DiT output → scan dtype mismatch)
    freqs = jnp.exp(-math.log(base) * jnp.arange(0, dim, 2, dtype=jnp.float32) / dim).astype(jnp.float32)
    angles = pos[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1).astype(jnp.float32)


def _sincos_nd(grid_size: Sequence[int], dim: int) -> jnp.ndarray:
    """N-D sincos embedding of shape (*grid_size, dim).

    Splits ``dim`` across axes evenly (last axis absorbs the remainder).
    """
    n = len(grid_size)
    per_axis = dim // n
    rem = dim - per_axis * (n - 1)
    pe_parts = []
    for i, size in enumerate(grid_size):
        d = rem if i == n - 1 else per_axis
        # round d up to even — sincos requires even dim
        if d % 2 == 1:
            d += 1
        pe_i = _sincos_1d(size, d)  # (size, d)
        # broadcast pe_i to full grid shape
        shape = [1] * n + [d]
        shape[i] = size
        pe_parts.append(pe_i.reshape(shape))
    pe = jnp.concatenate([jnp.broadcast_to(p, (*grid_size, p.shape[-1])) for p in pe_parts], axis=-1)
    # trim any rounding overshoot to exactly dim
    return pe[..., :dim]


class APE(eqx.Module):
    """Absolute positional embedding broadcast-added to the last axis.

    Field is named ``pos_embed`` to match upstream torch's ``register_buffer``
    name. ``learnable=False`` keeps it frozen via stop_gradient. With
    ``leading_batch=True`` the stored tensor has an extra leading ``(1,)``
    axis — used by ``vel_pe`` in the 5D AE to mirror torch's shape exactly.
    """

    pos_embed: jax.Array
    learnable: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        learnable: bool = False,
        init: str = "sincos",
        key=None,
    ):
        if init == "sincos":
            pe = _sincos_nd(tuple(grid_size), dim)
        elif init == "zeros":
            pe = jnp.zeros((*grid_size, dim))
        elif init == "normal":
            assert key is not None
            pe = 0.02 * jr.normal(key, (*grid_size, dim))
        else:
            raise ValueError(init)
        self.pos_embed = pe
        self.learnable = learnable

    def __call__(self, x: jax.Array) -> jax.Array:
        pe = self.pos_embed if self.learnable else jax.lax.stop_gradient(self.pos_embed)
        # prepend singleton batch axes so pe broadcasts over x
        while pe.ndim < x.ndim:
            pe = pe[None, ...]
        return x + pe


class ContinuousConditionEmbed(eqx.Module):
    """Sinusoidal embedding for continuous scalars (timestep, ITG, dg, ...).

    Faithful port of ``neugk/models/layers.py:ContinuousConditionEmbed``:
    splits ``dim`` evenly across ``n_cond`` axes, sincos-encodes each with a
    shared log-spaced ``omega`` buffer, zero-pads to ``dim``, then projects
    through a single ``Linear(dim, 4·dim)`` followed by SiLU. Output is
    ``(..., 4·dim)`` and ``self.cond_dim = 4·dim``.
    """

    mlp: list  # [Linear] — list mirrors torch nn.Sequential indexing
    omega: jax.Array
    dim: int = eqx.field(static=True)
    n_cond: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    cond_per_wave: int = eqx.field(static=True)
    max_wavelength: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        n_cond: int,
        *,
        key,
        max_wavelength: float = 10000.0,
    ):
        self.dim = dim
        self.n_cond = n_cond
        self.max_wavelength = max_wavelength

        ndim_padding = dim % n_cond
        dim_per_ndim = (dim - ndim_padding) // n_cond
        sincos_padding = dim_per_ndim % 2
        padding = ndim_padding + sincos_padding * n_cond
        cond_per_wave = (dim - padding) // n_cond
        assert cond_per_wave > 0
        self.padding = padding
        self.cond_per_wave = cond_per_wave

        # force f32: with x64 enabled (gyaradax import), Python float
        # ``max_wavelength`` would promote omega to f64 and contaminate the DiT output.
        omega = 1.0 / (max_wavelength ** (
            jnp.arange(0, cond_per_wave, 2, dtype=jnp.float32) / cond_per_wave
        ))
        self.omega = jax.lax.stop_gradient(omega.astype(jnp.float32))

        self.cond_dim = 4 * dim
        self.mlp = [Linear(dim, self.cond_dim, key=key)]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (..., n_cond); handle scalar by adding trailing dim
        if x.ndim == 0:
            x = x[None]
        # ensure trailing n_cond axis
        if x.shape[-1] != self.n_cond:
            x = x.reshape(*x.shape[:-1], self.n_cond)
        # (..., n_cond, 1) * (cond_per_wave/2,) → (..., n_cond, cond_per_wave/2)
        out = x[..., None] * self.omega
        emb = jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=-1)
        # flatten n_cond * cond_per_wave into a single feature axis
        emb = emb.reshape(*emb.shape[:-2], self.n_cond * self.cond_per_wave)
        if self.padding > 0:
            pad = jnp.zeros((*emb.shape[:-1], self.padding), dtype=emb.dtype)
            emb = jnp.concatenate([emb, pad], axis=-1)
        emb = self.mlp[0](emb)
        return jax.nn.silu(emb)




def _build_rpb_table(window_size: Sequence[int]) -> jnp.ndarray:
    """Continuous log-scaled relative-coordinate table for the cpb MLP.

    Shape ``(1, *(2w-1)_per_axis, space)`` — leading ``(1,)`` matches torch's
    buffer. Per-axis coords are normalised to ``[-1, 1]``, scaled to ``[-8, 8]``,
    then softened by ``sign(c)·log₂(|c|+1)/log₂8``.
    """
    coords = [jnp.arange(-(w - 1), w, dtype=jnp.float32) for w in window_size]
    table = jnp.stack(jnp.meshgrid(*coords, indexing="ij"), axis=-1)
    norms = jnp.asarray([max(w - 1, 1) for w in window_size], dtype=jnp.float32)
    table = 8.0 * (table / norms)
    table = (
        jnp.sign(table)
        * jnp.log2(jnp.abs(table) + 1.0)
        / jnp.log2(jnp.asarray(8.0, dtype=jnp.float32))
    )
    return table[None, ...]


def _build_rpb_idx(window_size: Sequence[int]) -> jnp.ndarray:
    """Integer index ``(sl, sl)`` gathering per-pair biases from the flat table."""
    import numpy as _np
    space = len(window_size)
    grids = _np.stack(
        _np.meshgrid(*[_np.arange(w) for w in window_size], indexing="ij")
    )
    flat = grids.reshape(space, -1)
    dists = flat[:, :, None] - flat[:, None, :]
    out = _np.zeros_like(dists[0])
    for i in range(space):
        stride = 1
        for j in range(i + 1, space):
            stride *= 2 * window_size[j] - 1
        out = out + (dists[i] + (window_size[i] - 1)) * stride
    return jnp.asarray(out, dtype=jnp.int32)


class RPB(eqx.Module):
    """SwinV2 relative position bias.

    Faithful port of ``neugk/models/nd_vit/positional.py:RPB``:

    1. ``rpb``: frozen continuous coordinate table, shape ``(1, *(2w-1), space)``.
    2. ``rpb_idx``: integer ``(sl, sl)`` gather indices into that flat table.
    3. ``cpb_mlp``: ``space → 512 → heads`` (output linear has ``bias=False``).
    4. Output bias = ``16·sigmoid(cpb_mlp(table)[rpb_idx])`` reshaped to
       ``(heads, sl, sl)`` for the attention logits.
    """

    cpb_mlp: MLP
    rpb: jax.Array
    rpb_idx: jax.Array
    num_heads: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)
    space: int = eqx.field(static=True)

    def __init__(
        self,
        window_size: Sequence[int],
        num_heads: int,
        *,
        key,
        hidden: int = 512,
    ):
        space = len(window_size)
        kw1, kw2 = jr.split(key, 2)
        self.cpb_mlp = MLP([space, hidden, num_heads], key=kw1, act_fn=relu)
        last = self.cpb_mlp.layers[-1]
        self.cpb_mlp = eqx.tree_at(
            lambda m: m.layers[-1],
            self.cpb_mlp,
            Linear(last.weight.shape[1], last.weight.shape[0], key=kw2, use_bias=False),
        )
        self.rpb = jax.lax.stop_gradient(_build_rpb_table(window_size))
        self.rpb_idx = jax.lax.stop_gradient(_build_rpb_idx(window_size))
        seq_len = 1
        for w in window_size:
            seq_len *= w
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.space = space

    def __call__(self) -> jax.Array:
        bias_table = self.cpb_mlp(self.rpb)
        bias_flat = bias_table.reshape(-1, self.num_heads)
        bias = bias_flat[self.rpb_idx.reshape(-1)]
        bias = bias.reshape(self.seq_len, self.seq_len, self.num_heads)
        bias = 16.0 * jax.nn.sigmoid(bias)
        return jnp.transpose(bias, (2, 0, 1))
