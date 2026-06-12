"""N-dimensional Swin transformer layers (shifted-window attention)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.models.attention import MultiHeadSelfAttention
from neugk_jax.models.utils import MLP, LayerNorm, RMSNorm, DiTModulation, gelu
from neugk_jax.models.patching import pad_to_blocks, unpad


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _effective_window(grid_size, window_size):
    """Per-axis effective window: never larger than the grid; 0/None → no partition."""
    eff = []
    for g, w in zip(grid_size, window_size):
        if w is None or w == 0 or w >= g:
            eff.append(g)
        else:
            eff.append(w)
    return tuple(eff)


def _build_partition_grid(grid_size, window_size):
    """Return (effective_window_size, partition_grid) where each grid axis = g // w."""
    eff = _effective_window(grid_size, window_size)
    grid = tuple(g // w for g, w in zip(grid_size, eff))
    return eff, grid


def window_partition(x: jnp.ndarray, window_size: Sequence[int]) -> jnp.ndarray:
    """``(*spatial, dim) → (num_windows, prod(window), dim)``."""
    n = len(window_size)
    spatial = x.shape[:n]
    dim = x.shape[-1]
    new_shape = []
    for s, w in zip(spatial, window_size):
        new_shape.extend([s // w, w])
    new_shape.append(dim)
    x = x.reshape(new_shape)
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2)) + [2 * n]
    x = jnp.transpose(x, perm)
    return x.reshape(-1, _prod(window_size), dim)


def window_reverse(
    windows: jnp.ndarray, window_size: Sequence[int], spatial: Sequence[int]
) -> jnp.ndarray:
    """Inverse of window_partition."""
    n = len(window_size)
    dim = windows.shape[-1]
    grid = tuple(s // w for s, w in zip(spatial, window_size))
    x = windows.reshape(*grid, *window_size, dim)
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(i + n)
    perm.append(2 * n)
    x = jnp.transpose(x, perm)
    return x.reshape(*spatial, dim)


def _pad_to_window(spatial: Sequence[int], window_size: Sequence[int]) -> tuple[int, ...]:
    """Per-axis pad amounts so each spatial dim becomes a multiple of its window."""
    pads = []
    for s, w in zip(spatial, window_size):
        r = s % w
        pads.append(0 if r == 0 else w - r)
    return tuple(pads)


def _build_shift_mask(
    spatial: Sequence[int], window_size: Sequence[int], shift: Sequence[int]
) -> Optional[jnp.ndarray]:
    """Pre-compute the cyclic-shift attention mask on the *padded* grid.

    Returns ``(num_windows, W, W)`` additive bias (0 / -inf) or None when
    no shift applies on any axis.
    """
    if all(s == 0 for s in shift):
        return None
    pads = _pad_to_window(spatial, window_size)
    padded = tuple(s + p for s, p in zip(spatial, pads))
    # build a per-position region id by counting slices on each axis
    region = np.zeros(padded, dtype=np.int32)
    for axis, (size, w, sh) in enumerate(zip(padded, window_size, shift)):
        if sh == 0:
            continue
        slices = [(0, size - w), (size - w, size - sh), (size - sh, size)]
        for i, (lo, hi) in enumerate(slices):
            idx = [slice(None)] * len(padded)
            idx[axis] = slice(lo, hi)
            region[tuple(idx)] += i * (10 ** axis)
    region = jnp.asarray(region)[..., None]  # (*padded, 1)
    win = window_partition(region, window_size)  # (n_win, W, 1)
    win = win[..., 0]
    mask = win[:, :, None] != win[:, None, :]  # (n_win, W, W)
    return jnp.where(mask, -1e9, 0.0).astype(jnp.float32)


class _DropPath(eqx.Module):
    """Stochastic depth on a residual branch. ``rate=0`` is a no-op."""

    rate: float = eqx.field(static=True)

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = False) -> jnp.ndarray:
        if inference or self.rate == 0.0 or key is None:
            return x
        keep = 1.0 - self.rate
        mask = jr.bernoulli(key, p=keep, shape=()).astype(x.dtype)
        return x * mask / keep


class SwinBlock(eqx.Module):
    """One shifted-window transformer block (norm → MSA → norm → MLP)."""

    norm1: object
    norm2: object
    attn: MultiHeadSelfAttention
    mlp: MLP
    drop_path: _DropPath
    grid_size: tuple[int, ...] = eqx.field(static=True)
    window_size: tuple[int, ...] = eqx.field(static=True)
    shift_size: tuple[int, ...] = eqx.field(static=True)
    attn_mask: Optional[jax.Array]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        *,
        key,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
    ):
        eff_w, _ = _build_partition_grid(grid_size, window_size)
        if shift:
            # matches upstream get_window_size: skip shift on axes where window covers the full grid
            shift_size = tuple(
                (e // 2) if (g > w and e > 1) else 0
                for g, w, e in zip(grid_size, window_size, eff_w)
            )
        else:
            shift_size = tuple(0 for _ in eff_w)
        self.grid_size = tuple(grid_size)
        self.window_size = eff_w
        self.shift_size = shift_size

        self.norm1 = RMSNorm(dim, elementwise_affine=norm_affine) if rms_norm else LayerNorm(dim, elementwise_affine=norm_affine)
        self.norm2 = RMSNorm(dim, elementwise_affine=norm_affine) if rms_norm else LayerNorm(dim, elementwise_affine=norm_affine)
        katt, kmlp = jr.split(key, 2)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, key=katt,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            use_rpb=use_rpb, gated_attention=gated_attention,
            window_size=eff_w,
        )
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = MLP([dim, hidden, dim], key=kmlp, act_fn=act_fn)
        self.drop_path = _DropPath(drop_path)
        self.attn_mask = _build_shift_mask(self.grid_size, eff_w, shift_size)

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = False) -> jnp.ndarray:
        """SwinV2 post-norm forward with upstream's doubled residual.

        Mirrors ``neugk/models/nd_vit/swin_layers.py:SwinTransformerBlock.forward``:

        * ``forward_part1`` runs attention on the un-normed input and
          applies ``norm1`` to the result (post-norm).
        * ``forward_part2`` runs ``mlp`` then drop_path then ``norm2``.
        * The combine pattern is::

              x_res1 = shortcut + dp(norm1(attn_part(x)))
              out    = x_res1 + (x_res1 + norm2(dp(mlp(x_res1))))
                     = 2 * x_res1 + norm2(dp(mlp(x_res1)))

          — i.e. the residual_1 input is accumulated *twice*.
        """
        spatial = x.shape[:-1]
        shortcut = x  # upstream self.skip is Identity (dim_out == dim)

        h, pads = pad_to_blocks(x, self.window_size)
        padded_spatial = h.shape[:-1]
        if any(s > 0 for s in self.shift_size):
            h = jnp.roll(h, shift=[-s for s in self.shift_size],
                         axis=list(range(len(padded_spatial))))
        windows = window_partition(h, self.window_size)  # (n_win, W, dim) per block
        if self.attn_mask is not None:
            windows = jax.vmap(lambda w, m: self.attn(w, attn_bias=m[None]))(windows, self.attn_mask)
        else:
            windows = jax.vmap(lambda w: self.attn(w))(windows)
        h = window_reverse(windows, self.window_size, padded_spatial)
        if any(s > 0 for s in self.shift_size):
            h = jnp.roll(h, shift=list(self.shift_size),
                         axis=list(range(len(padded_spatial))))
        h = unpad(h, pads, spatial)
        h = self.norm1(h)  # post-norm per SwinV2 convention

        key1, key2 = (None, None) if key is None else jr.split(key, 2)
        x_res1 = shortcut + self.drop_path(h, key=key1, inference=inference)

        mlp_out = self.norm2(self.drop_path(self.mlp(x_res1), key=key2, inference=inference))

        return x_res1 + x_res1 + mlp_out


class DiTSwinBlock(eqx.Module):
    """SwinBlock with DiT-style 6-way conditioning modulation."""

    norm1: object
    norm2: object
    attn: MultiHeadSelfAttention
    mlp: MLP
    drop_path: _DropPath
    mod: DiTModulation
    grid_size: tuple[int, ...] = eqx.field(static=True)
    window_size: tuple[int, ...] = eqx.field(static=True)
    shift_size: tuple[int, ...] = eqx.field(static=True)
    attn_mask: Optional[jax.Array]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        *,
        key,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
    ):
        eff_w, _ = _build_partition_grid(grid_size, window_size)
        # same upstream-faithful shift rule as SwinBlock above
        shift_size = tuple(
            (e // 2) if (shift and g > w and e > 1) else 0
            for g, w, e in zip(grid_size, window_size, eff_w)
        )
        self.grid_size = tuple(grid_size)
        self.window_size = eff_w
        self.shift_size = shift_size

        self.norm1 = LayerNorm(dim, elementwise_affine=False)
        self.norm2 = LayerNorm(dim, elementwise_affine=False)
        katt, kmlp, kmod = jr.split(key, 3)
        self.attn = MultiHeadSelfAttention(dim, num_heads, key=katt)
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = MLP([dim, hidden, dim], key=kmlp, act_fn=act_fn)
        self.drop_path = _DropPath(drop_path)
        self.mod = DiTModulation(cond_dim, dim, key=kmod)
        self.attn_mask = _build_shift_mask(self.grid_size, eff_w, shift_size)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, *, key=None, inference=False) -> jnp.ndarray:
        spatial = x.shape[:-1]
        # upstream order: (scale1, shift1, gate1, scale2, shift2, gate2) from DiTModulation
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.mod(cond)
        def _bc(t):
            for _ in range(len(spatial)):
                t = t[None, ...]
            return t

        shift_msa, scale_msa, gate_msa = _bc(shift_msa), _bc(scale_msa), _bc(gate_msa)
        shift_mlp, scale_mlp, gate_mlp = _bc(shift_mlp), _bc(scale_mlp), _bc(gate_mlp)

        shortcut = x
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa

        h, pads = pad_to_blocks(h, self.window_size)
        padded_spatial = h.shape[:-1]

        if any(s > 0 for s in self.shift_size):
            h = jnp.roll(h, shift=[-s for s in self.shift_size],
                         axis=list(range(len(padded_spatial))))
        windows = window_partition(h, self.window_size)
        if self.attn_mask is not None:
            windows = jax.vmap(lambda w, m: self.attn(w, attn_bias=m[None]))(windows, self.attn_mask)
        else:
            windows = jax.vmap(lambda w: self.attn(w))(windows)
        h = window_reverse(windows, self.window_size, padded_spatial)
        if any(s > 0 for s in self.shift_size):
            h = jnp.roll(h, shift=list(self.shift_size),
                         axis=list(range(len(padded_spatial))))
        h = unpad(h, pads, spatial)

        key1, key2 = (None, None) if key is None else jr.split(key, 2)
        x = shortcut + gate_msa * self.drop_path(h, key=key1, inference=inference)
        h2 = self.mlp(self.norm2(x) * (1.0 + scale_mlp) + shift_mlp)
        x = x + gate_mlp * self.drop_path(h2, key=key2, inference=inference)
        return x


class SwinLayer(eqx.Module):
    """Stack of ``depth`` Swin blocks alternating non-shifted / shifted."""

    blocks: list[SwinBlock]
    dim: int = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    window_size: tuple[int, ...] = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)
    drop_path: float = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)
    norm_layer: type = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        *,
        key,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        norm_layer: type = LayerNorm,
        use_checkpoint: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        **_unused,
    ):
        keys = jr.split(key, depth)
        self.blocks = [
            SwinBlock(
                dim,
                num_heads,
                grid_size,
                window_size,
                key=keys[i],
                shift=bool(i % 2),
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                act_fn=act_fn,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                use_rpb=use_rpb,
                gated_attention=gated_attention,
                norm_affine=norm_affine,
            )
            for i in range(depth)
        ]
        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.window_size = tuple(window_size)
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.act_fn = act_fn

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = False, **_) -> jnp.ndarray:
        keys = jr.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for blk, k in zip(self.blocks, keys):
            call = blk if not self.use_checkpoint else eqx.filter_checkpoint(blk)
            x = call(x, key=k, inference=inference)
        return x


class DiTSwinLayer(eqx.Module):
    """Stack of ``depth`` DiTSwin blocks; takes a per-sample condition."""

    blocks: list[DiTSwinBlock]
    dim: int = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    window_size: tuple[int, ...] = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)
    drop_path: float = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)
    norm_layer: type = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        *,
        key,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        norm_layer: type = LayerNorm,
        use_checkpoint: bool = False,
        **_unused,
    ):
        keys = jr.split(key, depth)
        self.blocks = [
            DiTSwinBlock(
                dim,
                num_heads,
                cond_dim,
                grid_size,
                window_size,
                key=keys[i],
                shift=bool(i % 2),
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                act_fn=act_fn,
            )
            for i in range(depth)
        ]
        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.window_size = tuple(window_size)
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.act_fn = act_fn

    def __call__(self, x, condition, *, key=None, inference=False, **_):
        keys = jr.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for blk, k in zip(self.blocks, keys):
            call = blk if not self.use_checkpoint else eqx.filter_checkpoint(blk)
            x = call(x, condition, key=k, inference=inference)
        return x
