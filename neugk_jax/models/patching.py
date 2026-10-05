"""N-dimensional patch operations.

The three modules below all share a single spatial primitive — N-D
**fold / unfold** (a.k.a. im2col / col2im) — and differ only in the linear
mixer applied to the channel axis. Conceptually:

* ``PatchEmbed``  =  fold + MLP project up
* ``PatchMerge``  =  fold (patch=2) + norm + linear project up
* ``PatchExpand`` =  linear/MLP project + unfold + (optional crop)

Inputs are unbatched and shaped ``(*spatial, channels)``. Batched callers
``jax.vmap`` over the leading axis.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.utils import MLP, Linear, leaky_relu, make_norm


def _normalize_patch(patch_size: Sequence[int]) -> tuple[int, ...]:
    # 0/none entries become 1
    return tuple(p if p and p > 0 else 1 for p in patch_size)


def fold_patches(x: jnp.ndarray, patch_size: Sequence[int]) -> jnp.ndarray:
    """``(*spatial, c) → (*grid, prod(patch)*c)`` where ``grid_i = spatial_i // patch_i``.

    Generic N-D im2col: reshape each axis into ``(grid, patch)``, transpose
    so all grid axes come first and all patch axes second, then flatten the
    patch + channel suffix. Axes with ``patch=1`` are passthrough.
    """
    ps = _normalize_patch(patch_size)
    n = len(ps)
    spatial = x.shape[:n]
    new_shape = []
    for s, p in zip(spatial, ps):
        new_shape.extend([s // p, p])
    x = x.reshape(*new_shape, x.shape[-1])
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2)) + [2 * n]
    x = jnp.transpose(x, perm)
    return x.reshape(*(s // p for s, p in zip(spatial, ps)), -1)


def unfold_patches(
    x: jnp.ndarray, expand_by: Sequence[int], *, out_channels: Optional[int] = None
) -> jnp.ndarray:
    """``(*grid, prod(expand)*out_c) → (*expanded, out_c)``. Inverse of ``fold_patches``."""
    eb = _normalize_patch(expand_by)
    n = len(eb)
    grid = x.shape[:n]
    if out_channels is None:
        out_channels = x.shape[-1] // math.prod(eb)
    x = x.reshape(*grid, *eb, out_channels)
    perm = [a for i in range(n) for a in (i, i + n)] + [2 * n]
    x = jnp.transpose(x, perm)
    return x.reshape(*[g * e for g, e in zip(grid, eb)], out_channels)


def pad_amounts(spatial: Sequence[int], block_size: Sequence[int]) -> tuple[int, ...]:
    return tuple(-s % b for s, b in zip(spatial, _normalize_patch(block_size)))


def pad_to_blocks(x: jnp.ndarray, block_size: Sequence[int]) -> jnp.ndarray:
    pads = pad_amounts(x.shape[: len(block_size)], block_size)
    return jnp.pad(x, [(0, p) for p in pads] + [(0, 0)] * (x.ndim - len(pads)))


def unpad(x: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
    return x[tuple(slice(0, s) for s in shape)]


class PatchEmbed(eqx.Module):
    """Fold + MLP channel mixer.

    Input  ``(*spatial, in_channels)`` → output ``(*grid, embed_dim)``.
    The MLP is stored under ``patch`` (``patch_embed.patch.mlp.{0,3}.{weight,bias}``).
    """

    patch: MLP
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        base_resolution: Sequence[int],
        patch_size: Sequence[int],
        in_channels: int,
        embed_dim: int,
        *,
        key,
        mlp_depth: int = 2,
        mlp_ratio: float = 8.0,
        act_fn=leaky_relu,
    ):
        ps = _normalize_patch(patch_size)
        self.patch_size = ps
        self.grid_size = tuple(s // p for s, p in zip(base_resolution, ps))
        # hidden = embed_dim * mlp_ratio, no max-clamp
        hidden = int(embed_dim * mlp_ratio)
        dims = [math.prod(ps) * in_channels] + [hidden] * (mlp_depth - 1) + [embed_dim]
        self.patch = MLP(dims, key=key, act_fn=act_fn, use_bias=False)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.patch(fold_patches(x, self.patch_size))


def merge_grid(grid_size: Sequence[int]) -> tuple[int, ...]:
    return tuple((g + 1) // 2 if g > 2 else g for g in grid_size)


class PatchMerge(eqx.Module):
    """Fold with patch=2 + norm + linear up-project.

    Halves every spatial axis with more than 2 patches; channels become ``dim * c_multiplier``.
    """

    proj: Linear
    norm: object
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        c_multiplier: int = 2,
        rms_norm: bool = False,
    ):
        self.patch_size = tuple(2 if g > 2 else 1 for g in grid_size)
        # odd-length axes round up; forward pads them to the next multiple
        self.target_grid_size = merge_grid(grid_size)
        in_features = dim * math.prod(self.patch_size)
        self.out_dim = dim * c_multiplier
        self.norm = make_norm(in_features, rms=rms_norm)
        self.proj = Linear(in_features, self.out_dim, key=key, use_bias=False)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = fold_patches(pad_to_blocks(x, self.patch_size), self.patch_size)
        return self.proj(self.norm(x))


class StridedConvTranspose(eqx.Module):
    """ConvTranspose with stride == kernel (non-overlapping) == the patch-expand op.

    The weight is laid out ``(in, out, *kernel)``; output placement is
    ``out[*(g_i*k_i), oc] = sum_ic x[*g, ic] * W[ic, oc, *k]`` (kernel block per grid
    cell), implemented as einsum + ``unfold_patches``.
    """

    weight: jax.Array  # (in, out, *kernel)
    bias: jax.Array  # (out,)
    expand_by: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, in_ch: int, out_ch: int, expand_by: Sequence[int], *, key):
        eb = tuple(expand_by)
        self.expand_by = eb
        self.weight = jr.normal(key, (in_ch, out_ch, *eb)) * 0.02
        self.bias = jnp.zeros((out_ch,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (*grid, in) -> (*grid*expand, out)
        n, out = len(self.expand_by), self.weight.shape[1]
        kl = "".join(chr(ord("p") + i) for i in range(n))
        y = jnp.einsum(f"...i,io{kl}->...{kl}o", x, self.weight)
        y = unfold_patches(y.reshape(*x.shape[:-1], -1), self.expand_by, out_channels=out)
        return y + self.bias


class PatchExpand(eqx.Module):
    """MLP channel mixer + unfold + optional crop.

    Upsamples spatial axes by ``expand_by`` and reduces channels by
    ``c_multiplier`` (or sets them to ``out_channels`` when given). The
    MLP is stored under ``expansion`` (``unpatch.expansion.mlp.0.weight`` etc.).
    """

    expansion: object  # mlp (mlp patch) or StridedConvTranspose (conv patch)
    proj_concat: Optional[Linear]
    modulation: Optional[object]  # Film, when cond_dim given (unpatch)
    norm: Optional[object]
    target_grid_size: tuple[int, ...] = eqx.field(static=True)
    expand_by: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    use_conv: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        *,
        key,
        c_multiplier: int = 2,
        expand_by: int | Sequence[int] = 2,
        target_grid_size: Optional[Sequence[int]] = None,
        out_channels: Optional[int] = None,
        mlp_depth: int = 1,
        mlp_ratio: float = 8.0,
        norm: bool = True,
        rms_norm: bool = False,
        use_conv: bool = False,
        patch_skip: bool = False,
        cond_dim: Optional[int] = None,
    ):
        gs = tuple(grid_size)
        if isinstance(expand_by, int):
            if target_grid_size is not None:
                # ceil so we never undershoot the target (crop after unfold)
                expand_by = [max(1, -(-t // max(1, g))) for g, t in zip(gs, target_grid_size)]
            else:
                expand_by = [expand_by if g > 1 else 1 for g in gs]
        eb = _normalize_patch(expand_by)
        if target_grid_size is None:
            target_grid_size = tuple(g * e for g, e in zip(gs, eb))
        self.target_grid_size = tuple(target_grid_size)
        self.expand_by = eb
        if out_channels is not None:
            inner = out_channels * math.prod(eb)
            self.out_dim = out_channels
        else:
            inner = max(1, (dim * math.prod(eb)) // c_multiplier)
            self.out_dim = max(1, dim // c_multiplier)

        self.use_conv = use_conv
        kexp, kpc, kmod = jr.split(key, 3)
        if use_conv:
            self.expansion = StridedConvTranspose(dim, self.out_dim, eb, key=kexp)
        else:
            # hidden = prod(expand_by) * mlp_ratio, not dim * mlp_ratio
            hidden = int(math.prod(eb) * mlp_ratio)
            dims = [dim] + [hidden] * (mlp_depth - 1) + [inner]
            self.expansion = MLP(dims, key=kexp, act_fn=leaky_relu, use_bias=True)
        # patch-skip residual projection (linear 2*dim->dim + leaky_relu) and film modulation
        self.proj_concat = Linear(2 * dim, dim, key=kpc) if patch_skip else None
        if cond_dim:
            from neugk_jax.models.swin import Film

            self.modulation = Film(cond_dim, dim, key=kmod)
        else:
            self.modulation = None
        # norm runs over out_dim channels after unfold
        self.norm = make_norm(self.out_dim, rms=rms_norm) if norm else None

    def __call__(self, x: jnp.ndarray, cond: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # order: proj_concat (skip residual) -> film -> expansion -> crop -> norm
        if self.proj_concat is not None:
            x = leaky_relu(self.proj_concat(x))
        if self.modulation is not None:
            x = self.modulation(x, cond)
        x = self.expansion(x)
        if not self.use_conv:
            x = unfold_patches(x, self.expand_by, out_channels=self.out_dim)
        # crop any overshoot from ceiling the expand factor
        x = unpad(x, self.target_grid_size)
        if self.norm is not None:
            x = self.norm(x)
        return x
