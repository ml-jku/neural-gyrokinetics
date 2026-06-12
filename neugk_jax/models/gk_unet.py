"""N-dimensional Swin-UNet backbone used by the autoencoder.

Mirrors the torch ``SwinNDUnet`` / ``Swin5DUnet`` structure but drops the
PINC-only branches (flux head, simsiam, mask augmentation). For M1 only
the un-conditioned forward path is wired; DiT-conditioned forward will be
added when the diffusion model is composed.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.models.embeddings import APE
from neugk_jax.models.utils import LayerNorm, Linear, gelu
from neugk_jax.models.patching import PatchEmbed, PatchExpand, PatchMerge, pad_to_blocks, unpad
from neugk_jax.models.swin import SwinLayer
from neugk_jax.models.vit import LayerModes, ViTLayer


def _as_seq(x, n):
    if isinstance(x, int):
        return [x] * n
    return list(x)


class SwinBlockDown(eqx.Module):
    """Encoder stage: optional APE → SwinLayer (or DiTSwinLayer) → PatchMerge.

    Pass ``cond_dim>0`` at construction to swap the inner ``SwinLayer`` for a
    DiT-conditioned variant; the ``__call__`` then accepts a ``condition`` arg.
    """

    pos_embed: Optional[APE]
    swin: object  # SwinLayer or DiTSwinLayer
    downsample: PatchMerge
    resampled_grid_size: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    use_cond: bool = eqx.field(static=True)

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        num_heads: int,
        depth: int,
        *,
        key,
        use_abs_pe: bool = False,
        drop_path: float = 0.1,
        mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        act_fn: Callable = gelu,
        use_checkpoint: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
        cond_dim: Optional[int] = None,
    ):
        k1, k2, _ = jr.split(key, 3)
        self.pos_embed = APE(dim, grid_size, init="sincos") if use_abs_pe else None
        self.use_cond = cond_dim is not None and cond_dim > 0
        if self.use_cond:
            from neugk_jax.models.swin import DiTSwinLayer
            self.swin = DiTSwinLayer(
                space, dim, depth=depth, num_heads=num_heads,
                grid_size=grid_size, window_size=window_size,
                cond_dim=cond_dim,
                key=k1, mlp_ratio=mlp_ratio, drop_path=drop_path,
                act_fn=act_fn, use_checkpoint=use_checkpoint,
            )
        else:
            self.swin = SwinLayer(
                space, dim, depth=depth, num_heads=num_heads,
                grid_size=grid_size, window_size=window_size,
                key=k1, mlp_ratio=mlp_ratio, drop_path=drop_path,
                act_fn=act_fn, use_checkpoint=use_checkpoint,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                use_rpb=use_rpb, gated_attention=gated_attention,
                norm_affine=norm_affine, rms_norm=rms_norm,
            )
        self.downsample = PatchMerge(
            dim, grid_size, key=k2, c_multiplier=c_multiplier, rms_norm=rms_norm,
        )
        self.resampled_grid_size = self.downsample.target_grid_size
        self.out_dim = self.downsample.out_dim

    def __call__(self, x, condition=None, *, key=None, inference=True, return_skip: bool = True):
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        if self.use_cond:
            x = self.swin(x, condition, key=key, inference=inference)
        else:
            x = self.swin(x, key=key, inference=inference)
        merged = self.downsample(x)
        return (merged, x) if return_skip else merged


class SwinBlockUp(eqx.Module):
    """Decoder stage: optional skip-concat → SwinLayer (or DiTSwinLayer) → PatchExpand."""

    proj_concat: Optional[Linear]
    pos_embed: Optional[APE]
    swin: object  # SwinLayer or DiTSwinLayer
    upsample: Optional[PatchExpand]
    resampled_grid_size: tuple[int, ...] = eqx.field(static=True)
    mode: LayerModes = eqx.field(static=True)
    use_cond: bool = eqx.field(static=True)

    def __init__(
        self,
        space: int,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        depth: int,
        num_heads: int,
        *,
        key,
        target_grid_size: Optional[Sequence[int]] = None,
        use_abs_pe: bool = False,
        drop_path: float = 0.1,
        mlp_ratio: float = 2.0,
        c_multiplier: int = 2,
        act_fn: Callable = gelu,
        use_checkpoint: bool = False,
        mode: LayerModes = LayerModes.UPSAMPLE,
        use_skip: bool = True,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
        cond_dim: Optional[int] = None,
    ):
        k1, k2, k3 = jr.split(key, 3)
        if use_skip:
            self.proj_concat = Linear(2 * dim, dim, key=k1)
        else:
            self.proj_concat = None
        self.pos_embed = APE(dim, grid_size, init="sincos") if use_abs_pe else None
        self.use_cond = cond_dim is not None and cond_dim > 0
        if self.use_cond:
            from neugk_jax.models.swin import DiTSwinLayer
            self.swin = DiTSwinLayer(
                space, dim, depth=depth, num_heads=num_heads,
                grid_size=grid_size, window_size=window_size,
                cond_dim=cond_dim,
                key=k2, mlp_ratio=mlp_ratio, drop_path=drop_path,
                act_fn=act_fn, use_checkpoint=use_checkpoint,
            )
        else:
            self.swin = SwinLayer(
                space, dim, depth=depth, num_heads=num_heads,
                grid_size=grid_size, window_size=window_size,
                key=k2, mlp_ratio=mlp_ratio, drop_path=drop_path,
                act_fn=act_fn, use_checkpoint=use_checkpoint,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                use_rpb=use_rpb, gated_attention=gated_attention,
                norm_affine=norm_affine, rms_norm=rms_norm,
            )
        if mode == LayerModes.UPSAMPLE:
            self.upsample = PatchExpand(
                dim, grid_size, key=k3,
                c_multiplier=c_multiplier,
                expand_by=2,
                target_grid_size=target_grid_size,
                mlp_depth=1,
                rms_norm=rms_norm,
            )
            self.resampled_grid_size = self.upsample.target_grid_size
        else:
            self.upsample = None
            self.resampled_grid_size = tuple(grid_size)
        self.mode = mode

    def __call__(self, x, s=None, condition=None, *, key=None, inference=True):
        if self.proj_concat is not None and s is not None:
            x = self.proj_concat(jnp.concatenate([x, s], axis=-1))
            x = jax.nn.gelu(x)
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        if self.use_cond:
            x = self.swin(x, condition, key=key, inference=inference)
        else:
            x = self.swin(x, key=key, inference=inference)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinNDUnet(eqx.Module):
    """N-dimensional Swin U-Net used as the AE backbone.

    Forward operates on a single unbatched sample of shape ``(C, *spatial)`` —
    callers vmap externally.
    """

    patch_embed: PatchEmbed
    down_blocks: list[SwinBlockDown]
    middle: ViTLayer
    middle_pe: Optional[APE]
    middle_upscale: PatchExpand
    up_blocks: list[SwinBlockUp]
    unpatch: PatchExpand

    # static shape / config fields
    space: int = eqx.field(static=True)
    base_resolution: tuple[int, ...] = eqx.field(static=True)
    padded_base_resolution: tuple[int, ...] = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)
    window_size: tuple[int, ...] = eqx.field(static=True)
    grid_sizes: tuple = eqx.field(static=True)
    down_dims: tuple = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    norm_output: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: int,
        dim: int,
        base_resolution: Sequence[int],
        in_channels: int,
        out_channels: int,
        patch_size,
        window_size,
        depth,
        num_heads,
        num_layers: int = 4,
        middle_depth: int = 2,
        middle_num_heads: int = 8,
        use_abs_pe: bool = False,
        c_multiplier: int = 2,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        merging_depth: int = 2,
        unmerging_depth: int = 2,
        act_fn: Callable = gelu,
        norm_output: bool = False,
        use_checkpoint: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
        up_use_skip: bool = True,
        cond_dim: Optional[int] = None,
        key,
    ):
        patch_size = _as_seq(patch_size, space)
        window_size = _as_seq(window_size, space)
        depth = _as_seq(depth, num_layers)
        num_heads = _as_seq(num_heads, num_layers)
        # right-pad base resolution to a multiple of patch_size
        padded_base = []
        for s, p in zip(base_resolution, patch_size):
            if p in (0, 1):
                padded_base.append(s)
            else:
                r = s % p
                padded_base.append(s if r == 0 else s + (p - r))

        keys = jr.split(key, num_layers * 2 + 4)
        ki = 0

        self.patch_embed = PatchEmbed(
            padded_base, patch_size, in_channels=in_channels, embed_dim=dim,
            key=keys[ki], mlp_depth=merging_depth, mlp_ratio=merging_hidden_ratio,
            rms_norm=rms_norm,
        )
        ki += 1
        grid_sizes = [self.patch_embed.grid_size]
        down_dims = [dim]
        down_blocks = []
        for i in range(num_layers):
            blk = SwinBlockDown(
                space, down_dims[i], grid_size=grid_sizes[i],
                window_size=window_size, num_heads=num_heads[i], depth=depth[i],
                key=keys[ki], use_abs_pe=use_abs_pe, drop_path=drop_path,
                mlp_ratio=hidden_mlp_ratio, c_multiplier=c_multiplier,
                act_fn=act_fn, use_checkpoint=use_checkpoint,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                use_rpb=use_rpb, gated_attention=gated_attention,
                norm_affine=norm_affine, rms_norm=rms_norm,
                cond_dim=cond_dim,
            )
            ki += 1
            down_blocks.append(blk)
            down_dims.append(blk.out_dim)
            grid_sizes.append(blk.resampled_grid_size)

        self.down_blocks = down_blocks
        self.grid_sizes = tuple(grid_sizes)
        self.down_dims = tuple(down_dims)

        # middle: global ViT attention at the deepest grid
        self.middle = ViTLayer(
            space, down_dims[-1], depth=middle_depth, num_heads=middle_num_heads,
            grid_size=grid_sizes[-1], key=keys[ki],
            mlp_ratio=hidden_mlp_ratio, drop_path=drop_path,
            act_fn=act_fn, use_checkpoint=use_checkpoint,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            gated_attention=gated_attention, norm_affine=norm_affine,
        )
        ki += 1
        self.middle_pe = APE(down_dims[-1], grid_sizes[-1], init="sincos") if use_abs_pe else None
        # upstream middle_upscale uses LayerNorm (unlike PatchMerge which uses RMSNorm)
        self.middle_upscale = PatchExpand(
            down_dims[-1], grid_sizes[-1], key=keys[ki],
            target_grid_size=grid_sizes[-2], c_multiplier=c_multiplier,
            mlp_depth=1, rms_norm=False,
        )
        ki += 1

        # up path
        up_dims = down_dims[::-1][1:]
        up_grid_sizes = grid_sizes[::-1][1:]
        up_blocks = []
        up_common = dict(
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            use_rpb=use_rpb, gated_attention=gated_attention,
            norm_affine=norm_affine, use_skip=up_use_skip,
            rms_norm=rms_norm, cond_dim=cond_dim,
        )
        for i in range(num_layers - 1):
            up_blocks.append(
                SwinBlockUp(
                    space, up_dims[i], grid_size=up_grid_sizes[i],
                    target_grid_size=up_grid_sizes[i + 1],
                    window_size=window_size, num_heads=num_heads[::-1][i],
                    depth=depth[::-1][i],
                    key=keys[ki], use_abs_pe=use_abs_pe, drop_path=drop_path,
                    mlp_ratio=hidden_mlp_ratio, c_multiplier=c_multiplier,
                    act_fn=act_fn, use_checkpoint=use_checkpoint,
                    **up_common,
                )
            )
            ki += 1
        # final decoder block: no upsample, SEQUENCE mode
        up_blocks.append(
            SwinBlockUp(
                space, up_dims[-1], grid_size=up_grid_sizes[-1],
                window_size=window_size, num_heads=num_heads[::-1][-1],
                depth=depth[::-1][-1], key=keys[ki],
                use_abs_pe=use_abs_pe, drop_path=drop_path,
                mlp_ratio=hidden_mlp_ratio, c_multiplier=c_multiplier,
                act_fn=act_fn, use_checkpoint=use_checkpoint,
                mode=LayerModes.SEQUENCE,
                **up_common,
            )
        )
        ki += 1
        self.up_blocks = up_blocks

        # unpatch: expand back to padded base resolution (norm=False, matching upstream)
        self.unpatch = PatchExpand(
            up_dims[-1], up_grid_sizes[-1], key=keys[ki],
            expand_by=tuple(p if p > 0 else 1 for p in patch_size),
            out_channels=out_channels,
            mlp_depth=unmerging_depth, mlp_ratio=unmerging_hidden_ratio,
            norm=False,
        )
        # assign static fields
        self.space = space
        self.base_resolution = tuple(base_resolution)
        self.padded_base_resolution = tuple(padded_base)
        self.patch_size = tuple(patch_size)
        self.window_size = tuple(window_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_output = norm_output

    # forward path

    def patch_encode(self, x: jnp.ndarray):
        # x: (C, *spatial) → (*spatial, C) → pad → patch_embed
        x = jnp.moveaxis(x, 0, -1)
        x, pad_axes = pad_to_blocks(x, self.patch_size)
        x = self.patch_embed(x)
        return x, pad_axes

    def patch_decode(self, z: jnp.ndarray, pad_axes) -> jnp.ndarray:
        x = self.unpatch(z)
        x = unpad(x, pad_axes, self.base_resolution)
        return jnp.moveaxis(x, -1, 0)

    def __call__(self, x: jnp.ndarray, *, key=None, inference=True):
        z, pad_axes = self.patch_encode(x)
        skips = []
        for blk in self.down_blocks:
            z, skip = blk(z, return_skip=True, inference=inference)
            skips.append(skip)
        if self.middle_pe is not None:
            z = self.middle_pe(z)
        z = self.middle(z, inference=inference)
        z = self.middle_upscale(z)
        for blk, skip in zip(self.up_blocks, reversed(skips)):
            z = blk(z, skip, inference=inference)
        return self.patch_decode(z, pad_axes)


class Swin5DUnet(SwinNDUnet):
    """5D wrapper with optional ``decouple_mu`` collapse.

    When ``decouple_mu=True`` the mu axis (index 1 after channels) is folded
    into the channel dimension and a learned ``vel_pe`` is added to provide
    velocity-space positional information.
    """

    decouple_mu: bool = eqx.field(static=True)
    full_resolution: tuple[int, ...] = eqx.field(static=True)
    decoupled_dim: int = eqx.field(static=True)
    original_in_channels: int = eqx.field(static=True)
    original_out_channels: int = eqx.field(static=True)
    vel_pe: Optional[APE]

    def __init__(
        self,
        *,
        space: int = 5,
        decouple_mu: bool = False,
        base_resolution: Sequence[int],
        in_channels: int,
        out_channels: int,
        patch_size,
        window_size,
        key,
        **kwargs,
    ):
        full_resolution = tuple(base_resolution)
        full_in = in_channels
        full_out = out_channels
        if decouple_mu:
            space = 4
            # drop mu axis (index 1) from spatial and patch/window specs
            patch_size = list(patch_size)
            window_size = list(window_size)
            decoupled_dim = base_resolution[1]
            base_resolution = [base_resolution[0]] + list(base_resolution[2:])
            patch_size = [patch_size[0]] + list(patch_size[2:])
            window_size = [window_size[0]] + list(window_size[2:])
            in_channels = full_in * decoupled_dim
            out_channels = full_out * decoupled_dim
        else:
            decoupled_dim = 0

        super().__init__(
            space=space,
            base_resolution=base_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            window_size=window_size,
            key=key,
            **kwargs,
        )
        self.decouple_mu = decouple_mu
        self.full_resolution = full_resolution
        self.decoupled_dim = decoupled_dim
        self.original_in_channels = full_in
        self.original_out_channels = full_out
        if decouple_mu:
            # learnable pe for the collapsed mu axis; shape mirrors torch vel_pe buffer
            vel_pe_resolution = (1, decoupled_dim, 1, 1, 1)
            self.vel_pe = APE(
                full_in, vel_pe_resolution, init="normal", learnable=True,
                key=jr.PRNGKey(0),
            )
        else:
            self.vel_pe = None

    def patch_encode(self, df: jnp.ndarray):
        # df: (C, vp, mu, s, x, y) → channel last → add vel_pe → collapse mu
        df = jnp.moveaxis(df, 0, -1)  # (vp, mu, s, x, y, C)
        if self.decouple_mu:
            df = self.vel_pe(df)
            # (vp, mu, s, x, y, C) -> (vp, s, x, y, C*mu)
            df = rearrange(df, "vp mu s x y c -> vp s x y (c mu)")
        df, pad_axes = pad_to_blocks(df, self.patch_size)
        df = self.patch_embed(df)
        return df, pad_axes

    def patch_decode(self, z: jnp.ndarray, pad_axes) -> jnp.ndarray:
        df = self.unpatch(z)
        df = unpad(df, pad_axes, self.base_resolution)
        if self.decouple_mu:
            # reshape (vp, s, x, y, C*mu) back to (C, vp, mu, s, x, y)
            df = rearrange(df, "vp s x y (c mu) -> c vp mu s x y", mu=self.decoupled_dim)
            return df
        return jnp.moveaxis(df, -1, 0)
