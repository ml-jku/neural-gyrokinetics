"""N-dimensional Swin U-Net backbone shared by the autoencoder and GyroSwin.

Drops the PINC-only branches (flux head, simsiam, mask augmentation).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.models.embeddings import APE, ContinuousConditionEmbed
from neugk_jax.models.patching import (
    PatchEmbed,
    PatchExpand,
    PatchMerge,
    _normalize_patch,
    merge_grid,
    pad_amounts,
    pad_to_blocks,
    unpad,
)
from neugk_jax.models.swin import BlockStack, swin_layer
from neugk_jax.models.utils import Linear, gelu


def _as_seq(x, n):
    if isinstance(x, int):
        return [x] * n
    return list(x)


class SwinBlockDown(eqx.Module):
    """Encoder stage: Swin layer (plain, FiLM- or DiT-conditioned) → PatchMerge."""

    swin: BlockStack
    downsample: PatchMerge
    resampled_grid_size: tuple[int, ...] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        num_heads: int,
        depth: int,
        *,
        key,
        c_multiplier: int = 2,
        rms_norm: bool = False,
        **layer_kw,
    ):
        k1, k2 = jr.split(key)
        self.swin = swin_layer(
            dim, depth, num_heads, grid_size, window_size, key=k1, rms_norm=rms_norm, **layer_kw
        )
        self.downsample = PatchMerge(
            dim, grid_size, key=k2, c_multiplier=c_multiplier, rms_norm=rms_norm
        )
        self.resampled_grid_size = self.downsample.target_grid_size
        self.out_dim = self.downsample.out_dim

    def __call__(self, x, condition=None, *, key=None, inference=True, return_skip: bool = True):
        x = self.swin(x, condition, key=key, inference=inference)
        merged = self.downsample(x)
        return (merged, x) if return_skip else merged


class SwinBlockUp(eqx.Module):
    """Decoder stage: optional skip-concat → Swin layer → optional PatchExpand.

    The Swin layer uses ``decoder_rms_norm`` for its norms, the upsample ``rms_norm``.
    """

    proj_concat: Optional[Linear]
    swin: BlockStack
    upsample: Optional[PatchExpand]
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        grid_size: Sequence[int],
        window_size: Sequence[int],
        depth: int,
        num_heads: int,
        *,
        key,
        target_grid_size: Optional[Sequence[int]] = None,
        c_multiplier: int = 2,
        act_fn: Callable = gelu,
        upsample: bool = True,
        use_skip: bool = True,
        rms_norm: bool = False,
        decoder_rms_norm: bool = False,
        **layer_kw,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.act_fn = act_fn
        self.proj_concat = Linear(2 * dim, dim, key=k1) if use_skip else None
        self.swin = swin_layer(
            dim,
            depth,
            num_heads,
            grid_size,
            window_size,
            key=k2,
            act_fn=act_fn,
            rms_norm=decoder_rms_norm,
            **layer_kw,
        )
        self.upsample = None
        if upsample:
            self.upsample = PatchExpand(
                dim,
                grid_size,
                key=k3,
                c_multiplier=c_multiplier,
                expand_by=2,
                target_grid_size=target_grid_size,
                mlp_depth=1,
                rms_norm=rms_norm,
            )

    def __call__(self, x, s=None, condition=None, *, key=None, inference=True):
        if self.proj_concat is not None and s is not None:
            x = self.act_fn(self.proj_concat(jnp.concatenate([x, s], axis=-1)))
        x = self.swin(x, condition, key=key, inference=inference)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinNDUnet(eqx.Module):
    """N-dimensional Swin U-Net on a single unbatched ``(C, *spatial)`` sample.

    ``build_down=False`` leaves out the patch embedding and the encoder stages (their grids
    and widths are still derived), ``build_middle=False`` the global-attention bottleneck and
    its upscale; owners of such a U-Net drive those parts themselves.
    ``decoder_rms_norm`` selects the norm of the decoder Swin layers and the bottleneck
    upscale, independently of ``rms_norm``. ``enc_cond_dim`` / ``dec_cond_dim`` set the
    condition width of the encoder / decoder stages (0: unconditioned), else ``n_cond`` sets both.
    """

    patch_embed: Optional[PatchEmbed]
    cond_embed: Optional[ContinuousConditionEmbed]
    down_blocks: list[SwinBlockDown]
    middle: Optional[BlockStack]
    middle_upscale: Optional[PatchExpand]
    up_blocks: list[SwinBlockUp]
    unpatch: PatchExpand

    base_resolution: tuple[int, ...] = eqx.field(static=True)
    patch_size: tuple[int, ...] = eqx.field(static=True)
    grid_sizes: tuple = eqx.field(static=True)
    down_dims: tuple = eqx.field(static=True)

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
        c_multiplier: int = 2,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        merging_depth: int = 2,
        unmerging_depth: int = 2,
        act_fn: Callable = gelu,
        use_checkpoint: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
        decoder_rms_norm: bool = False,
        up_use_skip: bool = True,
        enc_cond_dim: Optional[int] = None,
        dec_cond_dim: Optional[int] = None,
        cond_mode: str = "dit",
        legacy_double_shortcut: bool = False,
        n_cond: int = 0,
        cond_embed_dim: int = 128,
        build_down: bool = True,
        build_middle: bool = True,
        conv_patch: bool = False,
        unpatch_patch_skip: bool = False,
        key,
    ):
        patch_size = _as_seq(patch_size, space)
        window_size = _as_seq(window_size, space)
        depth = _as_seq(depth, num_layers)
        num_heads = _as_seq(num_heads, num_layers)
        # right-pad base resolution to a multiple of patch_size
        padded_base = [
            s + p for s, p in zip(base_resolution, pad_amounts(base_resolution, patch_size))
        ]
        # one key per stage: patch_embed, down blocks, middle, middle_upscale, up blocks, unpatch, cond_embed
        keys = jr.split(key, num_layers * 2 + 5)

        self.patch_embed = None
        if build_down:
            self.patch_embed = PatchEmbed(
                padded_base,
                patch_size,
                in_channels=in_channels,
                embed_dim=dim,
                key=keys[0],
                mlp_depth=merging_depth,
                mlp_ratio=merging_hidden_ratio,
                act_fn=act_fn,
            )
        # per-u-net conditioning embed: raw scalars to the cond_dim of every conditioned block
        self.cond_embed = None
        cond_dim = None
        if n_cond > 0:
            self.cond_embed = ContinuousConditionEmbed(cond_embed_dim, n_cond, key=keys[-1])
            cond_dim = self.cond_embed.cond_dim
        # per-path overrides of the u-net condition; 0 disables conditioning on that path
        down_cond = cond_dim if enc_cond_dim is None else (enc_cond_dim or None)
        up_cond = cond_dim if dec_cond_dim is None else (dec_cond_dim or None)
        layer_kw = dict(
            mlp_ratio=hidden_mlp_ratio,
            drop_path=drop_path,
            act_fn=act_fn,
            use_checkpoint=use_checkpoint,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            norm_affine=norm_affine,
            legacy_double_shortcut=legacy_double_shortcut,
            cond_dim=cond_dim,
            cond_mode=cond_mode,
        )

        grid_sizes = [tuple(s // p for s, p in zip(padded_base, _normalize_patch(patch_size)))]
        down_dims = [dim]
        self.down_blocks = []
        for i in range(num_layers):
            if build_down:
                blk = SwinBlockDown(
                    down_dims[i],
                    grid_sizes[i],
                    window_size,
                    num_heads[i],
                    depth[i],
                    key=keys[1 + i],
                    c_multiplier=c_multiplier,
                    rms_norm=rms_norm,
                    **{**layer_kw, "cond_dim": down_cond},
                )
                self.down_blocks.append(blk)
            down_dims.append(down_dims[i] * c_multiplier)
            grid_sizes.append(merge_grid(grid_sizes[i]))
        self.grid_sizes = tuple(grid_sizes)
        self.down_dims = tuple(down_dims)

        # global attention at the deepest grid: a swin layer whose window is the whole grid
        self.middle = self.middle_upscale = None
        if build_middle:
            self.middle = swin_layer(
                down_dims[-1],
                middle_depth,
                middle_num_heads,
                grid_sizes[-1],
                grid_sizes[-1],
                key=keys[num_layers + 1],
                rms_norm=rms_norm,
                **layer_kw,
            )
            self.middle_upscale = PatchExpand(
                down_dims[-1],
                grid_sizes[-1],
                key=keys[num_layers + 2],
                target_grid_size=grid_sizes[-2],
                c_multiplier=c_multiplier,
                mlp_depth=1,
                rms_norm=decoder_rms_norm,
                use_conv=conv_patch,
            )

        # up path; the final decoder block has no upsample
        up_dims = down_dims[::-1][1:]
        up_grid_sizes = grid_sizes[::-1][1:]
        self.up_blocks = []
        for i in range(num_layers):
            last = i == num_layers - 1
            self.up_blocks.append(
                SwinBlockUp(
                    up_dims[i],
                    up_grid_sizes[i],
                    window_size,
                    depth[::-1][i],
                    num_heads[::-1][i],
                    key=keys[num_layers + 3 + i],
                    target_grid_size=None if last else up_grid_sizes[i + 1],
                    c_multiplier=c_multiplier,
                    upsample=not last,
                    use_skip=up_use_skip,
                    rms_norm=rms_norm,
                    decoder_rms_norm=decoder_rms_norm,
                    **{**layer_kw, "cond_dim": up_cond},
                )
            )

        # unpatch: expand back to padded base resolution (norm=False)
        self.unpatch = PatchExpand(
            up_dims[-1],
            up_grid_sizes[-1],
            key=keys[2 * num_layers + 3],
            expand_by=tuple(p if p > 0 else 1 for p in patch_size),
            out_channels=out_channels,
            mlp_depth=unmerging_depth,
            mlp_ratio=unmerging_hidden_ratio,
            norm=False,
            use_conv=conv_patch,
            patch_skip=unpatch_patch_skip,
            cond_dim=up_cond,
        )
        self.base_resolution = tuple(base_resolution)
        self.patch_size = tuple(patch_size)

    def condition(self, cond):
        if self.cond_embed is None or cond is None:
            return None
        return self.cond_embed(cond)

    def patch_encode(self, x: jnp.ndarray) -> jnp.ndarray:
        # (C, *spatial) → (*spatial, C) → pad → patch_embed
        return self.patch_embed(pad_to_blocks(jnp.moveaxis(x, 0, -1), self.patch_size))

    def patch_decode(self, z: jnp.ndarray, condition=None) -> jnp.ndarray:
        x = unpad(self.unpatch(z, condition), self.base_resolution)
        return jnp.moveaxis(x, -1, 0)


class Swin5DUnet(SwinNDUnet):
    """5D wrapper with optional ``decouple_mu`` collapse.

    When ``decouple_mu=True`` the mu axis (index 1 after channels) is folded
    into the channel dimension and a learned ``vel_pe`` is added to provide
    velocity-space positional information.
    """

    decouple_mu: bool = eqx.field(static=True)
    decoupled_dim: int = eqx.field(static=True)
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
        full_in = in_channels
        decoupled_dim = 0
        if decouple_mu:
            # drop the mu axis (index 1) from the spatial, patch and window specs
            space = 4
            decoupled_dim = base_resolution[1]
            base_resolution = [base_resolution[0], *base_resolution[2:]]
            patch_size = [patch_size[0], *patch_size[2:]]
            window_size = [window_size[0], *window_size[2:]]
            in_channels *= decoupled_dim
            out_channels *= decoupled_dim
        k_unet, k_vel = jr.split(key)
        super().__init__(
            space=space,
            base_resolution=base_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            window_size=window_size,
            key=k_unet,
            **kwargs,
        )
        self.decouple_mu = decouple_mu
        self.decoupled_dim = decoupled_dim
        self.vel_pe = APE(full_in, (1, decoupled_dim, 1, 1, 1), key=k_vel) if decouple_mu else None

    def patch_encode(self, df: jnp.ndarray) -> jnp.ndarray:
        # (C, vp, mu, s, x, y) → channel last → vel_pe → mu folded into channels
        df = jnp.moveaxis(df, 0, -1)
        if self.decouple_mu:
            df = rearrange(self.vel_pe(df), "vp mu s x y c -> vp s x y (c mu)")
        return self.patch_embed(pad_to_blocks(df, self.patch_size))

    def patch_decode(self, z: jnp.ndarray, condition=None) -> jnp.ndarray:
        df = unpad(self.unpatch(z, condition), self.base_resolution)
        if self.decouple_mu:
            return rearrange(df, "vp s x y (c mu) -> c vp mu s x y", mu=self.decoupled_dim)
        return jnp.moveaxis(df, -1, 0)
