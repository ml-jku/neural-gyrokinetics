"""Swin5DAE — deterministic autoencoder built on Swin5DUnet.

The bottleneck inserts two extra global-attention stages (``middle_pre`` /
``middle_post``) around a channel projection that compresses the latent
dimension.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.gk_unet import Swin5DUnet
from neugk_jax.models.utils import LayerNorm, Linear, gelu
from neugk_jax.models.patching import PatchExpand
from neugk_jax.models.vit import ViTLayer


class Swin5DAE(eqx.Module):
    """Wraps Swin5DUnet with a bottleneck projection."""

    backbone: Swin5DUnet
    middle_pre: ViTLayer
    middle_post: ViTLayer
    middle_downproj: Linear
    middle_upproj: Linear
    middle_upscale: PatchExpand
    pre_z_norm: Optional[LayerNorm]
    post_z_norm: Optional[LayerNorm]

    bottleneck_dim: int = eqx.field(static=True)
    middle_dim: int = eqx.field(static=True)
    bottleneck_grid_size: tuple[int, ...] = eqx.field(static=True)
    normalized_latent: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: int = 5,
        decouple_mu: bool = False,
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
        bottleneck_dim: Optional[int] = None,
        bottleneck_depth: int = 2,
        bottleneck_num_heads: int = 2,
        normalized_latent: bool = False,
        c_multiplier: int = 2,
        drop_path: float = 0.1,
        hidden_mlp_ratio: float = 2.0,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        merging_depth: int = 2,
        unmerging_depth: int = 2,
        use_abs_pe: bool = False,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        key,
    ):
        kb, k1, k2, k3, k4, k5 = jr.split(key, 6)
        self.backbone = Swin5DUnet(
            space=space,
            decouple_mu=decouple_mu,
            dim=dim,
            base_resolution=base_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            num_layers=num_layers,
            middle_depth=middle_depth,
            middle_num_heads=middle_num_heads,
            c_multiplier=c_multiplier,
            drop_path=drop_path,
            hidden_mlp_ratio=hidden_mlp_ratio,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            merging_depth=merging_depth,
            unmerging_depth=unmerging_depth,
            use_abs_pe=use_abs_pe,
            act_fn=act_fn,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            use_rpb=use_rpb, gated_attention=gated_attention,
            norm_affine=norm_affine,
            rms_norm=True,  # upstream config uses RMSNorm
            # AE has no encoder→decoder skips
            up_use_skip=False,
            key=kb,
        )

        # bottleneck dims derived from the deepest encoder grid
        mid_dim = self.backbone.down_dims[-1]
        mid_grid = self.backbone.grid_sizes[-1]
        bd = bottleneck_dim or mid_dim

        self.middle_dim = mid_dim
        self.bottleneck_dim = bd
        self.bottleneck_grid_size = mid_grid

        # bottleneck ViT blocks use RMSNorm(elementwise_affine=True) regardless of encoder setting
        vit_kwargs = dict(
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            gated_attention=gated_attention,
            norm_affine=True,
            rms_norm=True,
        )
        self.middle_pre = ViTLayer(
            space=self.backbone.space, dim=mid_dim, depth=bottleneck_depth,
            num_heads=bottleneck_num_heads, grid_size=mid_grid,
            key=k1, mlp_ratio=hidden_mlp_ratio, drop_path=drop_path,
            act_fn=act_fn, **vit_kwargs,
        )
        self.middle_post = ViTLayer(
            space=self.backbone.space, dim=mid_dim, depth=bottleneck_depth,
            num_heads=bottleneck_num_heads, grid_size=mid_grid,
            key=k2, mlp_ratio=hidden_mlp_ratio, drop_path=drop_path,
            act_fn=act_fn, **vit_kwargs,
        )
        self.middle_downproj = Linear(mid_dim, bd, key=k3)
        self.middle_upproj = Linear(bd, mid_dim, key=k4)
        # AE middle_upscale uses LayerNorm (with weight + bias), matching upstream default
        self.middle_upscale = PatchExpand(
            mid_dim, mid_grid, key=k5,
            target_grid_size=self.backbone.grid_sizes[-2],
            c_multiplier=c_multiplier, mlp_depth=1, rms_norm=False,
        )

        if normalized_latent:
            self.pre_z_norm = LayerNorm(bd)
            self.post_z_norm = LayerNorm(bd)
        else:
            self.pre_z_norm = None
            self.post_z_norm = None
        self.normalized_latent = normalized_latent


    def encode(self, df: jnp.ndarray):
        z, pad_axes = self.backbone.patch_encode(df)
        for blk in self.backbone.down_blocks:
            z = blk(z, return_skip=False)
        z = self.middle_pre(z)
        z = self.middle_downproj(z)
        if self.normalized_latent:
            z = self.pre_z_norm(z)
        return z, pad_axes


    def decode(self, z: jnp.ndarray, pad_axes=None):
        if pad_axes is None:
            # reconstruct pad_axes from the base resolution
            from neugk_jax.models.patching import pad_to_blocks
            dummy = jnp.zeros((self.backbone.original_in_channels, *self.backbone.full_resolution))
            _, pad_axes = self.backbone.patch_encode(dummy)
        if self.normalized_latent:
            z = self.post_z_norm(z)
        z = self.middle_upproj(z)
        z = self.middle_post(z)
        z = self.middle_upscale(z)
        # no skip connections in AE decoder
        for blk in self.backbone.up_blocks:
            z = blk(z, s=None)
        df = self.backbone.patch_decode(z, pad_axes)
        return {"df": df}


    def __call__(self, df: jnp.ndarray, return_latent: bool = False):
        z, pad_axes = self.encode(df)
        out = self.decode(z, pad_axes)
        if return_latent:
            out["latent"] = z
        return out
