"""Swin5DAE — deterministic autoencoder built on Swin5DUnet.

The bottleneck inserts two extra global-attention stages (``middle_pre`` /
``middle_post``) around a channel projection that compresses the latent
dimension. Encoder and decoder can be conditioned separately on scalar
conditions (DiT modulation); the model takes the condition vector ordered by the
sorted union of both key sets.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.gk_unet import Swin5DUnet
from neugk_jax.models.patching import PatchExpand
from neugk_jax.models.swin import BlockStack
from neugk_jax.models.utils import LayerNorm, Linear, gelu, split_key
from neugk_jax.models.vit import vit_layer


class Swin5DAE(eqx.Module):
    """Wraps Swin5DUnet with a bottleneck projection."""

    backbone: Swin5DUnet
    enc_cond_embed: Optional[ContinuousConditionEmbed]
    dec_cond_embed: Optional[ContinuousConditionEmbed]
    middle_pre: BlockStack
    middle_post: BlockStack
    middle_downproj: Linear
    middle_upproj: Linear
    middle_upscale: PatchExpand
    pre_z_norm: Optional[LayerNorm]
    post_z_norm: Optional[LayerNorm]

    bottleneck_dim: int = eqx.field(static=True)
    bottleneck_grid_size: tuple[int, ...] = eqx.field(static=True)
    normalized_latent: bool = eqx.field(static=True)
    condition_keys: tuple[str, ...] = eqx.field(static=True)
    enc_indices: Optional[tuple[int, ...]] = eqx.field(static=True)
    dec_indices: Optional[tuple[int, ...]] = eqx.field(static=True)

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
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = True,
        legacy_double_shortcut: bool = False,
        decoder_rms_norm: bool = False,
        use_checkpoint: bool = False,
        encoder_conditioning: Sequence[str] = (),
        decoder_conditioning: Sequence[str] = (),
        cond_embed_dim: int = 32,
        key,
    ):
        kb, k1, k2, k3, k4, k5 = jr.split(key, 6)
        enc_keys = tuple(sorted(encoder_conditioning or ()))
        dec_keys = tuple(sorted(decoder_conditioning or ()))
        union = tuple(sorted(set(enc_keys) | set(dec_keys)))
        self.condition_keys = union
        self.enc_indices = tuple(union.index(k) for k in enc_keys) if enc_keys else None
        self.dec_indices = tuple(union.index(k) for k in dec_keys) if dec_keys else None

        def cond_embed(keys, i):
            if not keys:
                return None
            return ContinuousConditionEmbed(cond_embed_dim, len(keys), key=jr.fold_in(key, i))

        self.enc_cond_embed = cond_embed(enc_keys, 7)
        self.dec_cond_embed = cond_embed(dec_keys, 8)
        enc_cdim = self.enc_cond_embed.cond_dim if enc_keys else 0
        dec_cdim = self.dec_cond_embed.cond_dim if dec_keys else 0
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
            c_multiplier=c_multiplier,
            drop_path=drop_path,
            hidden_mlp_ratio=hidden_mlp_ratio,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            merging_depth=merging_depth,
            unmerging_depth=unmerging_depth,
            act_fn=act_fn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            norm_affine=norm_affine,
            legacy_double_shortcut=legacy_double_shortcut,
            use_checkpoint=use_checkpoint,
            rms_norm=rms_norm,
            decoder_rms_norm=decoder_rms_norm,
            enc_cond_dim=enc_cdim,
            dec_cond_dim=dec_cdim,
            # ae has no encoder→decoder skips and its own bottleneck
            up_use_skip=False,
            build_middle=False,
            key=kb,
        )

        # bottleneck dims derived from the deepest encoder grid
        mid_dim = self.backbone.down_dims[-1]
        mid_grid = self.backbone.grid_sizes[-1]
        bd = bottleneck_dim or mid_dim

        self.bottleneck_dim = bd
        self.bottleneck_grid_size = mid_grid

        # bottleneck vit blocks use an affine norm regardless of the encoder setting
        vit_kw = dict(
            mlp_ratio=hidden_mlp_ratio,
            drop_path=drop_path,
            act_fn=act_fn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            gated_attention=gated_attention,
            norm_affine=True,
            rms_norm=rms_norm,
        )
        self.middle_pre = vit_layer(
            mid_dim, bottleneck_depth, bottleneck_num_heads, key=k1, cond_dim=enc_cdim, **vit_kw
        )
        self.middle_post = vit_layer(
            mid_dim, bottleneck_depth, bottleneck_num_heads, key=k2, cond_dim=dec_cdim, **vit_kw
        )
        self.middle_downproj = Linear(mid_dim, bd, key=k3)
        self.middle_upproj = Linear(bd, mid_dim, key=k4)
        self.middle_upscale = PatchExpand(
            mid_dim,
            mid_grid,
            key=k5,
            target_grid_size=self.backbone.grid_sizes[-2],
            c_multiplier=c_multiplier,
            mlp_depth=1,
            rms_norm=decoder_rms_norm,
        )

        if normalized_latent:
            self.pre_z_norm = LayerNorm(bd)
            self.post_z_norm = LayerNorm(bd)
        else:
            self.pre_z_norm = None
            self.post_z_norm = None
        self.normalized_latent = normalized_latent

    @staticmethod
    def _embed(embed, condition, indices):
        if embed is None:
            return None
        if condition is None:
            raise ValueError("this autoencoder is conditioned; pass `condition`")
        return embed(condition[jnp.asarray(indices)])

    def bottleneck(self, z: jnp.ndarray, *, inference: bool = True) -> tuple[jnp.ndarray, dict]:
        return z, {}

    def encode(
        self, df: jnp.ndarray, condition=None, *, key=None, inference: bool = True
    ) -> jnp.ndarray:
        cond = self._embed(self.enc_cond_embed, condition, self.enc_indices)
        keys = split_key(key, len(self.backbone.down_blocks) + 1)
        z = self.backbone.patch_encode(df)
        for blk, k in zip(self.backbone.down_blocks, keys):
            z = blk(z, cond, return_skip=False, key=k, inference=inference)
        z = self.middle_downproj(self.middle_pre(z, cond, key=keys[-1], inference=inference))
        return self.pre_z_norm(z) if self.normalized_latent else z

    def decode(self, z: jnp.ndarray, condition=None, *, key=None, inference: bool = True):
        cond = self._embed(self.dec_cond_embed, condition, self.dec_indices)
        keys = split_key(key, len(self.backbone.up_blocks) + 1)
        if self.normalized_latent:
            z = self.post_z_norm(z)
        z = self.middle_post(self.middle_upproj(z), cond, key=keys[0], inference=inference)
        z = self.middle_upscale(z)
        # no skip connections in the ae decoder
        for blk, k in zip(self.backbone.up_blocks, keys[1:]):
            z = blk(z, None, cond, key=k, inference=inference)
        return {"df": self.backbone.patch_decode(z, cond)}

    def __call__(
        self,
        df: jnp.ndarray,
        condition=None,
        return_latent: bool = False,
        *,
        key=None,
        inference: bool = True,
    ):
        k_enc, k_dec = split_key(key, 2)
        z = self.encode(df, condition, key=k_enc, inference=inference)
        z, extra = self.bottleneck(z, inference=inference)
        out = self.decode(z, condition, key=k_dec, inference=inference)
        out.update(extra)
        if return_latent:
            out["latent"] = z
        return out
