"""GyroSwin multitask model — 5D df to 5D df + 3D phi with cross-attention mixing.

Composition:

* ``df_unet``: ``Swin5DUnet`` — full 5D Swin U-Net on the distribution function.
* ``phi_unet``: ``SwinNDUnet`` (space=3) without an encoder; its skips come from
  ``vspace_attn_down`` reducing the df features.
* ``vspace_attn_down`` / ``vspace_attn_middle`` / ``vspace_attn_patch_skip``:
  ``QueryPool`` velocity-space reductions of the 5D df latents to the 3D phi shape.
* ``df_mix_middle`` / ``phi_mix_middle``: bottleneck cross-attention.
* ``df_mix_up`` / ``phi_mix_up``: up-path cross-attention at each scale.
* ``flux_head``: ``FluxDecoder`` scalar flux (``flux`` or ``fluxavg``) from the
  per-scale (phi, df) latents, optionally FiLM-conditioned.

Conditioning (DiT or FiLM modulation of the Swin blocks) goes through the
per-u-net condition embeds; configs without conditioning still work.
"""

from __future__ import annotations

from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.gyroswin.models.x_layers import FluxDecoder, MixingBlock, QueryPool, velocity_pool
from neugk_jax.models.gk_unet import Swin5DUnet, SwinNDUnet
from neugk_jax.models.utils import split_key


class GyroSwinMultitask(eqx.Module):
    """5D df → 5D df + 3D phi (+ scalar flux) prediction with cross-attention mixing.

    ``attn_drop`` is the attention-probability dropout of every mixing block and
    velocity-space reduction; ``flux_drop`` is the flux head's projection/MLP
    dropout. ``flux_conditioning`` FiLM-conditions the flux head on the raw
    conditioning scalars.
    """

    df_unet: Swin5DUnet
    phi_unet: SwinNDUnet
    vspace_attn_down: list
    vspace_attn_middle: QueryPool
    vspace_attn_patch_skip: Optional[QueryPool]
    df_mix_middle: MixingBlock
    phi_mix_middle: MixingBlock
    df_mix_up: list
    phi_mix_up: list
    df_mix_unpatch: MixingBlock
    phi_mix_unpatch: MixingBlock
    flux_head: Optional[FluxDecoder]
    flux_key: Optional[str] = eqx.field(static=True)
    use_phi: bool = eqx.field(static=True)
    patch_skip: bool = eqx.field(static=True)
    detach_phi_cross_latents: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        dim: int,
        df_base_resolution: Sequence[int],
        df_patch_size: Sequence[int],
        df_window_size: Sequence[int],
        depth: int,
        num_heads: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 4,
        c_multiplier: int = 2,
        merging_hidden_ratio: float = 4.0,
        unmerging_hidden_ratio: float = 8.0,
        decouple_mu: bool = True,
        patch_skip: bool = True,
        use_rpb: bool = True,
        qk_norm: bool = True,
        gated_attention: bool = True,
        use_phi: bool = True,
        flux_key: Optional[str] = None,
        n_cond: int = 0,
        cond_embed_dim: int = 128,
        cond_mode: str = "film",
        flux_num_heads: int = 4,
        flux_depth: int = 1,
        flux_conditioning: bool = False,
        flux_drop: float = 0.1,
        attn_drop: float = 0.1,
        detach_flux_latents: bool = False,
        detach_phi_cross_latents: bool = False,
        rms_norm: bool = False,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
        legacy_double_shortcut: bool = False,
        key,
    ):
        self.patch_skip = patch_skip
        self.use_phi = use_phi
        self.flux_key = flux_key
        self.detach_phi_cross_latents = detach_phi_cross_latents
        keys = jr.split(key, 12)
        unet_kw = dict(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_mlp_ratio=8.0,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            qk_norm=qk_norm,
            use_rpb=use_rpb,
            gated_attention=gated_attention,
            use_checkpoint=use_checkpoint,
            n_cond=n_cond,
            cond_embed_dim=cond_embed_dim,
            cond_mode=cond_mode,
            unpatch_patch_skip=patch_skip,
            rms_norm=rms_norm,
            legacy_double_shortcut=legacy_double_shortcut,
            drop_path=drop_path,
        )
        self.df_unet = Swin5DUnet(
            space=5,
            decouple_mu=decouple_mu,
            base_resolution=list(df_base_resolution),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=list(df_patch_size),
            window_size=list(df_window_size),
            c_multiplier=c_multiplier,
            key=keys[0],
            **unet_kw,
        )
        # phi unet without an encoder; its skips are the df reductions
        self.phi_unet = SwinNDUnet(
            space=3,
            base_resolution=list(df_base_resolution[2:]),
            in_channels=1,
            out_channels=1,
            patch_size=list(df_patch_size[2:]),
            window_size=list(df_window_size[2:]),
            c_multiplier=2,
            conv_patch=True,
            build_down=False,
            key=keys[1],
            **unet_kw,
        )

        df_dims, phi_dims = list(self.df_unet.down_dims), list(self.phi_unet.down_dims)
        # one reduction per df down block; out_dim matches the corresponding phi up block
        phi_skip_dims = phi_dims[:-1]
        pool_kw = dict(num_heads=8, attn_drop=attn_drop)
        self.vspace_attn_down = [
            QueryPool(d, phi_skip_dims[i] if i < len(phi_skip_dims) else d, key=k, **pool_kw)
            for i, (d, k) in enumerate(zip(df_dims[:-1], jr.split(keys[2], len(df_dims) - 1)))
        ]
        bottleneck_dim = df_dims[-1]
        self.vspace_attn_middle = QueryPool(bottleneck_dim, bottleneck_dim, key=keys[3], **pool_kw)
        self.vspace_attn_patch_skip = (
            QueryPool(dim, dim, key=keys[4], **pool_kw) if patch_skip else None
        )

        mix_kw = dict(num_heads=8, attn_drop=attn_drop)
        self.df_mix_middle = MixingBlock(bottleneck_dim, bottleneck_dim, key=keys[5], **mix_kw)
        self.phi_mix_middle = MixingBlock(bottleneck_dim, bottleneck_dim, key=keys[6], **mix_kw)
        # up-path mixing: dims match the inputs to each SwinBlockUp (post middle_upscale)
        df_up, phi_up = df_dims[::-1][1:], phi_dims[::-1][1:]
        phi_up = [phi_up[i] if i < len(phi_up) else d for i, d in enumerate(df_up)]
        self.df_mix_up = [
            MixingBlock(d, p, key=k, **mix_kw)
            for d, p, k in zip(df_up, phi_up, jr.split(keys[7], len(df_up)))
        ]
        self.phi_mix_up = [
            MixingBlock(p, d, key=k, **mix_kw)
            for d, p, k in zip(df_up, phi_up, jr.split(keys[8], len(df_up)))
        ]
        # patch-space mixing runs after the patch-skip concat, so the dim doubles with patch_skip
        unpatch_dim = dim * (2 if patch_skip else 1)
        self.df_mix_unpatch = MixingBlock(unpatch_dim, unpatch_dim, key=keys[9], **mix_kw)
        self.phi_mix_unpatch = MixingBlock(unpatch_dim, unpatch_dim, key=keys[10], **mix_kw)

        # flux head stages run deepest first (phi=query, df=kv)
        self.flux_head = None
        if flux_key is not None:
            self.flux_head = FluxDecoder(
                left_dims=phi_dims[::-1],
                right_dims=df_dims[::-1],
                num_heads=flux_num_heads,
                depth=flux_depth,
                key=keys[11],
                attn_drop=attn_drop,
                drop=flux_drop,
                detach_latents=detach_flux_latents,
                n_cond=n_cond if flux_conditioning else 0,
                cond_embed_dim=cond_embed_dim,
            )

    def _phi_for_df(self, zphi):
        return jax.lax.stop_gradient(zphi) if self.detach_phi_cross_latents else zphi

    def __call__(
        self,
        df: jnp.ndarray,
        cond: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> dict:
        """Forward: df → ``{"df", "phi"?, flux_key?}``.

        df: ``(C, vp, mu, s, x, y)``; cond: ``(n_cond,)`` raw scalars. ``key``
        drives drop-path and dropout when ``inference=False``.
        """
        kw = dict(inference=inference)
        n_down, n_up = len(self.df_unet.down_blocks), len(self.df_unet.up_blocks)
        k_skip, k_down, k_mid, k_up, k_out = split_key(key, 5)
        c_df = self.df_unet.condition(cond)
        c_phi = self.phi_unet.condition(cond)

        zdf = self.df_unet.patch_encode(df)
        # patch-skip residuals: df0 (full patch grid) and its velocity-reduced phi0
        df0, phi0 = zdf, None
        if self.vspace_attn_patch_skip is not None:
            phi0 = velocity_pool(self.vspace_attn_patch_skip, df0, key=k_skip, **kw)
        # down path: df skips feed the df up blocks, their velocity reductions the phi up blocks
        df_skips, phi_skips = [], []
        for i, (blk, k) in enumerate(zip(self.df_unet.down_blocks, split_key(k_down, n_down))):
            k_blk, k_pool = split_key(k, 2)
            zdf, sk = blk(zdf, c_df, key=k_blk, return_skip=True, **kw)
            df_skips.append(sk)
            if self.use_phi and i < len(self.vspace_attn_down):
                phi_skips.append(velocity_pool(self.vspace_attn_down[i], sk, key=k_pool, **kw))
        # bottleneck: vspace-reduce df → phi, parallel cross-mix, then the middle swin layers
        k_pool, k_dfmix, k_phimix, k_dfmid, k_phimid, k_flux = split_key(k_mid, 6)
        zphi = velocity_pool(self.vspace_attn_middle, zdf, key=k_pool, **kw)
        zdf, zphi = (
            self.df_mix_middle(zdf, self._phi_for_df(zphi), key=k_dfmix, **kw),
            self.phi_mix_middle(zphi, zdf, key=k_phimix, **kw),
        )
        zdf = self.df_unet.middle(zdf, c_df, key=k_dfmid, **kw)
        zphi = self.phi_unet.middle(zphi, c_phi, key=k_phimid, **kw)
        flux_lats = []
        if self.flux_head is not None:
            flux_lats.append(self.flux_head.mix(0, zphi, zdf, cond, key=k_flux, **kw))
        zdf = self.df_unet.middle_upscale(zdf)
        zphi = self.phi_unet.middle_upscale(zphi)
        # up path: df mixes first, then phi mixes against the updated df
        blocks = zip(self.df_unet.up_blocks, self.phi_unet.up_blocks, split_key(k_up, n_up))
        for i, (df_blk, phi_blk, k) in enumerate(blocks):
            k_dfmix, k_phimix, k_dfblk, k_phiblk, k_flux = split_key(k, 5)
            zdf = self.df_mix_up[i](zdf, self._phi_for_df(zphi), key=k_dfmix, **kw)
            zphi = self.phi_mix_up[i](zphi, zdf, key=k_phimix, **kw)
            zdf = df_blk(zdf, df_skips[-(i + 1)], c_df, key=k_dfblk, **kw)
            phi_sk = phi_skips[i] if (self.use_phi and i < len(phi_skips)) else None
            zphi = phi_blk(zphi, phi_sk, c_phi, key=k_phiblk, **kw)
            if self.flux_head is not None:
                flux_lats.append(self.flux_head.mix(i + 1, zphi, zdf, cond, key=k_flux, **kw))
        if self.patch_skip:
            zdf = jnp.concatenate([zdf, df0], axis=-1)
            zphi = jnp.concatenate([zphi, phi0], axis=-1)
        k_dfmix, k_phimix, k_flux = split_key(k_out, 3)
        zdf = self.df_mix_unpatch(zdf, self._phi_for_df(zphi), key=k_dfmix, **kw)
        zphi = self.phi_mix_unpatch(zphi, zdf, key=k_phimix, **kw)
        out = {"df": self.df_unet.patch_decode(zdf, condition=c_df)}
        if self.use_phi:
            # phi_unet output (1, s, x, y) → (x, s, y), the dataset's phi layout
            phi = self.phi_unet.patch_decode(zphi, condition=c_phi)
            out["phi"] = jnp.transpose(jnp.squeeze(phi, axis=0), (1, 0, 2))
        if self.flux_head is not None:
            out[self.flux_key] = self.flux_head(flux_lats, key=k_flux, **kw)
        return out
