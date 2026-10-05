"""Latent-space DiT (Diffusion Transformer).

``encoder`` (single Linear, no bias, followed by ``act_fn``) projects
per-token latents into the transformer dim, ``ape`` adds learnable
absolute position embeddings, ``backbone`` runs the DiT-modulated
transformer stack with time + scalar condition embeddings concatenated,
and ``decoder`` (single Linear, no bias) projects back to ``z_dim``.

Patching (``patch_embed``/``unpatch``) is omitted — the production
``DIFF_FLOW`` config uses ``patch_size: null``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.embeddings import APE, ContinuousConditionEmbed
from neugk_jax.models.swin import BlockStack
from neugk_jax.models.utils import Linear, gelu
from neugk_jax.models.vit import vit_layer


class DiT(eqx.Module):
    """Latent DiT.

    Forward signature (per sample):
        ``__call__(x: (*grid, z_dim), tstep: scalar, condition: (n_cond,))``
        → ``(*grid, z_dim)``
    """

    encoder: list  # single-element list holding the projection Linear
    ape: APE
    backbone: BlockStack
    decoder: Linear
    time_embed: ContinuousConditionEmbed
    cond_embed: Optional[ContinuousConditionEmbed]
    act: Callable = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        z_dim: int,
        dim: int,
        grid_size: Sequence[int],
        depth: int,
        num_heads: int,
        n_cond: int,
        key,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
    ):
        keys = jr.split(key, 6)
        self.time_embed = ContinuousConditionEmbed(32, 1, key=keys[0])
        cdim = self.time_embed.cond_dim
        self.cond_embed = None
        if n_cond > 0:
            self.cond_embed = ContinuousConditionEmbed(32, n_cond, key=keys[1])
            cdim += self.cond_embed.cond_dim
        self.cond_dim = cdim
        self.encoder = [Linear(z_dim, dim, key=keys[2], use_bias=False)]
        self.ape = APE(dim, grid_size, key=keys[3])
        self.backbone = vit_layer(
            dim,
            depth,
            num_heads,
            key=keys[4],
            cond_dim=cdim,
            cond_mode="dit",
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            act_fn=act_fn,
            norm_affine=True,
        )
        self.decoder = Linear(dim, z_dim, key=keys[5], use_bias=False)
        self.act = act_fn
        self.latent_shape = (*tuple(grid_size), z_dim)

    def __call__(
        self,
        x: jnp.ndarray,
        tstep: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        # x: (*grid, z_dim), tstep: scalar, condition: (n_cond,) or None
        cond = self.time_embed(jnp.asarray(tstep).reshape((1,)))
        if condition is not None and self.cond_embed is not None:
            cond = jnp.concatenate([cond, self.cond_embed(condition)], axis=-1)
        h = self.ape(self.act(self.encoder[0](x)))
        h = self.backbone(h, cond, key=key, inference=inference)
        return self.decoder(h)
