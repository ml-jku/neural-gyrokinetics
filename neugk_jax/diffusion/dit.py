"""Latent-space DiT (Diffusion Transformer).

Mirrors ``neugk/diffusion/models/dit.py``: ``encoder`` (single Linear, no
bias, followed by ``act_fn``) projects per-token latents into the
transformer dim, ``ape`` adds learnable absolute position embeddings,
``backbone`` runs the DiT-modulated transformer stack with time + scalar
condition embeddings concatenated, and ``decoder`` (single Linear, no
bias) projects back to ``z_dim``.

Patching (``patch_embed``/``unpatch``) is omitted — the production
``DIFF_FLOW`` config uses ``patch_size: null``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.embeddings import APE, ContinuousConditionEmbed
from neugk_jax.models.utils import Linear, gelu
from neugk_jax.models.vit import DiTLayer


class DiT(eqx.Module):
    """Latent DiT.

    Forward signature (per sample):
        ``__call__(x: (*grid, z_dim), tstep: scalar, condition: (n_cond,))``
        → ``(*grid, z_dim)``
    """

    encoder: list  # [Linear] — mirrors torch Sequential(Linear, act)
    ape: APE
    backbone: DiTLayer
    decoder: Linear
    time_embed: ContinuousConditionEmbed
    cond_embed: Optional[ContinuousConditionEmbed]
    act: Callable = eqx.field(static=True)
    z_dim: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    grid_size: tuple[int, ...] = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        space: int,
        z_dim: int,
        dim: int,
        grid_size: Sequence[int],
        depth: int,
        num_heads: int,
        n_cond: int,
        key,
        time_embed_dim: int = 32,
        cond_embed_dim: int = 32,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
    ):
        keys = jr.split(key, 6)

        self.time_embed = ContinuousConditionEmbed(
            time_embed_dim, 1, key=keys[0],
        )
        cdim = self.time_embed.cond_dim
        if n_cond > 0:
            self.cond_embed = ContinuousConditionEmbed(
                cond_embed_dim, n_cond, key=keys[1],
            )
            cdim += self.cond_embed.cond_dim
        else:
            self.cond_embed = None
        self.cond_dim = cdim

        self.encoder = [Linear(z_dim, dim, key=keys[2], use_bias=False)]
        self.ape = APE(dim, grid_size, init="normal", learnable=True, key=keys[3])
        self.backbone = DiTLayer(
            space=space,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            grid_size=grid_size,
            key=keys[4],
            cond_dim=cdim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            act_fn=act_fn,
        )
        self.decoder = Linear(dim, z_dim, key=keys[5], use_bias=False)

        self.act = act_fn
        self.z_dim = z_dim
        self.dim = dim
        self.grid_size = tuple(grid_size)
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
        t_emb = self.time_embed(jnp.asarray(tstep).reshape((1,)))
        if condition is not None and self.cond_embed is not None:
            c_emb = self.cond_embed(condition)
            cond = jnp.concatenate([t_emb, c_emb], axis=-1)
        else:
            cond = t_emb

        h = self.act(self.encoder[0](x))
        h = self.ape(h)
        h = self.backbone(h, cond, key=key, inference=inference)
        return self.decoder(h)
