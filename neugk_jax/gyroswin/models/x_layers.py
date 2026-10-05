"""Cross-attention layers used to mix the df and phi U-Net latents.

``MixingBlock`` is a single cross-attention + MLP block; ``QueryPool`` pools groups of
tokens with a learned query token (the velocity-space integral of the df latents).
``FluxDecoder`` is the scalar flux head, optionally FiLM-conditioned on the raw
conditioning scalars.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.models.attention import MultiHeadCrossAttention, einsum_attention
from neugk_jax.models.embeddings import ContinuousConditionEmbed
from neugk_jax.models.swin import Film, _DropPath, run_blocks
from neugk_jax.models.utils import MLP, LayerNorm, Linear, gelu, split_key


class MixingBlock(eqx.Module):
    """Cross-attention + MLP. ``left`` queries kv from ``right``; output dim = left_dim.

    ``attn_drop`` drops attention probabilities; ``drop`` is the output-projection
    and MLP dropout.
    """

    norm1: LayerNorm
    attn: MultiHeadCrossAttention
    drop_path: _DropPath
    norm2: LayerNorm
    mlp: MLP

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        num_heads: int,
        *,
        key,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        act_fn=gelu,
    ):
        k1, k2 = jr.split(key, 2)
        self.norm1 = LayerNorm(left_dim, elementwise_affine=True)
        self.attn = MultiHeadCrossAttention(
            q_dim=left_dim,
            kv_dim=right_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            key=k1,
        )
        self.drop_path = _DropPath(drop_path)
        self.norm2 = LayerNorm(left_dim, elementwise_affine=True)
        self.mlp = MLP(
            [left_dim, int(left_dim * mlp_ratio), left_dim], act_fn=act_fn, drop=drop, key=k2
        )

    def __call__(
        self,
        left: jnp.ndarray,
        right: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        right = right if right is not None else left
        l_shape = left.shape
        l_tok = left.reshape(-1, l_shape[-1])
        r_tok = right.reshape(-1, right.shape[-1])
        k_attn, k_dp1, k_mlp, k_dp2 = split_key(key, 4)
        # post-norm on the attn output, pre-norm on the mlp branch
        h = self.norm1(self.attn(l_tok, r_tok, key=k_attn, inference=inference))
        x = l_tok + self.drop_path(h, key=k_dp1, inference=inference)
        h = self.mlp(self.norm2(x), key=k_mlp, inference=inference)
        x = x + self.drop_path(h, key=k_dp2, inference=inference)
        return x.reshape(l_shape)


class QueryPool(eqx.Module):
    """Learned-query attention pooling ``(groups, tokens, dim) -> (groups, out_dim)``.

    The query token ``integral_token`` (a buffer) attends over the tokens of each group.
    """

    kv: Linear
    proj: Linear
    integral_token: jax.Array
    buffer_fields = ("integral_token",)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        *,
        key,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        kkv, kp, ktoken = jr.split(key, 3)
        self.kv = Linear(dim, 2 * dim, key=kkv, use_bias=qkv_bias)
        self.proj = Linear(dim, out_dim, key=kp, use_bias=True)
        self.integral_token = 1e-2 * jr.normal(ktoken, (1, 1, dim))

    def __call__(self, x: jnp.ndarray, *, key=None, inference: bool = True) -> jnp.ndarray:
        g, n, dim = x.shape
        kv = self.kv(x).reshape(g, n, 2, self.num_heads, self.head_dim)
        q = jnp.broadcast_to(
            self.integral_token.reshape(1, 1, self.num_heads, self.head_dim),
            (g, 1, self.num_heads, self.head_dim),
        )
        scale = self.head_dim**-0.5
        out = einsum_attention(
            q, kv[:, :, 0], kv[:, :, 1], scale, None, self.attn_drop, key, inference
        )
        return self.proj(out.reshape(g, dim))


def velocity_pool(pool: QueryPool, df: jnp.ndarray, *, key=None, inference: bool = True):
    pattern = "vp s x y c -> (s x y) vp c" if df.ndim == 5 else "vp mu s x y c -> (s x y) (vp mu) c"
    out = pool(rearrange(df, pattern), key=key, inference=inference)
    return out.reshape(*df.shape[-4:-1], -1)


class LatentMixingTransformer(eqx.Module):
    """A stack of ``depth`` cross-attention ``MixingBlock``s (one FluxDecoder stage).

    With ``cond_dim`` set, each block input is FiLM-modulated by a condition
    embedded from the raw scalars by this stage's own ``cond_embed``.
    """

    blocks: list
    cond_embed: Optional[ContinuousConditionEmbed]
    conditioning: Optional[list]

    def __init__(
        self,
        left_dim: int,
        right_dim: int,
        num_heads: int,
        depth: int,
        *,
        key,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        n_cond: int = 0,
        cond_embed_dim: int = 128,
    ):
        kb, kc, kf = jr.split(key, 3)
        self.blocks = [
            MixingBlock(
                left_dim,
                right_dim,
                num_heads,
                key=k,
                mlp_ratio=2.0,
                qkv_bias=True,
                attn_drop=attn_drop,
                drop=drop,
            )
            for k in jr.split(kb, depth)
        ]
        if n_cond > 0:
            self.cond_embed = ContinuousConditionEmbed(cond_embed_dim, n_cond, key=kc)
            self.conditioning = [
                Film(self.cond_embed.cond_dim, left_dim, key=k) for k in jr.split(kf, depth)
            ]
        else:
            self.cond_embed = None
            self.conditioning = None

    def __call__(
        self, left: jnp.ndarray, right: jnp.ndarray, cond=None, *, key=None, inference: bool = True
    ) -> jnp.ndarray:
        c = self.cond_embed(cond) if self.cond_embed is not None else None
        return run_blocks(
            self.blocks, self.conditioning, left, c, right, key=key, inference=inference
        )


class FluxDecoder(eqx.Module):
    """Predict a scalar flux from the per-scale (phi, df) latents.

    One ``LatentMixingTransformer`` stage per scale: stage ``i`` cross-attends the
    phi latent (query) to the df latent (kv), max-pools over space to a vector of
    ``left_dims[i]``, and the per-scale vectors are concatenated and fed to ``flux_mlp``
    (sum -> half -> 1). ``n_cond > 0`` FiLM-conditions every stage on the raw
    conditioning scalars.
    """

    blocks: list
    flux_mlp: MLP
    detach_latents: bool = eqx.field(static=True)
    use_cond: bool = eqx.field(static=True)

    def __init__(
        self,
        left_dims,
        right_dims,
        num_heads: int,
        depth: int,
        *,
        key,
        attn_drop: float = 0.1,
        drop: float = 0.0,
        detach_latents: bool = False,
        n_cond: int = 0,
        cond_embed_dim: int = 128,
    ):
        kb, km = jr.split(key)
        self.blocks = [
            LatentMixingTransformer(
                left_dims[i],
                right_dims[i],
                num_heads,
                depth,
                key=k,
                attn_drop=attn_drop,
                drop=drop,
                n_cond=n_cond,
                cond_embed_dim=cond_embed_dim,
            )
            for i, k in enumerate(jr.split(kb, len(left_dims)))
        ]
        flux_latent = int(sum(left_dims))
        self.flux_mlp = MLP([flux_latent, flux_latent // 2, 1], act_fn=gelu, drop=drop, key=km)
        self.detach_latents = detach_latents
        self.use_cond = n_cond > 0

    def mix(
        self,
        i: int,
        left: jnp.ndarray,
        right: jnp.ndarray,
        cond=None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        if self.detach_latents:
            left, right = jax.lax.stop_gradient(left), jax.lax.stop_gradient(right)
        x = self.blocks[i](
            left, right, cond if self.use_cond else None, key=key, inference=inference
        )
        return jnp.max(x.reshape(-1, x.shape[-1]), axis=0)

    def __call__(self, flux_lats, *, key=None, inference: bool = True) -> jnp.ndarray:
        return self.flux_mlp(jnp.concatenate(flux_lats, axis=-1), key=key, inference=inference)
