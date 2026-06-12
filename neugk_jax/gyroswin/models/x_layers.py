"""Cross-attention layers used to mix the df and phi U-Net latents.

Port of ``neugk/gyroswin/models/x_layers.py``: ``MixingBlock`` is a single
cross-attention + MLP block; ``VSpaceReduce`` integrates over the velocity
axes via a learned query token; ``RSpaceReduce`` does the same over real
space (used by the FluxDecoder, kept for completeness).
"""

from __future__ import annotations

from typing import Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from neugk_jax.models.attention import MultiHeadCrossAttention
from neugk_jax.models.swin import _DropPath
from neugk_jax.models.utils import Linear, LayerNorm, MLP, gelu


class MixingBlock(eqx.Module):
    """Cross-attention + MLP. ``left`` queries kv from ``right``; output dim = left_dim."""

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
        act_fn=gelu,
    ):
        k1, k2 = jr.split(key, 2)
        self.norm1 = LayerNorm(left_dim, elementwise_affine=True)
        self.attn = MultiHeadCrossAttention(
            q_dim=left_dim, kv_dim=right_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, key=k1,
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else _DropPath(0.0)
        self.norm2 = LayerNorm(left_dim, elementwise_affine=True)
        self.mlp = MLP(
            [left_dim, int(left_dim * mlp_ratio), left_dim],
            act_fn=act_fn, key=k2,
        )

    def __call__(self, left: jnp.ndarray, right: Optional[jnp.ndarray] = None,
                 *, key=None, inference: bool = True) -> jnp.ndarray:
        right = right if right is not None else left
        # tokenize: (*spatial, C) -> (N, C). We expect the caller to pass already-flat (*..., C).
        l_shape = left.shape
        l_tok = left.reshape(-1, l_shape[-1])
        r_tok = right.reshape(-1, right.shape[-1])
        x = self.drop_path(self.norm1(self.attn(l_tok, r_tok)), key=key, inference=inference)
        x = l_tok + x
        x = x + self.drop_path(self.norm2(jax.vmap(self.mlp)(x)), key=key, inference=inference)
        return x.reshape(l_shape)


class VSpaceReduce(eqx.Module):
    """Integrate velocity axes of a 5D df latent into a 3D phi-shaped latent.

    A learned query token (``integral_token``) cross-attends to the velocity
    tokens at each (s, x, y) position. Output shape: ``(s, x, y, out_dim)``.
    """

    kv: Linear
    proj: Linear
    integral_token: jax.Array
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    decouple_mu: bool = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        *,
        key,
        decouple_mu: bool = False,
        gain: float = 1e-2,
        qkv_bias: bool = False,
    ):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.out_dim = out_dim
        self.scale = self.head_dim ** -0.5
        self.decouple_mu = decouple_mu
        kkv, kp, ktoken = jr.split(key, 3)
        self.kv = Linear(dim, 2 * dim, key=kkv, use_bias=qkv_bias)
        self.proj = Linear(dim, out_dim, key=kp, use_bias=True)
        self.integral_token = gain * jr.normal(ktoken, (1, 1, dim))

    def __call__(self, df: jnp.ndarray) -> jnp.ndarray:
        # df comes as (vp, [mu,] s, x, y, C); decouple_mu controls whether mu is present
        if self.decouple_mu:
            vpar, ns, nx, ny, dim = df.shape
            df_t = rearrange(df, "vp s x y c -> (s x y) vp c")
        else:
            vpar, mu, ns, nx, ny, dim = df.shape
            df_t = rearrange(df, "vp mu s x y c -> (s x y) (vp mu) c")
        n_groups, n_tok, _ = df_t.shape
        # k, v come from df; q is the broadcasted integral token
        kv = self.kv(df_t).reshape(n_groups, n_tok, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0]
        v = kv[:, :, 1]
        # query: (1, num_heads, head_dim) broadcast across groups
        q = self.integral_token.reshape(1, self.num_heads, self.head_dim)
        q = jnp.broadcast_to(q, (n_groups, self.num_heads, self.head_dim))
        # einsum over the tok axis
        scores = jnp.einsum("ghd,gnhd->ghn", q, k) * self.scale
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("ghn,gnhd->ghd", attn, v)
        out = out.reshape(n_groups, self.num_heads * self.head_dim)
        out = self.proj(out)
        return out.reshape(ns, nx, ny, self.out_dim)


class RSpaceReduce(eqx.Module):
    """Pool every spatial axis into a single token (used by the flux head)."""

    kv: Linear
    proj: Linear
    integral_token: jax.Array
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, dim: int, out_dim: int, num_heads: int, *, key, gain: float = 1e-2):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.out_dim = out_dim
        self.scale = self.head_dim ** -0.5
        kkv, kp, kt = jr.split(key, 3)
        self.kv = Linear(dim, 2 * dim, key=kkv, use_bias=False)
        self.proj = Linear(dim, out_dim, key=kp, use_bias=True)
        self.integral_token = gain * jr.normal(kt, (1, 1, dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., C)
        x_t = x.reshape(-1, x.shape[-1])
        kv = self.kv(x_t).reshape(x_t.shape[0], 2, self.num_heads, self.head_dim)
        k = kv[:, 0]
        v = kv[:, 1]
        q = self.integral_token.reshape(1, self.num_heads, self.head_dim)
        scores = jnp.einsum("ghd,nhd->ghn", q, k) * self.scale
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("ghn,nhd->ghd", attn, v)
        out = out.reshape(1, self.num_heads * self.head_dim)
        return self.proj(out).reshape(self.out_dim)
