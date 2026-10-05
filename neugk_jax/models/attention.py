"""Multi-head self- and cross-attention on flattened ``(n_tokens, dim)`` tokens per sample."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.embeddings import RPB
from neugk_jax.models.utils import Gate, Linear, RMSNorm, dropout, split_key


def einsum_attention(q, k, v, scale, bias=None, attn_drop=0.0, key=None, inference=True):
    """Softmax attention of ``q`` (..., n, heads, head_dim) over ``k``/``v`` (..., m, heads, head_dim)."""
    logits = jnp.einsum("...nhd,...mhd->...hnm", q, k) * scale
    if bias is not None:
        logits = logits + bias
    attn = jax.nn.softmax(logits, axis=-1)
    attn = dropout(attn, attn_drop, key=key, inference=inference)
    return jnp.einsum("...hnm,...mhd->...nhd", attn, v)


class MultiHeadSelfAttention(eqx.Module):
    """Multi-head self-attention with optional extras.

    Switches:

    * ``qkv_bias`` — toggle the qkv bias term.
    * ``qk_norm``  — RMSNorm per-head on q and k pre-softmax.
    * ``use_rpb``  — additive relative-position-bias from an internal tiny MLP.
    * ``gated_attention`` — headwise multiplicative gate from a separate
      Linear, applied to the attention output.
    * ``attn_drop`` / ``proj_drop`` — dropout on the attention probabilities
      and on the output projection, active only with a key and ``inference=False``.
    """

    qkv: Linear
    proj: Linear
    q_norm: object | None
    k_norm: object | None
    rpb: object | None
    gate: object | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)
    proj_drop: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rpb: bool = False,
        gated_attention: bool = False,
        window_size: Optional[tuple[int, ...]] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        kqkv, kproj, kqn, kkn, krpb, kgate = jr.split(key, 6)
        self.qkv = Linear(dim, 3 * dim, key=kqkv, use_bias=qkv_bias)
        self.proj = Linear(dim, dim, key=kproj, use_bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        if use_rpb:
            assert window_size is not None, "use_rpb requires window_size"
            self.rpb = RPB(window_size, num_heads, key=krpb)
        else:
            self.rpb = None
        self.gate = Gate(self.head_dim, key=kgate) if gated_attention else None
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def __call__(
        self,
        x: jnp.ndarray,
        attn_bias: Optional[jnp.ndarray] = None,
        *,
        key=None,
        inference: bool = True,
    ) -> jnp.ndarray:
        n, dim = x.shape
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # fold rpb bias into the attention bias slot
        if self.rpb is not None:
            rpb_bias = self.rpb()  # shape: (heads, sl, sl)
            attn_bias = rpb_bias if attn_bias is None else attn_bias + rpb_bias
        ka, kp = split_key(key, 2)
        out = einsum_attention(q, k, v, self.scale, attn_bias, self.attn_drop, ka, inference)
        # out: (n, H, D); apply optional headwise gate before flattening to (n, dim)
        if self.gate is not None:
            out = self.gate(out, q)
        out = out.reshape(n, dim)
        return dropout(self.proj(out), self.proj_drop, key=kp, inference=inference)


class MultiHeadCrossAttention(eqx.Module):
    """Cross-attention: queries from ``left``, keys/values from ``right``.

    Used by the GyroSwin mixing layers — ``left`` attends to ``right``,
    output dim matches ``left``. Both inputs are tokenised ``(n, dim)``;
    the caller flattens spatial axes before the call. ``attn_drop`` /
    ``proj_drop`` act as in ``MultiHeadSelfAttention``.
    """

    q: Linear
    kv: Linear
    proj: Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    attn_drop: float = eqx.field(static=True)
    proj_drop: float = eqx.field(static=True)

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        *,
        key,
        out_dim: Optional[int] = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        assert q_dim % num_heads == 0, f"q_dim={q_dim} not divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim**-0.5
        kq, kkv, kp = jr.split(key, 3)
        out_dim = out_dim or q_dim
        self.q = Linear(q_dim, q_dim, key=kq, use_bias=qkv_bias)
        self.kv = Linear(kv_dim, 2 * q_dim, key=kkv, use_bias=qkv_bias)
        self.proj = Linear(q_dim, out_dim, key=kp, use_bias=True)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def __call__(
        self, left: jnp.ndarray, right: jnp.ndarray, *, key=None, inference: bool = True
    ) -> jnp.ndarray:
        n_q, _ = left.shape
        n_kv, _ = right.shape
        q = self.q(left).reshape(n_q, self.num_heads, self.head_dim)
        kv = self.kv(right).reshape(n_kv, 2, self.num_heads, self.head_dim)
        k = kv[:, 0]
        v = kv[:, 1]
        ka, kp = split_key(key, 2)
        out = einsum_attention(q, k, v, self.scale, None, self.attn_drop, ka, inference)
        out = out.reshape(n_q, -1)
        return dropout(self.proj(out), self.proj_drop, key=kp, inference=inference)
