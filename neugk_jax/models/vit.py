"""Vision transformer blocks (full global attention) and their stacks."""

from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx
import jax.random as jr

from neugk_jax.models.attention import MultiHeadSelfAttention
from neugk_jax.models.swin import BlockStack, _DropPath, block_stack
from neugk_jax.models.utils import MLP, DiTModulation, gelu, make_norm, split_key


class ViTBlock(eqx.Module):
    """Standard pre-norm transformer block on ``(n_tokens, dim)`` (no windowing)."""

    norm1: object
    norm2: object
    attn: MultiHeadSelfAttention
    mlp: MLP
    drop_path: _DropPath

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        norm_affine: bool = False,
        rms_norm: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gated_attention: bool = False,
    ):
        katt, kmlp = jr.split(key, 2)
        self.norm1 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.norm2 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads,
            key=katt,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            gated_attention=gated_attention,
        )
        self.mlp = MLP([dim, max(int(dim * mlp_ratio), dim), dim], key=kmlp, act_fn=act_fn)
        self.drop_path = _DropPath(drop_path)

    def __call__(self, x, *, key=None, inference=True):
        key1, key2 = split_key(key, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)), key=key1, inference=inference)
        return x + self.drop_path(self.mlp(self.norm2(x)), key=key2, inference=inference)


class DiTViTBlock(eqx.Module):
    """ViT block with DiT modulation (gated, pre-norm residuals)."""

    norm1: object
    norm2: object
    attn: MultiHeadSelfAttention
    mlp: MLP
    drop_path: _DropPath
    mod: DiTModulation

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        *,
        key,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = True,
        rms_norm: bool = False,
    ):
        katt, kmlp, kmod = jr.split(key, 3)
        self.norm1 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.norm2 = make_norm(dim, rms=rms_norm, affine=norm_affine)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads,
            key=katt,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            gated_attention=gated_attention,
        )
        self.mlp = MLP([dim, max(int(dim * mlp_ratio), dim), dim], key=kmlp, act_fn=act_fn)
        self.drop_path = _DropPath(drop_path)
        self.mod = DiTModulation(cond_dim, dim, key=kmod)

    def __call__(self, x, cond, *, key=None, inference=True):
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.mod(cond)
        key1, key2 = split_key(key, 2)
        h = self.attn(self.norm1(x) * (1.0 + scale_msa) + shift_msa)
        x = x + gate_msa * self.drop_path(h, key=key1, inference=inference)
        h2 = self.mlp(self.norm2(x) * (1.0 + scale_mlp) + shift_mlp)
        return x + gate_mlp * self.drop_path(h2, key=key2, inference=inference)


def vit_layer(
    dim: int,
    depth: int,
    num_heads: int,
    *,
    key,
    cond_dim: Optional[int] = None,
    cond_mode: str = "dit",
    mlp_ratio: float = 4.0,
    drop_path: float = 0.0,
    act_fn: Callable = gelu,
    use_checkpoint: bool = False,
    qkv_bias: bool = False,
    qk_norm: bool = False,
    gated_attention: bool = False,
    norm_affine: bool = False,
    rms_norm: bool = False,
) -> BlockStack:
    """``depth`` ViT blocks over the flattened ``(*grid, dim)`` tokens."""
    common = dict(mlp_ratio=mlp_ratio, drop_path=drop_path, act_fn=act_fn, qkv_bias=qkv_bias)
    attn_kw = dict(
        qk_norm=qk_norm, gated_attention=gated_attention, norm_affine=norm_affine, rms_norm=rms_norm
    )

    def plain(i, k):
        return ViTBlock(dim, num_heads, key=k, **attn_kw, **common)

    def dit(i, k):
        return DiTViTBlock(dim, num_heads, cond_dim, key=k, **attn_kw, **common)

    return block_stack(
        plain,
        dit,
        depth,
        dim,
        key=key,
        cond_dim=cond_dim,
        cond_mode=cond_mode,
        tokens=True,
        use_checkpoint=use_checkpoint,
    )
