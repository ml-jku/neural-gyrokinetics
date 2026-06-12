"""Vision transformer layers (full global attention) for the bottleneck."""

from __future__ import annotations

import enum
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neugk_jax.models.attention import MultiHeadSelfAttention
from neugk_jax.models.utils import MLP, LayerNorm, RMSNorm, DiTModulation, gelu
from neugk_jax.models.swin import _DropPath


class LayerModes(enum.Enum):
    DOWNSAMPLE = "downsample"
    UPSAMPLE = "upsample"
    SEQUENCE = "sequence"


class ViTBlock(eqx.Module):
    """Standard transformer block (no windowing)."""

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
        self.norm1 = RMSNorm(dim, elementwise_affine=norm_affine) if rms_norm else LayerNorm(dim, elementwise_affine=norm_affine)
        self.norm2 = RMSNorm(dim, elementwise_affine=norm_affine) if rms_norm else LayerNorm(dim, elementwise_affine=norm_affine)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, key=katt,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            gated_attention=gated_attention, use_rpb=False,  # ViT has no windowing, so no RPB
        )
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = MLP([dim, hidden, dim], key=kmlp, act_fn=act_fn)
        self.drop_path = _DropPath(drop_path)

    def __call__(self, x, *, key=None, inference=False):
        # x: (n_tokens, dim); standard pre-norm transformer block
        key1, key2 = (None, None) if key is None else jr.split(key, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)), key=key1, inference=inference)
        x = x + self.drop_path(self.mlp(self.norm2(x)), key=key2, inference=inference)
        return x


class DiTViTBlock(eqx.Module):
    """ViT block with DiT modulation."""

    norm1: LayerNorm
    norm2: LayerNorm
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
        norm_affine: bool = True,
    ):
        katt, kmlp, kmod = jr.split(key, 3)
        self.norm1 = LayerNorm(dim, elementwise_affine=norm_affine)
        self.norm2 = LayerNorm(dim, elementwise_affine=norm_affine)
        self.attn = MultiHeadSelfAttention(dim, num_heads, key=katt, qkv_bias=qkv_bias)
        hidden = max(int(dim * mlp_ratio), dim)
        self.mlp = MLP([dim, hidden, dim], key=kmlp, act_fn=act_fn)
        self.drop_path = _DropPath(drop_path)
        self.mod = DiTModulation(cond_dim, dim, key=kmod)

    def __call__(self, x, cond, *, key=None, inference=False):
        # upstream order: (scale1, shift1, gate1, scale2, shift2, gate2) — matches DiT.forward in models/layers.py
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.mod(cond)
        shift_msa = shift_msa[None, :]
        scale_msa = scale_msa[None, :]
        gate_msa = gate_msa[None, :]
        shift_mlp = shift_mlp[None, :]
        scale_mlp = scale_mlp[None, :]
        gate_mlp = gate_mlp[None, :]
        key1, key2 = (None, None) if key is None else jr.split(key, 2)
        h = self.attn(self.norm1(x) * (1.0 + scale_msa) + shift_msa)
        x = x + gate_msa * self.drop_path(h, key=key1, inference=inference)
        h2 = self.mlp(self.norm2(x) * (1.0 + scale_mlp) + shift_mlp)
        x = x + gate_mlp * self.drop_path(h2, key=key2, inference=inference)
        return x


class ViTLayer(eqx.Module):
    """Stack of ViT blocks operating on flattened ``(*grid, dim)`` tokens."""

    blocks: list[ViTBlock]
    grid_size: tuple[int, ...] = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    drop_path: float = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)
    norm_layer: type = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        *,
        key,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        norm_layer: type = LayerNorm,
        use_checkpoint: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        gated_attention: bool = False,
        norm_affine: bool = False,
        rms_norm: bool = False,
        **_unused,
    ):
        keys = jr.split(key, depth)
        self.blocks = [
            ViTBlock(
                dim, num_heads, key=keys[i],
                mlp_ratio=mlp_ratio, drop_path=drop_path, act_fn=act_fn,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                gated_attention=gated_attention,
                norm_affine=norm_affine, rms_norm=rms_norm,
            )
            for i in range(depth)
        ]
        self.grid_size = tuple(grid_size)
        self.dim = dim
        self.drop_path = drop_path
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.act_fn = act_fn

    def __call__(self, x: jnp.ndarray, *, key=None, inference=False, **_):
        # x: (*grid, dim) → flatten → blocks → reshape
        spatial = x.shape[:-1]
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        keys = jr.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for blk, k in zip(self.blocks, keys):
            x = blk(x, key=k, inference=inference)
        return x.reshape(*spatial, dim)


class DiTLayer(eqx.Module):
    """Stack of DiT-ViT blocks with conditioning."""

    blocks: list[DiTViTBlock]
    grid_size: tuple[int, ...] = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    drop_path: float = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)
    norm_layer: type = eqx.field(static=True)
    act_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        num_heads: int,
        grid_size: Sequence[int],
        *,
        key,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        act_fn: Callable = gelu,
        norm_layer: type = LayerNorm,
        use_checkpoint: bool = False,
        qkv_bias: bool = False,
        norm_affine: bool = True,
        **_unused,
    ):
        keys = jr.split(key, depth)
        self.blocks = [
            DiTViTBlock(
                dim, num_heads, cond_dim, key=keys[i],
                mlp_ratio=mlp_ratio, drop_path=drop_path, act_fn=act_fn,
                qkv_bias=qkv_bias, norm_affine=norm_affine,
            )
            for i in range(depth)
        ]
        self.grid_size = tuple(grid_size)
        self.dim = dim
        self.drop_path = drop_path
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm_layer = norm_layer
        self.act_fn = act_fn

    def __call__(self, x, condition, *, key=None, inference=False, **_):
        spatial = x.shape[:-1]
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        keys = jr.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for blk, k in zip(self.blocks, keys):
            x = blk(x, condition, key=k, inference=inference)
        return x.reshape(*spatial, dim)
