"""Primitive layers and small utilities used everywhere in the model code:

* Activation wrappers — thin top-level forwarders around ``jax.nn.*`` that
  pickle cleanly (the jit-wrapped ``jax.nn.relu``/``leaky_relu`` symbols
  fail pickle identity checks when stored as Equinox static fields).
* ``Linear``, ``LayerNorm`` — thin wrappers around ``eqx.nn.*`` that add
  arbitrary leading-dim support + mixed-precision dtype casting.
* ``MLP``, ``DiTModulation`` — small composites.
* ``dropout``, ``make_norm`` — stateless dropout and the LayerNorm/RMSNorm switch.
* ``RMSNorm``, ``Gate`` — used by the WindowAttention extras
  (``qk_norm``, ``gated_attention``).

All layers operate on tensors of shape ``(..., dim)`` — leading axes are
broadcast freely without explicit vmap.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def split_key(key, n):
    return [None] * n if key is None else list(jr.split(key, n))


def gelu(x):
    # exact erf gelu; jax defaults to the tanh approximation
    return jax.nn.gelu(x, approximate=False)


def relu(x):
    return jax.nn.relu(x)


def leaky_relu(x):
    return jax.nn.leaky_relu(x, negative_slope=0.01)


def silu(x):
    return jax.nn.silu(x)


class Linear(eqx.Module):
    """Wrapper around ``eqx.nn.Linear`` with two extras:

    1. **Arbitrary leading dims.** ``eqx.nn.Linear`` requires 1D input; here
       we apply ``x @ W.T`` so any shape ``(..., in_features)`` works.
    2. **Mixed precision.** Weights live in fp32 (per the policy) but are
       cast to the input dtype at call time, so a bf16 activation never has
       to materialize a fresh weight copy as a leaf.
    """

    inner: eqx.nn.Linear
    use_bias: bool = eqx.field(static=True)

    def __init__(self, in_dim: int, out_dim: int, *, key, use_bias: bool = True):
        self.inner = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=key)
        self.use_bias = use_bias

    @property
    def weight(self):
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.inner.weight.astype(x.dtype)
        y = jnp.matmul(x, w.T)
        if self.use_bias:
            y = y + self.inner.bias.astype(x.dtype)
        return y


class LayerNorm(eqx.Module):
    """Wrapper around ``eqx.nn.LayerNorm``.

    Equinox's LayerNorm is shape-specific (single sample). We need it over
    the trailing axis of an arbitrary leading-dim tensor, with an fp32 upcast
    around the variance for stability under bf16 activations.
    """

    inner: eqx.nn.LayerNorm
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-5, elementwise_affine: bool = True):
        self.inner = eqx.nn.LayerNorm(
            (dim,), eps=eps, use_weight=elementwise_affine, use_bias=elementwise_affine
        )
        self.dim = dim

    @property
    def weight(self):
        return self.inner.weight

    @property
    def bias(self):
        return self.inner.bias

    def __call__(self, x: jax.Array) -> jax.Array:
        in_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        flat = x32.reshape(-1, self.dim)
        out = jax.vmap(self.inner)(flat)
        return out.reshape(x.shape).astype(in_dtype)


def dropout(x: jax.Array, rate: float, *, key=None, inference: bool = True) -> jax.Array:
    """Inverted dropout; identity when ``rate == 0``, ``inference`` or no ``key``."""
    if inference or rate == 0.0 or key is None:
        return x
    keep = 1.0 - rate
    mask = jr.bernoulli(key, p=keep, shape=x.shape)
    return jnp.where(mask, x / keep, 0.0).astype(x.dtype)


class MLP(eqx.Module):
    """Multi-layer perceptron over the last axis, with optional dropout after every linear."""

    layers: list[Linear]
    act: Callable = eqx.field(static=True)
    drop: float = eqx.field(static=True)

    def __init__(
        self,
        dims: Sequence[int],
        *,
        key,
        act_fn: Callable = gelu,
        use_bias: bool = True,
        drop: float = 0.0,
    ):
        keys = jr.split(key, len(dims) - 1)
        self.layers = [
            Linear(dims[i], dims[i + 1], key=keys[i], use_bias=use_bias)
            for i in range(len(dims) - 1)
        ]
        self.act = act_fn
        self.drop = drop

    def __call__(self, x: jax.Array, *, key=None, inference: bool = True) -> jax.Array:
        n = len(self.layers)
        for i, (lyr, k) in enumerate(zip(self.layers, split_key(key, n))):
            x = dropout(lyr(x), self.drop, key=k, inference=inference)
            if i < n - 1:
                x = self.act(x)
        return x


class DiTModulation(eqx.Module):
    """DiT-style 6-way modulation: (scale1, shift1, gate1, scale2, shift2, gate2)."""

    proj: Linear

    def __init__(self, cond_dim: int, dim: int, *, key):
        self.proj = Linear(cond_dim, 6 * dim, key=key)

    def __call__(self, cond: jax.Array):
        # cond: (..., cond_dim) → 6 tensors of shape (..., dim); SiLU already applied in ContinuousConditionEmbed
        return jnp.split(self.proj(cond), 6, axis=-1)


class RMSNorm(eqx.Module):
    """Root-mean-square normalisation on the last axis.

    With ``elementwise_affine=False`` no learnable weight is allocated.
    """

    weight: jax.Array | None
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-8, elementwise_affine: bool = True):
        self.weight = jnp.ones((dim,)) if elementwise_affine else None
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        in_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + self.eps)
        y = x32 / rms
        if self.weight is not None:
            y = y * self.weight
        return y.astype(in_dtype)


def make_norm(dim: int, *, rms: bool, affine: bool = True):
    return (
        RMSNorm(dim, elementwise_affine=affine)
        if rms
        else LayerNorm(dim, elementwise_affine=affine)
    )


class Gate(eqx.Module):
    """Headwise multiplicative gate: ``sigmoid(linear(relu(g))) * x``."""

    proj: Linear

    def __init__(self, head_dim: int, *, key):
        self.proj = Linear(head_dim, head_dim, key=key, use_bias=True)

    def __call__(self, x: jax.Array, g: jax.Array) -> jax.Array:
        # x, g: (n, H, D); gate is sigmoid(linear(relu(g)))
        return x * jax.nn.sigmoid(self.proj(relu(g)))


def trainable_mask(model):
    """Bool pytree over ``model``: True on trainable arrays, False on non-arrays and buffers.

    A module marks buffers by listing field names in ``buffer_fields``; the mask is built
    on a concrete model and reused as the filter spec inside jitted steps.
    """
    frozen = set()

    def visit(node):
        if isinstance(node, eqx.Module):
            for name in getattr(node, "buffer_fields", ()):
                if getattr(node, name, None) is not None:
                    frozen.add(id(getattr(node, name)))
            for f in dataclasses.fields(node):
                visit(getattr(node, f.name, None))
        elif isinstance(node, (list, tuple)):
            for v in node:
                visit(v)
        elif isinstance(node, dict):
            for v in node.values():
                visit(v)

    visit(model)
    return jax.tree_util.tree_map(lambda x: eqx.is_array(x) and id(x) not in frozen, model)
