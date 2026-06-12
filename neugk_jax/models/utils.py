"""Primitive layers and small utilities used everywhere in the model code:

* Activation wrappers — thin top-level forwarders around ``jax.nn.*`` that
  pickle cleanly (the jit-wrapped ``jax.nn.relu``/``leaky_relu`` symbols
  fail pickle identity checks when stored as Equinox static fields).
* ``Linear``, ``LayerNorm`` — thin wrappers around ``eqx.nn.*`` that add
  arbitrary leading-dim support + mixed-precision dtype casting.
* ``MLP``, ``Film``, ``DiTModulation`` — small composites.
* ``RMSNorm``, ``Gate`` — used by the WindowAttention extras
  (``qk_norm``, ``gated_attention``).

All layers operate on tensors of shape ``(..., dim)`` — leading axes are
broadcast freely without explicit vmap.
"""

from __future__ import annotations

from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr




def gelu(x):
    """Exact (erf-based) GELU — matches ``torch.nn.GELU()`` default.

    ``jax.nn.gelu`` defaults to ``approximate=True`` (tanh approximation);
    we want exact so the math lines up with torch-trained weights.
    """
    return jax.nn.gelu(x, approximate=False)


def relu(x):
    """ReLU forwarder. We can't pickle ``jax.nn.relu`` directly because its
    pjit-wrapped identity drifts across imports; this top-level function does.
    """
    return jax.nn.relu(x)


def leaky_relu(x):
    """LeakyReLU(0.01) forwarder — same pickle workaround as ``relu``."""
    return jax.nn.leaky_relu(x, negative_slope=0.01)


def silu(x):
    """SiLU forwarder — same pickle workaround."""
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
    eps: float = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-5, elementwise_affine: bool = True):
        self.inner = eqx.nn.LayerNorm(
            (dim,),
            eps=eps,
            use_weight=elementwise_affine,
            use_bias=elementwise_affine,
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine
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


class MLP(eqx.Module):
    """Multi-layer perceptron over the last axis."""

    layers: list[Linear]
    act: Callable = eqx.field(static=True)
    last_act: bool = eqx.field(static=True)

    def __init__(
        self,
        dims: Sequence[int],
        *,
        key,
        act_fn: Callable = gelu,
        use_bias: bool = True,
        last_act: bool = False,
    ):
        keys = jr.split(key, len(dims) - 1)
        self.layers = [
            Linear(dims[i], dims[i + 1], key=keys[i], use_bias=use_bias)
            for i in range(len(dims) - 1)
        ]
        self.act = act_fn
        self.last_act = last_act

    def __call__(self, x: jax.Array) -> jax.Array:
        n = len(self.layers)
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i < n - 1 or self.last_act:
                x = self.act(x)
        return x


class Film(eqx.Module):
    """Feature-wise linear modulation: scale * x + shift driven by a condition."""

    proj: Linear
    dim: int = eqx.field(static=True)

    def __init__(self, cond_dim: int, dim: int, *, key):
        self.proj = Linear(cond_dim, 2 * dim, key=key)
        self.dim = dim

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        # cond: (..., cond_dim); x: (..., dim)
        scale, shift = jnp.split(self.proj(jax.nn.silu(cond)), 2, axis=-1)
        # add trailing spatial singleton axes so scale/shift broadcast over (*spatial, dim)
        while scale.ndim < x.ndim:
            scale = scale[..., None, :]
            shift = shift[..., None, :]
        return x * (1.0 + scale) + shift


class DiTModulation(eqx.Module):
    """DiT-style 6-way modulation: (shift1, scale1, gate1, shift2, scale2, gate2)."""

    proj: Linear
    dim: int = eqx.field(static=True)

    def __init__(self, cond_dim: int, dim: int, *, key):
        # small init keeps the initial residual near identity
        wkey, _ = jr.split(key)
        self.proj = Linear(cond_dim, 6 * dim, key=wkey)
        self.dim = dim

    def __call__(self, cond: jax.Array):
        # cond: (..., cond_dim) → 6 tensors of shape (..., dim); SiLU already applied in ContinuousConditionEmbed
        return jnp.split(self.proj(cond), 6, axis=-1)




class RMSNorm(eqx.Module):
    """Root-mean-square normalisation on the last axis.

    Mirrors ``nn.RMSNorm(dim, elementwise_affine=...)``. With
    ``elementwise_affine=False`` no learnable weight is allocated —
    matches torch's swin block norms.
    """

    weight: jax.Array | None
    eps: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)

    def __init__(
        self, dim: int, *, eps: float = 1e-8, elementwise_affine: bool = True
    ):
        self.weight = jnp.ones((dim,)) if elementwise_affine else None
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine

    def __call__(self, x: jax.Array) -> jax.Array:
        in_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + self.eps)
        y = x32 / rms
        if self.elementwise_affine:
            y = y * self.weight
        return y.astype(in_dtype)


class Gate(eqx.Module):
    """Headwise multiplicative gate (``gated_attention=True`` in upstream).

    Mirrors ``Gate(head_dim) = Sequential(ReLU, Linear(d, d), Sigmoid)``.
    Torch stores the Linear at ``gate.gate.1.{weight,bias}``; ours lives at
    ``gate.proj.inner.{weight,bias}`` (translator bridges the two).
    """

    proj: Linear

    def __init__(self, head_dim: int, *, key):
        self.proj = Linear(head_dim, head_dim, key=key, use_bias=True)

    def __call__(self, x: jax.Array, g: jax.Array) -> jax.Array:
        # x, g: (n, H, D); gate is sigmoid(linear(relu(g)))
        return x * jax.nn.sigmoid(self.proj(relu(g)))
