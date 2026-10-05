"""Vector quantizers of the VQ-VAE bottleneck.

Each maps per-sample tokens ``(N, D)`` to ``(quantized, indices, aux)``, straight-through when
``inference=False``; ``aux`` (empty at inference) holds the per-sample terms ``batch_loss``
reduces over the vmapped batch.

* ``VectorQuantizer`` — euclidean or cosine codebook kept in buffers, advanced outside the
  gradient by ``ema_update`` (EMA with dead-code replacement).
* ``FSQ`` — finite scalar quantization, implicit codebook of ``prod(levels)`` codes.
* ``LFQ`` — lookup-free sign quantization with an entropy regularizer.
"""

from __future__ import annotations

import math
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

HIGHEST = jax.lax.Precision.HIGHEST


def l2norm(x, eps: float = 1e-12):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def straight_through(x, q):
    return x + jax.lax.stop_gradient(q - x)


def entropy(prob, eps: float = 1e-5):
    return -jnp.sum(prob * jnp.log(jnp.maximum(prob, eps)), axis=-1)


class VectorQuantizer(eqx.Module):
    """Single-codebook VQ with EMA codebook updates and dead-code replacement."""

    embed: jax.Array
    embed_avg: jax.Array
    cluster_size: jax.Array
    buffer_fields = ("embed", "embed_avg", "cluster_size")
    dim: int = eqx.field(static=True)
    codebook_size: int = eqx.field(static=True)
    decay: float = eqx.field(static=True)
    commitment_weight: float = eqx.field(static=True)
    threshold_ema_dead_code: float = eqx.field(static=True)
    use_cosine_sim: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-5)

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        *,
        key,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        threshold_ema_dead_code: float = 2.0,
        use_cosine_sim: bool = False,
    ):
        # kaiming-uniform with fan_in = K * D
        bound = math.sqrt(6.0 / (codebook_size * dim))
        embed = jr.uniform(key, (codebook_size, dim), jnp.float32, -bound, bound)
        self.embed = l2norm(embed) if use_cosine_sim else embed
        self.embed_avg = jnp.array(self.embed)
        self.cluster_size = jnp.zeros((codebook_size,), jnp.float32)
        self.dim, self.codebook_size, self.decay = dim, codebook_size, decay
        self.commitment_weight = commitment_weight
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.use_cosine_sim = use_cosine_sim

    def codes(self, indices):
        return jax.lax.stop_gradient(self.embed)[indices]

    def __call__(self, z, *, inference: bool = True):
        x = z.astype(jnp.float32)
        x = l2norm(x) if self.use_cosine_sim else x
        embed = jax.lax.stop_gradient(self.embed)
        xe = jnp.matmul(x, embed.T, precision=HIGHEST)
        if self.use_cosine_sim:
            idx = jnp.argmax(xe, axis=-1)
        else:
            d2 = jnp.sum(x**2, -1, keepdims=True) + jnp.sum(embed**2, -1)[None] - 2.0 * xe
            idx = jnp.argmax(-jnp.sqrt(jnp.maximum(d2, 0.0)), axis=-1)
        q = embed[idx]
        if inference:
            return q, idx, {}
        commit = jnp.mean((jax.lax.stop_gradient(q) - x) ** 2)
        return straight_through(x, q), idx, {"commit": commit, "z": jax.lax.stop_gradient(x)}

    def batch_loss(self, aux):
        return self.commitment_weight * jnp.mean(aux["commit"])

    def ema_update(self, flat, indices, key):
        """``(quantizer, n_replaced)`` after one EMA step on tokens ``flat`` ``(M, D)`` with codes ``indices``.

        Codes whose EMA cluster size falls under ``threshold_ema_dead_code`` are re-seeded from
        random batch tokens.
        """
        k, w = self.codebook_size, 1.0 - self.decay
        flat = flat.astype(jnp.float32)
        bins = jax.ops.segment_sum(jnp.ones(indices.shape, jnp.float32), indices, k)
        cluster_size = self.cluster_size + w * (bins - self.cluster_size)
        embed_avg = self.embed_avg + w * (jax.ops.segment_sum(flat, indices, k) - self.embed_avg)
        n = jnp.sum(cluster_size)
        embed = embed_avg / ((cluster_size + self.eps) / (n + k * self.eps) * n)[:, None]
        embed = l2norm(embed) if self.use_cosine_sim else embed
        n_replaced = jnp.zeros((), jnp.int32)
        if self.threshold_ema_dead_code > 0:
            expired = cluster_size < self.threshold_ema_dead_code
            m = flat.shape[0]
            # distinct batch rows when the batch holds at least one token per code
            rows = jr.permutation(key, m)[:k] if m >= k else jr.randint(key, (k,), 0, m)
            sampled = l2norm(flat[rows]) if self.use_cosine_sim else flat[rows]
            reset = float(self.threshold_ema_dead_code)
            embed = jnp.where(expired[:, None], sampled, embed)
            cluster_size = jnp.where(expired, reset, cluster_size)
            embed_avg = jnp.where(expired[:, None], sampled * reset, embed_avg)
            n_replaced = jnp.sum(expired.astype(jnp.int32))
        new = eqx.tree_at(
            lambda q: (q.embed, q.embed_avg, q.cluster_size), self, (embed, embed_avg, cluster_size)
        )
        return new, n_replaced


class FSQ(eqx.Module):
    """Finite scalar quantization over ``len(levels)`` bounded, rounded channels."""

    levels: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, levels: Sequence[int]):
        self.levels = tuple(int(v) for v in levels)

    @property
    def dim(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels)

    def _tables(self):
        lv = jnp.asarray(self.levels, jnp.int32)
        basis = jnp.cumprod(jnp.asarray((1,) + self.levels[:-1], jnp.int32))
        return lv, basis, lv // 2

    def bound(self, z, eps: float = 1e-3):
        lv = jnp.asarray(self.levels, jnp.int32)
        half_l = (lv - 1).astype(jnp.float32) * (1 + eps) / 2
        offset = jnp.where(lv % 2 == 0, 0.5, 0.0)
        return jnp.tanh(z + jnp.arctanh(offset / half_l)) * half_l - offset

    def codes(self, indices):
        lv, basis, hw = self._tables()
        return ((indices[..., None] // basis) % lv - hw) / hw

    def __call__(self, z, *, inference: bool = True):
        _, basis, hw = self._tables()
        b = self.bound(z.astype(jnp.float32))
        q = straight_through(b, jnp.round(b)) / hw
        idx = jnp.round(jnp.sum((q * hw + hw) * basis, axis=-1)).astype(jnp.int32)
        return q, idx, {}

    def batch_loss(self, aux):
        return jnp.zeros((), jnp.float32)


class LFQ(eqx.Module):
    """Lookup-free quantization: sign bits per channel, the index is the bit pattern."""

    codebook_size: int = eqx.field(static=True)
    entropy_loss_weight: float = eqx.field(static=True)
    diversity_gamma: float = eqx.field(static=True)
    commitment_weight: float = eqx.field(static=True)
    inv_temperature: float = eqx.field(static=True, default=100.0)

    def __init__(
        self,
        codebook_size: int,
        *,
        entropy_loss_weight: float = 0.1,
        diversity_gamma: float = 1.0,
        commitment_weight: float = 0.25,
    ):
        if codebook_size & (codebook_size - 1):
            raise ValueError(f"LFQ codebook_size must be a power of 2, got {codebook_size}")
        self.codebook_size = int(codebook_size)
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_gamma = diversity_gamma
        self.commitment_weight = commitment_weight

    @property
    def dim(self) -> int:
        return self.codebook_size.bit_length() - 1

    def _bits(self):
        return 2 ** jnp.arange(self.dim - 1, -1, -1, dtype=jnp.int32)

    def codes(self, indices):
        return ((indices[..., None] & self._bits()) != 0).astype(jnp.float32) * 2 - 1

    def __call__(self, z, *, inference: bool = True):
        x = z.astype(jnp.float32)
        q = straight_through(x, jnp.where(x > 0, 1.0, -1.0))
        idx = jnp.sum((q > 0).astype(jnp.int32) * self._bits(), axis=-1)
        if inference:
            return q, idx, {}
        logits = 2 * jnp.matmul(x, self.codes(jnp.arange(self.codebook_size)).T, precision=HIGHEST)
        probs = jax.nn.softmax(logits * self.inv_temperature, axis=-1)
        aux = {
            "commit": jnp.mean((x - jax.lax.stop_gradient(q)) ** 2),
            "entropy": jnp.mean(entropy(probs)),
            "avg_probs": jnp.mean(probs, axis=0),
        }
        return q, idx, aux

    def batch_loss(self, aux):
        codebook_entropy = entropy(jnp.mean(aux["avg_probs"], axis=0))
        ent = jnp.mean(aux["entropy"]) - self.diversity_gamma * codebook_entropy
        return ent * self.entropy_loss_weight + jnp.mean(aux["commit"]) * self.commitment_weight


def build_quantizer(vq: dict, *, key):
    """Quantizer of a ``model.vq`` config (``quantizer``: ``vq`` | ``fsq`` | ``lfq``)."""
    kind = vq.get("quantizer", "vq")
    if kind == "vq":
        return VectorQuantizer(
            int(vq.get("embedding_dim", 256)),
            int(vq.get("codebook_size", 8192)),
            key=key,
            decay=float(vq.get("ema_decay", 0.99)),
            commitment_weight=float(vq.get("commitment_weight", 0.25)),
            threshold_ema_dead_code=float(vq.get("threshold_ema_dead_code", 2)),
            use_cosine_sim=vq.get("codebook_type", "euclidean") == "cosine",
        )
    if kind == "fsq":
        return FSQ(vq.get("levels", (8, 8, 8, 5, 5, 5)))
    if kind == "lfq":
        return LFQ(
            int(vq.get("codebook_size", 8192)),
            entropy_loss_weight=float(vq.get("entropy_loss_weight", 0.1)),
            diversity_gamma=float(vq.get("diversity_gamma", 1.0)),
            commitment_weight=float(vq.get("commitment_weight", 0.25)),
        )
    raise NotImplementedError(f"vq.quantizer {kind!r} is not supported")
