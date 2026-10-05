"""Swin5DVQVAE — Swin5DAE whose bottleneck latent is quantized.

The latent projection maps the ``middle_pre`` features to the quantizer width (``embedding_dim``
for VQ, ``len(levels)`` for FSQ, ``log2(codebook_size)`` for LFQ) and the decoder consumes the
quantized latent. Outputs add ``vq_indices`` (the code grid) and, in training, ``vq_aux`` (the
per-sample quantizer terms reduced by ``vq.batch_loss``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr

from neugk_jax.pinc.quantizers import build_quantizer
from neugk_jax.pinc.swin5d_ae import Swin5DAE


class Swin5DVQVAE(Swin5DAE):
    """Vector-quantized Swin5D autoencoder."""

    vq: eqx.Module

    def __init__(self, *, vq_config: dict, key, **kwargs):
        kq, kae = jr.split(key)
        vq = build_quantizer(vq_config, key=kq)
        super().__init__(
            key=kae, **{**kwargs, "normalized_latent": False, "bottleneck_dim": vq.dim}
        )
        self.vq = vq

    @property
    def codebook_size(self) -> int:
        return int(self.vq.codebook_size)

    def quantize(self, z, *, inference: bool = True):
        q, idx, aux = self.vq(z.reshape(-1, z.shape[-1]), inference=inference)
        return q.reshape(z.shape), idx.reshape(z.shape[:-1]), aux

    def encode_indices(self, df, condition=None):
        return self.quantize(self.encode(df, condition))[1]

    def decode_from_indices(self, indices, condition=None):
        z = self.vq.codes(indices.reshape(self.bottleneck_grid_size))
        return self.decode(z, condition)

    def bottleneck(self, z, *, inference: bool = True):
        q, idx, aux = self.quantize(z, inference=inference)
        extra = {"vq_indices": idx}
        if not inference:
            extra["vq_aux"] = aux
        return q, extra


@eqx.filter_jit
def encode_indices_batch(ae, df, cond):
    return jax.vmap(ae.encode_indices)(df, cond)


def precompute_vq_indices(dataset, ae, cache_file, *, cond_slots=None, batch_size: int = 4):
    # flattened int64 code grid per sample; switches the dataset to diff mode
    from neugk_jax.diffusion.latents import precompute_latents

    def encode_fn(df, cond):
        return encode_indices_batch(ae, df, None if cond_slots is None else cond[:, cond_slots])

    precompute_latents(dataset, encode_fn=encode_fn, cache_file=cache_file, batch_size=batch_size)
