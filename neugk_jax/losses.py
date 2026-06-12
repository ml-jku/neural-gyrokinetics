"""Loss functions. AE/diffusion training uses relative-norm MSE on df,
mirroring upstream ``neugk/losses.py:relative_norm_mse``."""

from __future__ import annotations

import jax.numpy as jnp


def mse_df(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error reduced to a scalar (raw, unnormalized)."""
    return jnp.mean((pred - target) ** 2)


def relative_norm_mse(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    """``mean_b ||pred - target||² / (||target||² + eps)``.

    Mirrors ``neugk/losses.py:relative_norm_mse`` (squared variant). Batch
    axis 0 is preserved as the reduction axis; everything else flattened.
    Lands in the 1-10 range when target is z-scored unit-variance.
    """
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} != {target.shape}"
    if pred.ndim > 1:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    diff_sq = jnp.sum((pred - target) ** 2, axis=-1)
    tgt_sq = jnp.sum(target ** 2, axis=-1)
    return jnp.mean(diff_sq / (tgt_sq + eps))


def df_loss(pred: jnp.ndarray, target: jnp.ndarray, *, separate_zf: bool = False) -> jnp.ndarray:
    """Upstream ``df`` loss: plain MSE on zf slot + relative-norm MSE elsewhere.

    Matches ``neugk/losses.py:LossWrapper.forward`` lines 178-185 when
    ``separate_zf=True``. Channel slots 0:2 are the zf split, 2: are the
    other components. Without separate_zf falls back to relative-norm MSE.
    """
    if separate_zf and pred.shape[1] >= 4:
        zf_loss = jnp.mean((pred[:, :2] - target[:, :2]) ** 2)
        other_loss = relative_norm_mse(pred[:, 2:], target[:, 2:])
        return zf_loss + other_loss
    return relative_norm_mse(pred, target)


def per_sample_mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Per-sample MSE, returns shape (B,). Used for diffusion SNR weighting."""
    diff = (pred - target) ** 2
    return diff.reshape(diff.shape[0], -1).mean(axis=-1)
