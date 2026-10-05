"""Per-sample error metrics shared by the losses and the evaluators, and the training losses."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def _flat(x):
    return x.reshape(x.shape[0], -1)


def per_sample_mse(p, t):
    return jnp.mean(_flat(p - t) ** 2, axis=-1)


def per_sample_rel_l2(p, t, eps: float = 1e-12):
    return jnp.linalg.norm(_flat(p - t), axis=-1) / (jnp.linalg.norm(_flat(t), axis=-1) + eps)


def per_sample_rel_l1(p, t, eps: float = 1e-8):
    return jnp.sum(jnp.abs(_flat(p - t)), axis=-1) / (jnp.sum(jnp.abs(_flat(t)), axis=-1) + eps)


def per_sample_rel_norm_mse(p, t, eps: float = 1e-4):
    return jnp.sum(_flat(p - t) ** 2, axis=-1) / (jnp.sum(_flat(t) ** 2, axis=-1) + eps)


def rel_err(p, t, eps: float = 1e-12):
    return jnp.abs(p - t) / (jnp.abs(t) + eps)


def masked_mean(values, mask):
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def relative_norm_mse(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} != {target.shape}"
    return jnp.mean(per_sample_rel_norm_mse(pred, target, eps))


def df_loss(pred: jnp.ndarray, target: jnp.ndarray, *, separate_zf: bool = False) -> jnp.ndarray:
    """``df`` loss: plain MSE on the zf slot + relative-norm MSE elsewhere.

    With ``separate_zf=True`` channel slots 0:2 are the zf split and 2: the
    other components. Without separate_zf falls back to relative-norm MSE.
    """
    if separate_zf and pred.shape[1] >= 4:
        zf_loss = jnp.mean((pred[:, :2] - target[:, :2]) ** 2)
        return zf_loss + relative_norm_mse(pred[:, 2:], target[:, 2:])
    return relative_norm_mse(pred, target)


def recon_loss(pred, target, loss_type: str = "mse", *, separate_zf: bool = False, eps=1e-8):
    """Batch ``mse`` | ``l1`` | ``relative_mse`` | ``relative_l1`` (global ratios) loss.

    ``separate_zf`` sums it over the zf (0:2) and the other (2:) channel slots.
    """
    if separate_zf:
        zf = recon_loss(pred[:, :2], target[:, :2], loss_type)
        return zf + recon_loss(pred[:, 2:], target[:, 2:], loss_type)
    if loss_type == "l1":
        return l1(pred, target)
    d = pred - target
    if loss_type == "mse":
        return jnp.mean(d**2)
    if loss_type == "relative_mse":
        return jnp.sum(d**2) / (jnp.sum(target**2) + eps)
    if loss_type == "relative_l1":
        return jnp.sum(jnp.abs(d)) / (jnp.sum(jnp.abs(target)) + eps)
    raise NotImplementedError(f"loss_type {loss_type!r} is not supported")


def l1(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred - target.reshape(pred.shape)))


def integral_losses(
    geom_t: dict,
    pred_df: jnp.ndarray,
    pred_phi: Optional[jnp.ndarray],
    tgt_phi: jnp.ndarray,
    tgt_flux: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Physics-integral losses on denormalized batches.

    ``pred_df`` is ``(B, C, vp, mu, s, x, y)`` (a separate-zf ``C=4`` layout is
    recombined), ``geom_t`` a batched :func:`precompute_geometry` dict. Returns
    ``phi_int = mse(phi(df), tgt_phi)`` and
    ``flux_int = mean(pflux^2) + mse(eflux(df, phi), tgt_flux)``.
    """
    from neugk_jax.evaluate.integrals import flux_integral
    from neugk_jax.utils import recombine_zf

    pred_df = recombine_zf(pred_df, axis=1)
    if pred_phi is None:
        phi_int, (pflux, eflux, _) = jax.vmap(flux_integral)(geom_t, pred_df)
    else:
        phi_int, (pflux, eflux, _) = jax.vmap(flux_integral)(geom_t, pred_df, pred_phi)
    tgt_phi = tgt_phi.reshape(phi_int.shape)
    return {
        "phi_int": jnp.mean((phi_int - tgt_phi) ** 2),
        "flux_int": jnp.mean(pflux**2) + jnp.mean((eflux - tgt_flux.reshape(eflux.shape)) ** 2),
    }
