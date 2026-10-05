"""PINC losses: per-snapshot relative L1 of the field-solve integrals and log1p spectral losses."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from neugk_jax.evaluate.integrals import spectral_integrals
from neugk_jax.losses import per_sample_rel_l1
from neugk_jax.utils import recombine_zf


def relative_l1_snap(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(per_sample_rel_l1(pred, target))


def served_spectral_loss(pred, target, mode_std=None, eps: float = 1e-8) -> jnp.ndarray:
    pl, tl = jnp.log1p(jnp.abs(pred)), jnp.log1p(jnp.abs(target))
    std = jnp.std(tl, axis=0, ddof=1) if mode_std is None else mode_std
    return jnp.mean(jnp.abs((pl - tl) / (std + eps)))


def pinc_integrals(geom_t: dict, df: jnp.ndarray, *, ds: float) -> dict:
    """Batched :func:`spectral_integrals` of a denormalized df ``(B, C, vp, mu, s, x, y)``.

    ``geom_t`` is a batched :func:`precompute_geometry` dict; separate-zf layouts are recombined.
    """

    return jax.vmap(lambda g, d: spectral_integrals(g, d, ds=ds))(geom_t, recombine_zf(df, axis=1))


def pinc_terms(p: dict, t: dict, mode_std: Optional[dict] = None) -> dict[str, jnp.ndarray]:
    """PINC loss terms of prediction integrals ``p`` against target integrals ``t``.

    ``phi_int`` and ``flux_int = mean|pflux| + eflux`` are per-snapshot relative L1,
    ``kyspec`` / ``qspec`` the log1p spectral loss over ``mode_std``; ``phi_int_mse`` and
    ``flux_int_l1`` (``mean|pflux| + mean|eflux - eflux_t|``) are gradient-free monitors.
    """
    mode_std = mode_std or {}
    pflux = jnp.mean(jnp.abs(p["pflux"]))
    return {
        "phi_int": relative_l1_snap(p["phi"], t["phi"]),
        "flux_int": pflux + relative_l1_snap(p["eflux"], t["eflux"]),
        "kyspec": served_spectral_loss(p["kyspec"], t["kyspec"], mode_std.get("kyspec")),
        "qspec": served_spectral_loss(p["qspec"], t["qspec"], mode_std.get("qspec")),
        "phi_int_mse": jax.lax.stop_gradient(jnp.mean((p["phi"] - t["phi"]) ** 2)),
        "flux_int_l1": jax.lax.stop_gradient(pflux + jnp.mean(jnp.abs(p["eflux"] - t["eflux"]))),
    }


def pinc_losses(geom_t, pred_df, tgt_df, *, ds: float, mode_std=None):
    p = pinc_integrals(geom_t, pred_df, ds=ds)
    t = jax.lax.stop_gradient(pinc_integrals(geom_t, tgt_df, ds=ds))
    return pinc_terms(p, t, mode_std)
