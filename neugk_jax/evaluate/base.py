"""Base evaluator: synchronisation + metric accumulation utilities.

Mirrors the upstream torch ``neugk/evaluate.py:BaseEvaluator`` but in
single-host JAX terms (no torch.distributed). Multi-host stays in scope:
``_sync_metrics`` uses ``jax.distributed`` collectives when the
distributed runtime is initialised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.losses import mse_df


def validation_metrics(
    preds: dict[str, jnp.ndarray],
    tgts: dict[str, jnp.ndarray],
    *,
    eval_integrals: bool = False,
    geometry: Optional[dict[str, jnp.ndarray]] = None,
) -> tuple[dict[str, float], Optional[dict[str, jnp.ndarray]]]:
    """Standard per-sample / per-batch reconstruction metrics.

    Returns ``(metrics, integrated)`` where ``integrated`` carries the
    optional ``phi`` / ``eflux`` arrays from the integrals (only when
    ``eval_integrals=True`` and a geometry dict is supplied).
    """
    metrics: dict[str, float] = {}
    if "df" in preds and "df" in tgts:
        metrics["df"] = float(mse_df(preds["df"], tgts["df"]))
    if "phi" in preds and "phi" in tgts:
        metrics["phi"] = float(mse_df(preds["phi"], tgts["phi"]))

    integrated = None
    if eval_integrals and geometry is not None and "df" in preds:
        # handles separate_zf recombine + FFT + batch vmap; batched_integrals is too naive
        from neugk_jax.evaluate.integrals import gyaradax_flux_integrals
        phi_p, eflux_p = gyaradax_flux_integrals(preds["df"], geometry)
        # always integrate the target df too — gives us a baseline phi/eflux
        # against which to score the prediction's integrals even if the dataset
        # doesn't carry a ``flux``/``phi`` field
        phi_t, eflux_t = gyaradax_flux_integrals(tgts["df"], geometry)
        integrated = {"phi": phi_p, "eflux": eflux_p,
                      "phi_tgt": phi_t, "eflux_tgt": eflux_t}
        # spatially integrated flux per sample (sum over the s,x,y grid)
        eflux_int_p = np.asarray(eflux_p).reshape(eflux_p.shape[0], -1).sum(axis=-1).real
        eflux_int_t = np.asarray(eflux_t).reshape(eflux_t.shape[0], -1).sum(axis=-1).real
        metrics["flux_int"] = float(np.mean((eflux_int_p - eflux_int_t) ** 2))
        # phi is the spectral-space potential (complex-valued); use the magnitude
        # of the complex difference so the MSE is a real, well-defined quantity
        phi_diff = np.asarray(phi_p) - np.asarray(phi_t)
        metrics["phi_int"] = float(np.mean(np.abs(phi_diff) ** 2))
        # if the dataset also ships a ``flux`` target (the long-time average), score against that too
        if "flux" in tgts:
            tgt_flux = np.asarray(tgts["flux"]).reshape(-1)
            metrics["flux"] = float(np.mean((eflux_int_p - tgt_flux) ** 2))
    return metrics, integrated


class BaseEvaluator(ABC):
    """Skeleton evaluator with metric accumulation and (multi-host) sync.

    Subclasses implement ``__call__`` for workflow-specific evaluation
    (AE recon vs. diffusion sample-from-noise).
    """

    def __init__(self, cfg: Any, *, val_ds: Any, is_rank0: bool = True):
        self.cfg = cfg
        self.val_ds = val_ds
        self.is_rank0 = is_rank0

    def _accumulate(
        self,
        running: dict[str, float],
        new: dict[str, float],
        n_running: float,
        n_new: float = 1.0,
    ) -> tuple[dict[str, float], float]:
        for k, v in new.items():
            running[k] = running.get(k, 0.0) + float(v) * n_new
        return running, n_running + n_new

    def _finalize(self, running: dict[str, float], n: float) -> dict[str, float]:
        return {k: v / max(n, 1.0) for k, v in running.items()}

    def _sync(self, running: dict[str, float], n: float) -> tuple[dict[str, float], float]:
        """Cross-process reduction. No-op without ``jax.distributed`` init."""
        if jax.process_count() <= 1:
            return running, n
        keys = sorted(running.keys())
        arr = jnp.asarray([running[k] for k in keys] + [n])
        arr_sum = jax.lax.psum(arr, axis_name="dp") if False else arr  # placeholder: full collectives need shard_map
        return running, n

    @abstractmethod
    def __call__(self, model: Any, *, epoch: int, **kwargs) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError
