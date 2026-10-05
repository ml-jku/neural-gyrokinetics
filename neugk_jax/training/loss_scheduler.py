"""Progress-based loss-weight schedules + multi-task loss builder.

Training-mode loss terms used by GyroSwin: ``df``, ``phi``,
``flux``/``fluxavg`` and the physics-integral losses ``phi_int``/``flux_int``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import jax.numpy as jnp

from neugk_jax.losses import df_loss, l1, relative_norm_mse
from neugk_jax.utils import to_dict


def linear_burn_in(
    start: float, end: float, start_fraction: float, end_fraction: float
) -> Callable[[float], float]:
    """Linear ramp from ``start`` to ``end`` over [start_fraction, end_fraction]."""

    def fn(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress > end_fraction:
            return end
        if progress < start_fraction:
            return start
        return start + (progress - start_fraction) * (end - start) / (end_fraction - start_fraction)

    return fn


def cyclical_annealing(
    start: float,
    end: float,
    start_fraction: float,
    end_fraction: float,
    n_cycles: int = 4,
    ratio: float = 0.5,
) -> Callable[[float], float]:
    """Cyclical annealing — ``n_cycles`` ramps within [start_fraction, end_fraction]."""

    def fn(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < start_fraction:
            return start
        if progress > end_fraction:
            return end
        active = (progress - start_fraction) / (end_fraction - start_fraction)
        cycle = (active * n_cycles) % 1.0
        if cycle < ratio:
            return start + (end - start) * (cycle / ratio)
        return end

    return fn


DATA_LOSSES = ("df", "phi", "flux", "fluxavg")
INTEGRAL_LOSSES = ("phi_int", "flux_int")
REMOVED_LOSSES = ("phi_cross", "flux_cross")


def _truthy(sched_cfg: Any, key: str) -> bool:
    return bool(sched_cfg) and key in sched_cfg and bool(sched_cfg[key])


class LossConfig:
    """Resolved multi-task loss setup.

    ``weights`` merges ``loss_weights`` and ``extra_loss_weights``; ``active`` are
    the keys whose weight is positive or that carry a schedule (the static term
    set); ``outputs`` are the model outputs (``loss_weights`` keys only). A
    scheduled weight is replaced by ``sched(progress_remaining)`` each step.
    """

    def __init__(
        self, loss_weights: Any, extra_loss_weights: Any = None, loss_scheduler: Any = None
    ):
        lw = {k: float(v or 0.0) for k, v in to_dict(loss_weights).items()}
        elw = {k: float(v or 0.0) for k, v in to_dict(extra_loss_weights).items()}
        loss_scheduler = to_dict(loss_scheduler)
        known = set(DATA_LOSSES) | set(INTEGRAL_LOSSES) | set(REMOVED_LOSSES)
        sched_keys = set(loss_scheduler)
        unknown = sorted((set(lw) | set(elw) | sched_keys) - known)
        if unknown:
            raise ValueError(f"unknown loss keys {unknown}; supported: {sorted(known)}")
        self.weights = {**lw, **elw}
        self.schedulers = {
            k: fn for k, fn in build_scheduler_dict(loss_scheduler).items() if k in self.weights
        }
        self.active = tuple(
            k for k in self.weights if self.weights[k] > 0.0 or k in self.schedulers
        )
        removed = [k for k in self.active if k in REMOVED_LOSSES]
        if removed:
            raise ValueError(f"cross losses {removed} are not supported; set their weight to 0")
        self.outputs = tuple(k for k in lw if lw[k] > 0.0 or _truthy(loss_scheduler, k))
        if len([k for k in self.outputs if k.startswith("flux")]) > 1:
            raise ValueError("cannot predict both flux and fluxavg")
        self.flux_key = next((k for k in self.outputs if k in ("flux", "fluxavg")), None)
        self.integrals = tuple(k for k in self.active if k in INTEGRAL_LOSSES)

    def weights_at(self, step: int, total_steps: int) -> dict[str, jnp.ndarray]:
        """float32 weights of the active terms at training ``step`` of ``total_steps``."""
        progress_remaining = max(0.0, 1.0 - step / max(total_steps, 1))
        out = {k: self.weights[k] for k in self.active}
        for k, fn in self.schedulers.items():
            out[k] = float(fn(progress_remaining))
        return {k: jnp.asarray(v, jnp.float32) for k, v in out.items()}


def compute_multi_task_loss(
    preds: Mapping[str, jnp.ndarray],
    tgts: Mapping[str, jnp.ndarray],
    weights: Mapping[str, jnp.ndarray],
    active: Sequence[str],
    *,
    extra: Optional[Mapping[str, jnp.ndarray]] = None,
    separate_zf_loss: bool = False,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Weighted sum over the static ``active`` terms.

    ``df``/``phi`` use relative-norm MSE (``df`` optionally with the zonal-flow
    MSE split), ``flux``/``fluxavg`` L1; ``extra`` carries precomputed terms such
    as the integral losses. ``weights`` may hold traced scalars.
    """
    losses = {}
    for k in active:
        if extra is not None and k in extra:
            losses[k] = extra[k]
        elif k == "df":
            losses[k] = df_loss(preds[k], tgts[k], separate_zf=separate_zf_loss)
        elif k == "phi":
            losses[k] = relative_norm_mse(preds[k], tgts[k].reshape(preds[k].shape))
        elif k in ("flux", "fluxavg"):
            losses[k] = l1(preds[k], tgts[k])
        else:
            raise KeyError(f"no loss available for active term {k!r}")
    total = sum((weights[k] * losses[k] for k in active), jnp.float32(0.0))
    return total, losses


def build_scheduler_dict(loss_scheduler_cfg: Any) -> dict[str, Callable[[float], float]]:
    """Translate the ``loss_scheduler`` config into a name → fn dict.

    Skips keys whose value is ``None`` / ``{}`` (i.e. constant weight).
    """
    out: dict[str, Callable[[float], float]] = {}
    for key, sp in to_dict(loss_scheduler_cfg).items():
        if not sp:
            continue
        args = [sp.get(k) for k in ("start", "end", "start_fraction", "end_fraction")]
        if sp.get("type", "linear") == "cyclical":
            out[key] = cyclical_annealing(*args, sp.get("n_cycles", 4), sp.get("ratio", 0.5))
        else:
            out[key] = linear_burn_in(*args)
    return out
