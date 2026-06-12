"""Progress-based loss-weight schedules + multi-task loss builder.

Port of ``neugk/utils.py``'s scheduler helpers + a minimal
``compute_multi_task_loss`` that handles the four physical outputs the
gyroswin training uses: ``df``, ``phi``, ``flux``, ``avgflux``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import jax.numpy as jnp


def linear_burn_in(start: float, end: float, start_fraction: float, end_fraction: float) -> Callable[[float], float]:
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
    start: float, end: float, start_fraction: float, end_fraction: float,
    n_cycles: int = 4, ratio: float = 0.5,
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


_SUPPORTED_LOSSES = ("df", "phi", "flux", "avgflux")


def compute_multi_task_loss(
    preds: Mapping[str, jnp.ndarray],
    tgts: Mapping[str, jnp.ndarray],
    weights: Mapping[str, float],
) -> jnp.ndarray:
    """Weighted MSE across the supported gyroswin outputs.

    Recognised keys: ``df`` (5D), ``phi`` (3D), ``flux`` (scalar per sample),
    ``avgflux`` (scalar per sample). Any other key in ``weights`` is ignored.
    Terms with zero weight or with a missing pred/tgt are skipped.
    """
    loss = jnp.float32(0.0)
    for k in _SUPPORTED_LOSSES:
        w = float(weights.get(k, 0.0) or 0.0)
        if w == 0.0 or k not in preds or k not in tgts or preds[k] is None or tgts[k] is None:
            continue
        loss = loss + w * jnp.mean((preds[k] - tgts[k]) ** 2)
    return loss


def build_scheduler_dict(loss_scheduler_cfg: Any) -> dict[str, Callable[[float], float]]:
    """Translate the upstream ``loss_scheduler`` config into a name → fn dict.

    Skips keys whose value is ``None`` / ``{}`` (i.e. constant weight).
    """
    out: dict[str, Callable[[float], float]] = {}
    if not loss_scheduler_cfg:
        return out
    for key in loss_scheduler_cfg:
        sp = loss_scheduler_cfg[key]
        if not sp:
            continue
        kind = sp.get("type", "linear") if hasattr(sp, "get") else getattr(sp, "type", "linear")
        get = (lambda obj, k, d=None: obj.get(k, d)) if hasattr(sp, "get") else (lambda obj, k, d=None: getattr(obj, k, d))
        if kind == "cyclical":
            out[key] = cyclical_annealing(
                start=get(sp, "start"), end=get(sp, "end"),
                start_fraction=get(sp, "start_fraction"),
                end_fraction=get(sp, "end_fraction"),
                n_cycles=get(sp, "n_cycles", 4),
                ratio=get(sp, "ratio", 0.5),
            )
        else:
            out[key] = linear_burn_in(
                start=get(sp, "start"), end=get(sp, "end"),
                start_fraction=get(sp, "start_fraction"),
                end_fraction=get(sp, "end_fraction"),
            )
    return out
