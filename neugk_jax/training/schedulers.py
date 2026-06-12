"""Optax learning-rate schedules used by the training runners.

Matches ``BaseRunner.setup_scheduler`` from the upstream torch repo:

* warmup steps = ``total_steps // 6`` for long runs (``n_epochs > 150``)
  or ``max(total_steps // 10, 10 * steps_per_epoch)`` otherwise
* cosine decay from peak LR to ``min_lr`` over the remaining steps

The user can override individual knobs through the config.
"""

from __future__ import annotations

import optax


def warmup_cosine(
    *,
    peak_lr: float,
    total_steps: int,
    steps_per_epoch: int,
    n_epochs: int,
    min_lr: float = 1e-6,
) -> optax.Schedule:
    if n_epochs > 150:
        n_warmup = max(1, total_steps // 6)
    else:
        n_warmup = max(total_steps // 10, 10 * steps_per_epoch)
    n_warmup = min(n_warmup, max(1, total_steps - 1))
    # decay_steps is the TOTAL step count in optax; cosine phase length = decay_steps - warmup_steps
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=n_warmup,
        decay_steps=max(n_warmup + 1, total_steps),
        end_value=min_lr,
    )
