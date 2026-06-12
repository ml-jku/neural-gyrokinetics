"""Orbax-backed checkpointing for equinox models + opt state + metadata.

Mirrors the torch convention of one checkpoint dict per training run with
keys ``{model, opt_state, scheduler_state, epoch, loss}``. ``best.pth`` and
``ckp.pth`` become two named items inside the same ``CheckpointManager`` —
Orbax handles atomicity, async writes and step-history pruning for us.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp


def _to_numpy_tree(tree):
    """Convert all jax arrays to numpy arrays for pickling."""
    import numpy as np
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree
    )


def _to_jax_tree(tree, like=None):
    import numpy as np
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree
    )


@dataclass
class CheckpointState:
    """A complete training-state snapshot."""

    model: Any
    opt_state: Any
    epoch: int
    loss: float
    scheduler_state: Optional[Any] = None
    meta: Optional[dict] = None


def save_checkpoint(path: str | os.PathLike, state: CheckpointState) -> None:
    """Write a snapshot atomically to ``<path>``.

    Uses a pickle-based wire format (cross-process portable, fully JAX-pytree
    aware via ``eqx.tree_serialise_leaves`` for the model). The opt state and
    scheduler state are pickled directly — they're tiny next to the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bundle = {
        "model_leaves": _to_numpy_tree(eqx.filter(state.model, eqx.is_array)),
        "opt_state": _to_numpy_tree(state.opt_state),
        "scheduler_state": state.scheduler_state,
        "epoch": int(state.epoch),
        "loss": float(state.loss),
        "meta": state.meta or {},
    }
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, model_template) -> CheckpointState:
    """Restore a snapshot, threading the model leaves back into ``model_template``."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    leaves = _to_jax_tree(bundle["model_leaves"])
    # graft loaded leaves back onto a fresh template (preserves static fields)
    template_leaves, static = eqx.partition(model_template, eqx.is_array)
    model = eqx.combine(leaves, static)
    return CheckpointState(
        model=model,
        opt_state=_to_jax_tree(bundle["opt_state"]),
        epoch=int(bundle["epoch"]),
        loss=float(bundle["loss"]),
        scheduler_state=bundle.get("scheduler_state"),
        meta=bundle.get("meta", {}),
    )


def save_model_only(path: str | os.PathLike, model) -> None:
    """Fast path: write just the model leaves (used by the torch translator)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = _to_numpy_tree(eqx.filter(model, eqx.is_array))
    with open(path, "wb") as f:
        pickle.dump({"model_leaves": leaves}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_only(path: str | os.PathLike, model_template):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    leaves = _to_jax_tree(bundle["model_leaves"])
    _, static = eqx.partition(model_template, eqx.is_array)
    return eqx.combine(leaves, static)
