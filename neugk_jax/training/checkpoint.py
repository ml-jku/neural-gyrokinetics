"""Checkpointing for equinox models + opt state + metadata.

One pickle per snapshot (``ckp.eqx`` rolling, ``best.eqx`` best validation) holding
the model array leaves, opt state, epoch, loss and a ``meta`` dict; writes are atomic.
``AsyncCheckpointer`` copies the state to host synchronously and writes it in a
background thread.
"""

from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.utils import atomic_write


def resolve_checkpoint(path) -> Path:
    """Checkpoint file for a run directory (``best.eqx``, else ``best.pth``) or a file path."""
    p = Path(path)
    if p.is_dir():
        for name in ("best.eqx", "best.pth"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no best.eqx / best.pth in {p}")
    return p


def _to_numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree)


def _to_jax_tree(tree):
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
    meta: Optional[dict] = None


def host_bundle(state: CheckpointState) -> dict:
    """Host (numpy) copy of a snapshot, ready to pickle."""
    model_leaves, opt_state = jax.device_get(
        (eqx.filter(state.model, eqx.is_array), state.opt_state)
    )
    return {
        "model_leaves": _to_numpy_tree(model_leaves),
        "opt_state": _to_numpy_tree(opt_state),
        "epoch": int(state.epoch),
        "loss": float(state.loss),
        "meta": state.meta or {},
    }


def write_bundle(path: str | os.PathLike, bundle: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda f: pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL))


def save_checkpoint(path: str | os.PathLike, state: CheckpointState) -> None:
    write_bundle(path, host_bundle(state))


def _graft(saved_leaves, template):
    # the saved array leaves in order, onto the template's array structure
    params, static = eqx.partition(template, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(saved_leaves)
    ref = jax.tree_util.tree_leaves(params)
    if len(leaves) != len(ref) or any(np.shape(a) != np.shape(b) for a, b in zip(leaves, ref)):
        raise ValueError(
            f"checkpoint has {len(leaves)} array leaves that do not match the "
            f"model's {len(ref)}"
        )
    leaves = [jnp.asarray(a, dtype=b.dtype) for a, b in zip(leaves, ref)]
    return eqx.combine(
        jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), leaves), static
    )


def load_checkpoint(path: str | os.PathLike, model_template) -> CheckpointState:
    """Restore a snapshot, threading the model leaves back into ``model_template``."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return CheckpointState(
        model=_graft(bundle["model_leaves"], model_template),
        opt_state=_to_jax_tree(bundle["opt_state"]),
        epoch=int(bundle["epoch"]),
        loss=float(bundle["loss"]),
        meta=bundle.get("meta", {}),
    )


def read_checkpoint_meta(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f).get("meta", {})


class AsyncCheckpointer:
    """One background writer; a save waits for the previous write to finish."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def _write(self, path, bundle):
        try:
            write_bundle(path, bundle)
        except BaseException as e:
            self._error = e

    def save(self, path: str | os.PathLike, state: CheckpointState) -> None:
        self.join()
        bundle = host_bundle(state)
        self._thread = threading.Thread(target=self._write, args=(path, bundle), daemon=False)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise err


def load_meta(path: str | os.PathLike) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f).get("meta") or {}


def save_model_only(path: str | os.PathLike, model) -> None:
    write_bundle(
        path, {"model_leaves": _to_numpy_tree(jax.device_get(eqx.filter(model, eqx.is_array)))}
    )


def load_model_only(path: str | os.PathLike, model_template):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return _graft(bundle["model_leaves"], model_template)
