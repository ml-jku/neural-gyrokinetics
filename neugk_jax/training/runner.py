"""Training runner base shared by the AE, flow-matching and GyroSwin workflows.

``BaseRunner`` owns dataset construction, the prefetching batch pipeline, optimizer and
schedule, the jitted train step, the epoch loop with validation cadence and best-model
selection, resume and async checkpointing. Subclasses provide ``setup_data``,
``build_model``, ``loss_fn`` and ``make_evaluator`` (plus optional hooks).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from omegaconf import OmegaConf

from neugk_jax.dataset.factory import build_splits
from neugk_jax.models.utils import trainable_mask
from neugk_jax.training.checkpoint import AsyncCheckpointer, CheckpointState, load_checkpoint
from neugk_jax.training.data import BatchLoader, stack_fields, train_plans
from neugk_jax.training.ddp import (
    DistributedInfo,
    global_batch_size,
    init_distributed,
    local_view,
    replicate,
    shard_batch,
)
from neugk_jax.training.logging import Logger
from neugk_jax.training.schedulers import warmup_cosine
from neugk_jax.utils import count_trace, progress, to_dict


def weight_decay_mask(params, exclude):
    """Pytree of bools, False where the leaf path contains any ``exclude`` substring."""
    if exclude == "all":
        return jax.tree_util.tree_map(lambda _: False, params)
    exclude = [e.lower() for e in exclude or []]

    def keep(path, _):
        name = jax.tree_util.keystr(path).lower()
        return not any(e in name for e in exclude)

    return jax.tree_util.tree_map_with_path(keep, params)


def build_optimizer(schedule, tcfg, model, *, decoupled: bool, b2: float = 0.999, mask=None):
    """Clip + Adam chain over the ``mask`` leaves (default ``trainable_mask``).

    ``decoupled`` selects AdamW, else Adam with coupled L2 decay.
    """
    wd = tcfg.get("weight_decay", 0.0)
    params = eqx.filter(model, trainable_mask(model) if mask is None else mask)
    mask = weight_decay_mask(params, tcfg.get("exclude_from_wd", []))
    clip = (
        optax.clip_by_global_norm(tcfg.get("clip_to", 1.0))
        if tcfg.get("clip_grad", True)
        else optax.identity()
    )
    if wd <= 0:
        return optax.chain(clip, optax.adam(schedule, b2=b2))
    if decoupled:
        return optax.chain(clip, optax.adamw(schedule, b2=b2, weight_decay=wd, mask=mask))
    # coupled l2: wd * p is added to the gradient before the moment estimates
    return optax.chain(
        clip,
        optax.add_decayed_weights(wd, mask=mask),
        optax.scale_by_adam(b2=b2),
        optax.scale_by_learning_rate(schedule),
    )


def train_update(model, opt_state, loss_fn, optimizer, mask, *, has_aux: bool = False):
    """One optimizer step on the leaves ``mask`` marks trainable; buffers stay fixed."""
    params, static = eqx.partition(model, mask)
    out, grads = eqx.filter_value_and_grad(
        lambda p: loss_fn(eqx.combine(p, static)), has_aux=has_aux
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.combine(eqx.apply_updates(params, updates), static), opt_state, out


@dataclass(frozen=True, eq=False)
class StepSpec:
    """Static part of a train step: loss hook, optimizer, trainable mask and post-update hook."""

    name: str
    loss_fn: Callable
    optimizer: Any
    mask: Any
    post_update: Callable


@eqx.filter_jit(donate="all-except-first")
def train_step(inputs, model, opt_state, spec: StepSpec):
    """``inputs = (batch, ctx, key)``; ``ctx`` holds the run-constant device tables (not donated).

    A ``"state"`` entry of the loss aux is not logged but handed to
    ``spec.post_update(model, state, key) -> (model, logs)`` after the optimizer step.
    """
    count_trace(f"train_step:{spec.name}")
    batch, ctx, key = inputs

    def loss(m):
        return spec.loss_fn(m, {**ctx, **batch}, key)

    model, opt_state, (value, aux) = train_update(
        model, opt_state, loss, spec.optimizer, spec.mask, has_aux=True
    )
    aux = dict(aux)
    model, extra = spec.post_update(model, aux.pop("state", None), jr.fold_in(key, 1))
    return model, opt_state, {"total": value, **aux, **extra}


@eqx.filter_jit(donate="all")
def _add_logs(acc, logs):
    return jax.tree_util.tree_map(jnp.add, acc, logs)


def configure_compilation_cache(cfg) -> None:
    path = cfg.get("jax_compilation_cache_dir")
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(path))


class BaseRunner:
    """Epoch loop, validation, best selection, resume and checkpointing around a jitted step.

    Hooks: ``setup_data`` (sets ``train_ds``/``val_ds``), ``build_model(key)``,
    ``loss_fn(model, batch, key) -> (loss, aux)``, ``make_evaluator()``; optional
    ``load_batch(ds, indices, read)``, ``step_context()`` (run-constant device arrays merged
    into the batch), ``step_extras(step)`` (per-step arrays merged into the batch),
    ``trainable_mask(model)`` (the leaves the optimizer updates), ``post_update(model, state,
    key) -> (model, logs)`` (non-gradient buffer updates from the ``"state"`` loss aux),
    ``checkpoint_meta()`` (extra checkpoint metadata). Process 0 writes the resolved config to
    ``<output_path>/config.yaml``.
    """

    # validation metrics that select best.eqx, first present wins (lower is better)
    val_metrics: tuple[str, ...] = ("df", "df_mse")
    batch_fields: tuple[str, ...] = ("df",)
    decoupled_wd: bool = False
    adam_b2: float = 0.999
    dist: DistributedInfo

    def __init__(self, cfg, *, output_path: str | None = None):
        self.cfg = cfg
        configure_compilation_cache(cfg)
        self.dist = init_distributed()
        self.logger = Logger(
            is_rank0=self.dist.is_rank0, config=to_dict(cfg), logging=to_dict(cfg.get("logging"))
        )
        self.output_path = Path(output_path or cfg.get("output_path") or "outputs/run")
        self.tcfg = cfg.training
        self.start_epoch = 0
        self.best_val = math.inf
        self.checkpointer = AsyncCheckpointer()
        self.loader = BatchLoader(
            workers=self.tcfg.get("num_workers", 4), prefetch=self.tcfg.get("prefetch", 2)
        )
        self.setup_data()
        self.model = self.build_model(jr.PRNGKey(self.seed))
        self.setup_optimizer()
        self.ctx = replicate(self.dist, self.step_context())
        self.spec = StepSpec(
            type(self).__name__, self.loss_fn, self.optimizer, self.trainable, self.post_update
        )
        self._maybe_resume()
        self.evaluator = self.make_evaluator()
        self.save_config()

    @property
    def seed(self) -> int:
        return int(self.cfg.get("seed", 0))

    @property
    def vcfg(self) -> dict:
        return to_dict(self.cfg.get("validation"))

    @property
    def eval_batch_size(self) -> int:
        return self.vcfg.get("batch_size") or self.tcfg.batch_size

    def evaluator_kwargs(self) -> dict:
        return dict(
            val_ds=self.val_ds, dist=self.dist, loader=self.loader, batch_size=self.eval_batch_size
        )

    def build_data(self, mode: str, **kwargs) -> None:
        self.train_ds, self.val_ds = build_splits(
            self.cfg.dataset, dist=self.dist, mode=mode, **kwargs
        )

    def save_config(self) -> None:
        if self.dist.is_rank0:
            self.output_path.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, self.output_path / "config.yaml")

    def checkpoint_meta(self) -> dict:
        return {}

    def setup_data(self) -> None:
        raise NotImplementedError

    def build_model(self, key):
        raise NotImplementedError

    def loss_fn(self, model, batch: dict, key) -> tuple[jnp.ndarray, dict]:
        raise NotImplementedError

    def make_evaluator(self):
        return None

    def step_context(self) -> dict:
        return {}

    def trainable_mask(self, model):
        return trainable_mask(model)

    def post_update(self, model, state, key):
        return model, {}

    def step_extras(self, step: int) -> dict:
        return {}

    def load_batch(self, ds, indices, read) -> dict:
        return stack_fields(read(ds, indices), self.batch_fields)

    @property
    def global_batch_size(self) -> int:
        return global_batch_size(self.dist, self.tcfg.batch_size)

    def setup_optimizer(self) -> None:
        tcfg = self.tcfg
        self.steps_per_epoch = max(1, len(self.train_ds) // self.global_batch_size)
        self.total_steps = tcfg.n_epochs * self.steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=tcfg.learning_rate,
            total_steps=self.total_steps,
            steps_per_epoch=self.steps_per_epoch,
            n_epochs=tcfg.n_epochs,
            min_lr=tcfg.get("final_learning_rate", 1e-6),
        )
        self.trainable = self.trainable_mask(self.model)
        self.optimizer = build_optimizer(
            self.schedule,
            tcfg,
            self.model,
            decoupled=self.decoupled_wd,
            b2=self.adam_b2,
            mask=self.trainable,
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.model, self.trainable))
        self.model = replicate(self.dist, self.model)
        self.opt_state = replicate(self.dist, self.opt_state)

    def _maybe_resume(self) -> None:
        ckpt = self.output_path / "ckp.eqx"
        if not ckpt.exists():
            return
        state = load_checkpoint(ckpt, self.model)
        self.model = replicate(self.dist, state.model)
        self.opt_state = replicate(self.dist, state.opt_state)
        self.start_epoch = state.epoch
        self.best_val = float((state.meta or {}).get("best_val", math.inf))
        if self.dist.is_rank0:
            print(f"resumed from epoch {self.start_epoch} (val={self.best_val:.4e})")

    def save_checkpoint(self, epoch: int, val: float, name: str = "ckp.eqx") -> None:
        if not self.dist.is_rank0:
            return
        state = CheckpointState(
            model=local_view(self.dist, self.model),
            opt_state=local_view(self.dist, self.opt_state),
            epoch=epoch,
            loss=val,
            meta={"best_val": self.best_val, **self.checkpoint_meta()},
        )
        self.checkpointer.save(self.output_path / name, state)

    def train_epoch(self, epoch: int, key) -> tuple[dict, dict]:
        perm_key, step_key = jr.split(key)
        perm = np.asarray(jr.permutation(perm_key, len(self.train_ds)))
        plans = train_plans(self.dist, len(self.train_ds), self.tcfg.batch_size, perm)
        batches = self.loader.iterate(
            self.train_ds, plans, self.load_batch, lambda b: shard_batch(self.dist, b)
        )
        show = self.dist.is_rank0 and (self.cfg.get("logging") or {}).get("tqdm", False)
        batches = progress(batches, show, total=len(plans), desc=f"epoch {epoch}")
        acc, waits = None, []
        t_start = t_first = time.perf_counter()
        step0 = (epoch - 1) * self.steps_per_epoch
        for i, (_, batch, wait) in enumerate(batches):
            waits.append(wait)
            batch.pop("mask")
            batch.update(self.step_extras(step0 + i))
            self.model, self.opt_state, logs = train_step(
                (batch, self.ctx, jr.fold_in(step_key, i)), self.model, self.opt_state, self.spec
            )
            acc = logs if acc is None else _add_logs(acc, logs)
            if i == 0:
                jax.block_until_ready(acc)
                t_first = time.perf_counter()
        n = len(waits)
        if acc is None:
            return {}, {}
        sums = {k: float(v) for k, v in jax.device_get(acc).items()}
        t_end = time.perf_counter()
        info = {
            "first_step_ms": (t_first - t_start) * 1e3,
            "step_ms": (t_end - t_first) * 1e3 / max(n - 1, 1),
            "data_ms": float(np.median(waits[1:] or waits)),
        }
        return {k: v / n for k, v in sums.items()}, info

    def evaluate(self, epoch: int) -> tuple[dict, dict]:
        if self.evaluator is None:
            return {}, {}
        return self.evaluator(self.model, epoch=epoch)

    def _val_score(self, val_logs: dict) -> float:
        for k in self.val_metrics:
            if k in val_logs:
                return float(val_logs[k])
        raise KeyError(f"none of {self.val_metrics} in validation metrics {sorted(val_logs)}")

    def __call__(self) -> None:
        base_key = jr.PRNGKey(self.seed)
        val_every = self.vcfg.get("validate_every_n_epochs", 1)
        save_every = self.tcfg.get("save_every_n_epochs", 1)
        n_epochs = self.tcfg.n_epochs
        last_val = math.nan
        try:
            for epoch in range(self.start_epoch + 1, n_epochs + 1):
                t0 = time.perf_counter()
                loss_logs, info = self.train_epoch(epoch, jr.fold_in(base_key, epoch))
                t_train = time.perf_counter() - t0
                validating = epoch % val_every == 0 or epoch == 1 or epoch == n_epochs
                val_logs, val_plots = {}, {}
                if validating:
                    t0 = time.perf_counter()
                    val_logs, val_plots = self.evaluate(epoch)
                    info["eval_s"] = time.perf_counter() - t0
                logs = {f"train/{k}": v for k, v in loss_logs.items()}
                logs["train/lr"] = float(self.schedule(epoch * self.steps_per_epoch))
                logs.update({f"info/{k}": v for k, v in info.items()})
                logs.update({f"val_traj/{k}": v for k, v in val_logs.items()})
                logs.update({"epoch": epoch, "epoch_time_s": t_train})
                self.logger.log(logs, step=epoch, commit=not val_plots)
                if val_plots:
                    self.logger.log(
                        {f"val_plots/{k}": v for k, v in val_plots.items()}, step=epoch, commit=True
                    )
                if self.dist.is_rank0:
                    core = " ".join(f"{k}={v:.4e}" for k, v in loss_logs.items())
                    print(f"epoch {epoch:04d}  {core}  ({t_train:.1f}s)")
                if validating and val_logs:
                    last_val = self._val_score(val_logs)
                    if last_val < self.best_val:
                        self.best_val = last_val
                        self.save_checkpoint(epoch, last_val, "best.eqx")
                if epoch % save_every == 0 or epoch == n_epochs:
                    self.save_checkpoint(epoch, last_val, "ckp.eqx")
        finally:
            self.checkpointer.join()
            self.loader.close()
        self.logger.finish()


def conditioning_slots(dataset_conditions, model_conditions) -> Optional[np.ndarray]:
    # sorted condition names, as the models consume them
    if not model_conditions:
        return None
    return np.asarray(
        [list(dataset_conditions).index(c) for c in sorted(model_conditions)], np.int32
    )
