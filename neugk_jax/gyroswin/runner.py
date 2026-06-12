"""GyroSwin training runner — multi-task MSE on df + phi (+ optional flux).

Mirrors ``neugk/gyroswin/run.py`` minus the upstream extras we're not porting
yet (pushforward unrolls, Muon optimizer, GradientBalancer, baseline models).
"""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm

from neugk_jax.dataset import CycloneDataset, KvikIOBackend, NumpyBackend
from neugk_jax.gyroswin.models import build_gyroswin_from_config
from neugk_jax.training.loss_scheduler import (
    build_scheduler_dict, compute_multi_task_loss,
)
from neugk_jax.training.runner import BaseRunner
from neugk_jax.training.schedulers import warmup_cosine


def _mse(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((a - b) ** 2)


class GyroSwinRunner(BaseRunner):
    """Trains GyroSwinMultitask on df+phi multi-task MSE."""

    def setup_data(self) -> None:
        cfg = self.cfg
        backend = (
            KvikIOBackend(rank=self.dist.process_id)
            if getattr(cfg.dataset, "backend", "kvikio") == "kvikio"
            else NumpyBackend()
        )
        common = dict(
            path=cfg.dataset.path,
            fields_to_load=tuple(cfg.dataset.get("input_fields", ("df", "phi"))),
            conditions=tuple(cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))),
            mode="ae",
            backend=backend,
            separate_zf=cfg.dataset.get("separate_zf", True),
            normalization=cfg.dataset.get("normalization"),
            normalization_scope=cfg.dataset.get("normalization_scope", "dataset"),
            normalization_stats=getattr(cfg.dataset, "normalization_stats", None),
            offset=cfg.dataset.get("offset", 0),
        )
        self.train_ds = CycloneDataset(
            split="train", trajectories=cfg.dataset.training_trajectories,
            cond_filters=cfg.dataset.get("training_cond_filters"), **common,
        )
        self.val_ds = CycloneDataset(
            split="val", trajectories=cfg.dataset.validation_trajectories,
            cond_filters=cfg.dataset.get("eval_cond_filters"), **common,
        )

    def setup_components(self) -> None:
        cfg = self.cfg
        # we route via translate.build_gyroswin_from_config so the config layout
        # is identical to torch — pass our hydra ``cfg`` after dumping it to a YAML-shaped dict
        from omegaconf import OmegaConf
        cfg_d = OmegaConf.to_container(cfg, resolve=True)
        # build expects a {"model": ..., "dataset": ...} layout
        cfg_d.setdefault("dataset", cfg_d.get("dataset", {}))
        cfg_d["dataset"].setdefault("resolution", list(self.train_ds.resolution))
        import yaml
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"model": cfg_d["model"], "dataset": cfg_d["dataset"]}, f)
            tmp_cfg = f.name
        self.model = build_gyroswin_from_config(tmp_cfg, key=jr.PRNGKey(getattr(cfg, "seed", 0)))

        steps_per_epoch = max(1, len(self.train_ds) // cfg.training.batch_size)
        total = cfg.training.n_epochs * steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=cfg.training.learning_rate, total_steps=total,
            steps_per_epoch=steps_per_epoch, n_epochs=cfg.training.n_epochs,
            min_lr=cfg.training.get("final_learning_rate", 1e-6),
        )
        wd = cfg.training.get("weight_decay", 0.0)
        opt = optax.chain(
            optax.clip_by_global_norm(cfg.training.get("clip_to", 1.0))
            if cfg.training.get("clip_grad", True) else optax.identity(),
            optax.adamw(self.schedule, weight_decay=wd) if wd > 0 else optax.adam(self.schedule),
        )
        params, _ = eqx.partition(self.model, eqx.is_array)
        self.optimizer = opt
        self.opt_state = opt.init(params)
        self.steps_per_epoch = steps_per_epoch
        # static weights (sum loss_weights + extra_loss_weights) and progress-based schedulers
        lw = dict(cfg.model.get("loss_weights") or {})
        elw = dict(cfg.model.get("extra_loss_weights") or {})
        self.loss_weights = {k: float(v) for k, v in {**lw, **elw}.items()}
        self.loss_schedulers = build_scheduler_dict(cfg.model.get("loss_scheduler"))
        self.total_steps = total

    def _weights_at(self, step: int) -> dict[str, float]:
        progress_remaining = max(0.0, 1.0 - step / max(self.total_steps, 1))
        out = dict(self.loss_weights)
        for k, fn in self.loss_schedulers.items():
            out[k] = float(fn(progress_remaining))
        return out

    @eqx.filter_jit
    def _train_step(self, model, opt_state, batch_df, batch_phi, batch_cond,
                    batch_geom, batch_flux, w_dict):
        """Per-step training update.

        ``w_dict`` is a dict of float scalars; values may be 0.0 to disable
        a term. Integrals are computed only when ``w_dict[phi_int|flux_int]>0``.
        """
        from neugk_jax.evaluate.integrals import gyaradax_flux_integrals

        def loss_fn(m):
            preds = jax.vmap(lambda x, c: m(x, c))(batch_df, batch_cond)
            tgts = {"df": batch_df, "phi": batch_phi, "flux": batch_flux, "avgflux": batch_flux}
            loss = compute_multi_task_loss(preds, tgts, w_dict)
            # physics integrals — only run gyaradax when any integral weight is on
            need = (w_dict.get("phi_int", 0.0) > 0 or w_dict.get("flux_int", 0.0) > 0
                    or w_dict.get("phi_cross", 0.0) > 0 or w_dict.get("flux_cross", 0.0) > 0)
            if need and batch_geom is not None:
                phi_p, eflux_p = gyaradax_flux_integrals(preds["df"], batch_geom)
                # phi_int — compare integrated phi (spectral) magnitude to the gt phi
                if w_dict.get("phi_int", 0.0) > 0:
                    loss = loss + w_dict["phi_int"] * jnp.mean(jnp.abs(phi_p) ** 2 - jnp.abs(batch_phi) ** 2)
                # flux_int — compare integrated eflux to gt flux scalar
                if w_dict.get("flux_int", 0.0) > 0 and batch_flux is not None:
                    eflux_int = eflux_p.reshape(eflux_p.shape[0], -1).sum(axis=-1)
                    loss = loss + w_dict["flux_int"] * jnp.mean((eflux_int.real - batch_flux) ** 2)
                # cross terms only meaningful when the model itself outputs phi/flux directly
                if w_dict.get("phi_cross", 0.0) > 0 and "phi" in preds:
                    loss = loss + w_dict["phi_cross"] * _mse(preds["phi"], jnp.abs(phi_p))
            return loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params, static = eqx.partition(model, eqx.is_array)
        g_params, _ = eqx.partition(grads, eqx.is_array)
        updates, opt_state = self.optimizer.update(g_params, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return eqx.combine(params, static), opt_state, loss

    def train_epoch(self, epoch: int, key) -> tuple[dict, dict]:
        cfg = self.cfg
        bs = cfg.training.batch_size
        n = len(self.train_ds)
        idx = jr.permutation(key, n)
        starts = list(range(0, n - bs + 1, bs))
        from concurrent.futures import ThreadPoolExecutor
        import time as _time

        # only fetch geometry when the integral losses are actually active
        need_geom = any(self.loss_weights.get(k, 0.0) > 0 or k in self.loss_schedulers
                        for k in ("phi_int", "flux_int", "phi_cross", "flux_cross"))

        def _load(start):
            samples = [self.train_ds[int(idx[i])] for i in range(start, start + bs)]
            df = jnp.stack([jnp.asarray(s.df) for s in samples])
            phi = jnp.stack([jnp.asarray(s.phi) for s in samples]) if getattr(samples[0], "phi", None) is not None else None
            cond = jnp.stack([jnp.asarray(s.conditioning) for s in samples]) if getattr(samples[0], "conditioning", None) is not None else None
            flux = jnp.stack([jnp.asarray(s.flux) for s in samples]) if getattr(samples[0], "flux", None) is not None else None
            geom = None
            if need_geom and hasattr(self.train_ds, "get_batch_geometry"):
                import numpy as _np
                fid = _np.asarray([int(s.file_index) for s in samples])
                g = self.train_ds.get_batch_geometry(fid)
                geom = {k: jnp.asarray(v) for k, v in g.items()}
            return df, phi, cond, geom, flux

        ex = ThreadPoolExecutor(max_workers=1)
        losses, t_data, t_step = [], [], []
        future = ex.submit(_load, starts[0]) if starts else None
        try:
            for i, start in enumerate(starts):
                _t = _time.perf_counter_ns()
                df, phi, cond, geom, flux = future.result()
                t_data.append((_time.perf_counter_ns() - _t) / 1e6)
                if i + 1 < len(starts):
                    future = ex.submit(_load, starts[i + 1])
                _t = _time.perf_counter_ns()
                # per-step scheduled weights (linear/cyclical from cfg.model.loss_scheduler)
                global_step = (epoch - 1) * self.steps_per_epoch + i
                w_dict = self._weights_at(global_step)
                self.model, self.opt_state, loss = self._train_step(
                    self.model, self.opt_state, df, phi, cond, geom, flux, w_dict,
                )
                lf = float(loss)
                t_step.append((_time.perf_counter_ns() - _t) / 1e6)
                losses.append(lf)
        finally:
            ex.shutdown(wait=False)
        m_data = sorted(t_data[1:] or t_data)[len(t_data[1:] or t_data) // 2] if t_data else 0.0
        m_step = sorted(t_step[1:] or t_step)[len(t_step[1:] or t_step) // 2] if t_step else 0.0
        loss_logs = {"total": sum(losses) / max(len(losses), 1)}
        info = {"data_ms": m_data, "step_ms": m_step,
                "first_step_ms": t_step[0] if t_step else 0.0}
        return loss_logs, info

    def evaluate(self, epoch: int):
        from neugk_jax.gyroswin.eval import GyroSwinEvaluator
        ev = GyroSwinEvaluator(self.cfg, val_ds=self.val_ds, is_rank0=self.dist.is_rank0)
        metrics, plots = ev(self.model, epoch=epoch, batch_size=self.cfg.training.batch_size)
        return metrics, plots
