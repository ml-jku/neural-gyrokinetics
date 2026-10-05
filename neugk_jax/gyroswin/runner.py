"""GyroSwin training runner: next-step multi-task training on df, phi and flux targets."""

from __future__ import annotations

import jax
import jax.random as jr

from neugk_jax.evaluate.base import GeometryCache
from neugk_jax.gyroswin.eval import N_EVAL_STEPS
from neugk_jax.losses import integral_losses
from neugk_jax.models.build import build_gyroswin
from neugk_jax.training.data import stack_fields
from neugk_jax.training.loss_scheduler import DATA_LOSSES, LossConfig, compute_multi_task_loss
from neugk_jax.training.runner import BaseRunner


class GyroSwinRunner(BaseRunner):
    """Trains GyroSwinMultitask to predict the state at ``t + 1`` from ``t``."""

    adam_b2 = 0.95

    def setup_data(self) -> None:
        cfg = self.cfg
        m = cfg.model
        self.loss_cfg = LossConfig(
            m.get("loss_weights"), m.get("extra_loss_weights"), m.get("loss_scheduler")
        )
        fields = set(cfg.dataset.get("input_fields", ("df",)))
        fields |= {k for k in self.loss_cfg.outputs if k in ("df", "phi")}
        if self.loss_cfg.integrals:
            fields |= {"df", "phi"}
        # the val split ends n_eval_steps frames early; those frames are rollout targets
        tail = int(self.vcfg.get("n_eval_steps", N_EVAL_STEPS))
        self.build_data(
            "next",
            fields=tuple(sorted(fields)),
            conditions=cfg.model.get("conditioning"),
            val_overrides={"tail_offset": tail},
        )
        self.separate_zf_loss = bool(m.get("extra_zf_loss", False) and self.train_ds.separate_zf)
        self.geometry = GeometryCache(self.train_ds)

    def build_model(self, key):
        return build_gyroswin(self.cfg, self.train_ds, key=key)

    def step_context(self) -> dict:
        if not self.loss_cfg.integrals:
            return {}
        return {"norm": self.train_ds.norm}

    def step_extras(self, step: int) -> dict:
        return {"weights": self.loss_cfg.weights_at(step, self.total_steps)}

    def load_batch(self, ds, indices, read) -> dict:
        batch = stack_fields(
            read(ds, indices),
            ("df", "conditioning", "file_index", *(f"y_{k}" for k in DATA_LOSSES)),
        )
        if self.loss_cfg.integrals:
            batch["geom"] = self.geometry.stack(batch["file_index"])
        return batch

    def loss_fn(self, model, batch, key):
        loss_cfg = self.loss_cfg
        x, cond = batch["df"], batch.get("conditioning")
        keys = jr.split(key, x.shape[0])
        preds = jax.vmap(lambda xi, ci, k: model(xi, ci, key=k, inference=False))(x, cond, keys)
        # next-step targets live on CycloneSample as y_<field> (y_df, y_phi, y_flux, y_fluxavg)
        tgts = {k: batch.get(f"y_{k}") for k in DATA_LOSSES}
        ints = None
        if loss_cfg.integrals:
            norm, fids = batch["norm"].denormalize, batch["file_index"]
            ints = integral_losses(
                batch["geom"],
                norm("df", preds["df"], fids),
                norm("phi", preds["phi"], fids) if "phi" in preds else None,
                norm("phi", tgts["phi"], fids),
                norm("flux", tgts["flux"], fids),
            )
        return compute_multi_task_loss(
            preds,
            tgts,
            batch["weights"],
            loss_cfg.active,
            extra=ints,
            separate_zf_loss=self.separate_zf_loss,
        )

    def make_evaluator(self):
        from neugk_jax.gyroswin.eval import GyroSwinEvaluator

        return GyroSwinEvaluator(self.cfg, outputs=self.loss_cfg.outputs, **self.evaluator_kwargs())
