"""GyroSwin evaluator: autoregressive rollout metrics on denormalized data.

Each validation sample is rolled out for up to ``n_eval_steps`` steps, capped per
trajectory; step ``t`` targets are the next-step targets at ``timestep_index + t``.
Logs ``{field}_x{t}`` per step (relative-norm MSE for df/phi, MSE for flux/fluxavg,
optional ``phi_int`` (MSE of the integrated phi) and ``flux_int_rel_err``
(``|eflux - flux| / |flux|`` of the integrated heat flux)), ``df_rel_l2_x{t}``/
``phi_rel_l2_x{t}`` and the step means ``{field}``. Batches keep one shape; rollout steps beyond a trajectory are masked.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator, integrate
from neugk_jax.losses import per_sample_mse, per_sample_rel_l2, per_sample_rel_norm_mse, rel_err
from neugk_jax.training.data import stack_fields
from neugk_jax.training.loss_scheduler import DATA_LOSSES
from neugk_jax.utils import recombine_zf, traced_jit

# outputs the integrated heat flux is checked against
INTEGRAL_OUTPUTS = {"df", "phi", "flux"}
N_EVAL_STEPS = 1


@traced_jit("gyroswin_eval_step")
def gyroswin_eval_step(model, x, cond, tgt, fids, live, t, acc, norm, geom, fields):
    """One rollout step; adds the ``live``-masked metric sums at row ``t`` of ``acc``."""
    preds = jax.vmap(lambda xi, ci: model(xi, ci, inference=True))(x, cond)
    pred_d = {k: norm.denormalize(k, preds[k], fids) for k in fields}
    tgt_d = {k: norm.denormalize(k, tgt[k].reshape(preds[k].shape), fids) for k in fields}
    if "df" in fields:
        pred_d["df"] = recombine_zf(pred_d["df"], axis=1)
        tgt_d["df"] = recombine_zf(tgt_d["df"], axis=1)
    values = {}
    for k in fields:
        if k in ("df", "phi"):
            values[k] = per_sample_rel_norm_mse(pred_d[k], tgt_d[k])
            values[f"{k}_rel_l2"] = per_sample_rel_l2(pred_d[k], tgt_d[k])
        else:
            values[k] = per_sample_mse(pred_d[k], tgt_d[k])
    if geom is not None:
        phi_i, (_, eflux, _) = integrate(geom, fids, pred_d["df"], pred_d["phi"])
        values["phi_int"] = per_sample_mse(phi_i, tgt_d["phi"].reshape(phi_i.shape))
        values["flux_int_rel_err"] = rel_err(eflux, tgt_d["flux"].reshape(-1))
    out = {k: acc[k].at[t].add(jnp.sum(v * live)) for k, v in values.items()}
    out["_n"] = acc["_n"].at[t].add(jnp.sum(live))
    return preds["df"], out, pred_d, tgt_d


class GyroSwinEvaluator(BaseEvaluator):
    """``n_eval_steps`` autoregressive rollout over the (tail-cropped) validation set."""

    def __init__(self, cfg: Any, *, outputs: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(cfg, **kwargs)
        ds = self.ds
        self.n_eval = int(self.vcfg.get("n_eval_steps", N_EVAL_STEPS))
        outputs = tuple(outputs or ("df", "phi"))
        self.fields = tuple(k for k in DATA_LOSSES if k in outputs)
        if self.eval_integrals and set(outputs) != INTEGRAL_OUTPUTS:
            warnings.warn(
                f"validation.eval_integrals needs the outputs {sorted(INTEGRAL_OUTPUTS)}, the "
                f"model predicts {sorted(outputs)}; skipping the integral metrics"
            )
            self.eval_integrals = False
        self.t_slot = ds.conditions.index("timestep") if "timestep" in ds.conditions else None
        names = []
        for k in self.fields:
            names += [k, f"{k}_rel_l2"] if k in ("df", "phi") else [k]
        self.names = tuple(names + (["phi_int", "flux_int_rel_err"] if self.eval_integrals else []))

    def load(self, ds, indices, read):
        return stack_fields(
            read(ds, indices),
            ("df", "conditioning", "file_index", "timestep", *(f"y_{k}" for k in DATA_LOSSES)),
        )

    def _targets(self, fids, t0, t, steps) -> dict:
        tt = t0 + np.minimum(t, np.maximum(steps - 1, 0))
        rows = self.loader.map(lambda ft: self.ds.get_target(*ft), list(zip(fids, tt)))
        return {k: np.stack([np.asarray(r[k]) for r in rows]) for k in self.fields}

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        ds, n_eval = self.ds, self.n_eval
        model = self.local_model(model)
        geom = self.geometry if self.eval_integrals else None
        acc = self.zeros((*self.names, "_n"), (n_eval,))
        plots: dict[str, Any] = {}
        for plan, batch, _ in self.loader.iterate(ds, self.plans, self.load, self.place):
            fids, t0 = plan.fids, plan.t_idx
            steps = np.minimum(n_eval, np.asarray([ds.num_ts(f) for f in fids]) - t0 - 1)
            x, cond = batch["df"], batch.get("conditioning")
            # next-step targets live on CycloneSample as y_<field>
            tgt = {k: batch[f"y_{k}"] for k in self.fields}
            for t in range(int(steps[plan.mask > 0].max())):
                if t > 0:
                    tgt = self.place(self._targets(fids, t0, t, steps))
                if self.t_slot is not None:
                    ts = np.asarray([ds.get_timestep(f, ti + t) for f, ti in zip(fids, t0)])
                    cond = cond.at[:, self.t_slot].set(self.place(ts))
                live = self.place((plan.mask * (steps > t)).astype(np.float32))
                x, acc, pred_d, tgt_d = gyroswin_eval_step(
                    model,
                    x,
                    cond,
                    tgt,
                    batch["file_index"],
                    live,
                    jnp.int32(t),
                    acc,
                    self.norm,
                    geom,
                    self.fields,
                )
                if plan.number == 0 and t == 0 and self.is_rank0:
                    plots = self._plots(pred_d, tgt_d, batch)
        host = jax.device_get(acc)
        sums = self.sum_processes(
            {f"{k}_x{t + 1}": float(host[k][t]) for k in host for t in range(n_eval)}
        )
        metrics = {}
        for t in range(1, n_eval + 1):
            cnt = sums[f"_n_x{t}"]
            if cnt > 0:
                metrics.update({f"{k}_x{t}": sums[f"{k}_x{t}"] / cnt for k in self.names})
        for k in self.names:
            per_step = [
                metrics[f"{k}_x{t}"] for t in range(1, n_eval + 1) if f"{k}_x{t}" in metrics
            ]
            if per_step:
                metrics[k] = float(np.mean(per_step))
        return metrics, plots

    def _plots(self, pred_d, tgt_d, batch) -> dict[str, Any]:
        from neugk_jax.evaluate.plots import generate_val_plots

        roll = {k: pred_d[k][0] for k in ("df", "phi") if k in pred_d}
        gt = {k: tgt_d[k][0] for k in ("df", "phi") if k in tgt_d}
        return generate_val_plots(roll, gt, "random draw", ts=self.plot_time(batch))
