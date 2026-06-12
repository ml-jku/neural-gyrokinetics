"""GyroSwin evaluator — multi-step autoregressive recon metrics + plots.

Mirrors ``neugk/gyroswin/eval/eval.py``: for each starting sample, predict
``n_eval_steps`` ahead, comparing each step's ``df`` / ``phi`` against the
ground-truth at the same timestep. Logs ``df_x{t}`` / ``phi_x{t}`` keys.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator
from neugk_jax.losses import mse_df


class GyroSwinEvaluator(BaseEvaluator):
    """``n_eval_steps`` autoregressive rollout — df_x{t} / phi_x{t} metrics."""

    def __call__(self, model: Any, *, epoch: int, batch_size: int = 1, **_):
        ds = self.val_ds
        n = len(ds)
        n_eval = int(self.cfg.validation.get("n_eval_steps", 1))

        @eqx.filter_jit
        def fwd(m, x, c):
            return jax.vmap(lambda xi, ci: m(xi, ci))(x, c)

        from neugk_jax.evaluate.plots import generate_val_plots

        running: dict[str, float] = {}
        n_acc = 0.0
        val_plots: dict[str, object] = {}
        plot_drawn = False

        # walk val set in batches of ``batch_size``, leaving room for n_eval lookahead
        for start in range(0, n - batch_size - n_eval + 1, batch_size):
            samples = [ds[start + b] for b in range(batch_size)]
            # initial df + cond + targets at t..t+n_eval
            df0 = jnp.stack([jnp.asarray(s.df) for s in samples])
            cond = jnp.stack([jnp.asarray(s.conditioning) for s in samples]) \
                if getattr(samples[0], "conditioning", None) is not None else None
            df_pred = df0
            phi_tgts, df_tgts = [], []
            for t in range(1, n_eval + 1):
                tgts_t = [ds[start + b + t] for b in range(batch_size)]
                df_tgts.append(jnp.stack([jnp.asarray(s.df) for s in tgts_t]))
                if getattr(tgts_t[0], "phi", None) is not None:
                    phi_tgts.append(jnp.stack([jnp.asarray(s.phi) for s in tgts_t]))
                else:
                    phi_tgts.append(None)

            step_metrics: dict[str, float] = {}
            phi_pred_at_t1 = None
            df_pred_at_t1 = None
            for t in range(1, n_eval + 1):
                preds = fwd(model, df_pred, cond)
                step_metrics[f"df_x{t}"] = float(mse_df(preds["df"], df_tgts[t - 1]))
                if "phi" in preds and phi_tgts[t - 1] is not None:
                    step_metrics[f"phi_x{t}"] = float(mse_df(preds["phi"], phi_tgts[t - 1]))
                if t == 1:
                    df_pred_at_t1 = preds["df"]
                    phi_pred_at_t1 = preds.get("phi")
                df_pred = preds["df"]  # feed prediction back as next-step input

            running, n_acc = self._accumulate(running, step_metrics, n_acc, n_new=batch_size)

            # plot_nd panels on the first batch of the first step
            if not plot_drawn and self.is_rank0:
                try:
                    b = 0
                    rollout = {"df": np.asarray(df_pred_at_t1[b])}
                    gt = {"df": np.asarray(df_tgts[0][b])}
                    if phi_pred_at_t1 is not None and phi_tgts[0] is not None:
                        rollout["phi"] = np.asarray(phi_pred_at_t1[b])
                        gt["phi"] = np.asarray(phi_tgts[0][b])
                    panels = generate_val_plots(
                        rollout=rollout, gt=gt, phase="random draw",
                        ts=np.asarray(samples[b].timestep).reshape(-1),
                    )
                    val_plots.update(panels)
                except Exception as e:
                    print(f"[gyroswin eval] plot skipped: {e}")
                finally:
                    plot_drawn = True

        running, n_acc = self._sync(running, n_acc)
        return self._finalize(running, n_acc), val_plots
