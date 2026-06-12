"""Diffusion evaluator: sample → integrals → per-trajectory flux RMSE.

Mirrors ``neugk/diffusion/eval.py:DiffusionEvaluator`` in flow-matching
terms:

* draw ``n_samples`` per validation example (stochastic eval)
* integrate via gyaradax to get ``eflux`` from the decoded df
* aggregate predicted fluxes per ``iteration_<id>`` trajectory across all
  samples to get a mean ± std
* compare to the trajectory's ground-truth ``avg_flux`` to report
  ``avg_flux_rmse``
* emit the cross-section ``df`` / ``phi`` plots and the ``avg_flux_UQ``
  scatter for wandb logging
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator, validation_metrics
from neugk_jax.evaluate.plots import avg_flux_confidence, generate_val_plots


_TRAJ_RE = re.compile(r"iteration_\d+")


def _traj_id(path: str) -> Optional[str]:
    m = _TRAJ_RE.search(path)
    return m.group(0) if m else None


class DiffusionEvaluator(BaseEvaluator):
    """Sampling-based evaluator with per-trajectory flux UQ."""

    def __init__(
        self,
        cfg: Any,
        *,
        val_ds: Any,
        autoencoder: Any,
        sample_fn: Callable,
        is_rank0: bool = True,
    ):
        super().__init__(cfg, val_ds=val_ds, is_rank0=is_rank0)
        self.autoencoder = autoencoder
        self.sample_fn = sample_fn

    def __call__(
        self,
        model: Any,
        *,
        epoch: int,
        batch_size: int = 1,
        n_steps: int = 50,
        n_samples_per_traj: int = 1,
        eval_integrals: bool = True,
        max_batches: Optional[int] = None,
        val_subsample: int = 1,
        **kwargs,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run the FM sampler over the val set linearly — mirrors upstream
        ``neugk/diffusion/eval.py:DiffusionEvaluator``.

        Iterates the dataset linearly with stride ``val_subsample`` (the
        upstream ``cfg.dataset.val_subsample``, default 1; the paper uses
        10). Each batch produces one diffusion sample → eflux. After the
        loop, predicted instantaneous fluxes are grouped by ``iteration_N``
        trajectory id, averaged, and compared to ``tgt_avg_flux`` per
        trajectory.
        """
        ds = self.val_ds
        n = len(ds)
        # build the strided index list: [0, stride, 2*stride, ...] within bounds
        if val_subsample > 1:
            indices = list(range(0, n, val_subsample))
        else:
            indices = list(range(n))
        n_iter = (len(indices) + batch_size - 1) // batch_size
        if max_batches is not None:
            n_iter = min(n_iter, max_batches)

        running: dict[str, float] = {}
        n_acc = 0.0
        per_traj_pred = defaultdict(list)
        per_traj_tgt: dict[str, float] = {}
        val_plots: dict[str, Any] = {}
        key = jr.PRNGKey(epoch)

        import time
        _t_start = time.time()
        for batch_idx in range(n_iter):
            start = batch_idx * batch_size
            sel = indices[start:start + batch_size]
            # pad last batch by repeating the final index to keep the jit'd batch shape
            while len(sel) < batch_size:
                sel.append(sel[-1])
            samples = [ds[i] for i in sel]
            cond = (
                jnp.stack([jnp.asarray(s.conditioning) for s in samples])
                if samples[0].conditioning is not None
                else None
            )
            df_tgt = jnp.stack([jnp.asarray(s.df) for s in samples])
            tgt_avg_flux = np.asarray([float(s.avg_flux) for s in samples])
            file_idx = np.asarray([int(s.file_index) for s in samples])
            traj_ids = [_traj_id(ds.files[fi]) for fi in file_idx]

            # multiple stochastic samples per condition
            for _ in range(n_samples_per_traj):
                step_key, key = jr.split(key)
                out = self.sample_fn(key=step_key, batch=batch_size, cond=cond, steps=n_steps)
                df_pred = out["df"] if isinstance(out, dict) else out

                metrics, _ = validation_metrics(
                    preds={"df": df_pred},
                    tgts={"df": df_tgt, "flux": jnp.asarray(tgt_avg_flux)},
                    eval_integrals=False,  # use upstream torch FluxIntegral below
                    geometry=None,
                )
                running, n_acc = self._accumulate(running, metrics, n_acc, n_new=batch_size)

                # route through gyaradax (~300× faster than torch on CPU); parseval corrected; denorm before integral
                if eval_integrals and hasattr(ds, "get_batch_geometry"):
                    try:
                        from neugk_jax.evaluate.integrals import gyaradax_flux_integrals
                        df_pred_np = np.asarray(df_pred)
                        if ds.normalization is not None and hasattr(ds, "denormalize"):
                            df_pred_np = np.stack([
                                np.asarray(ds.denormalize(int(file_idx[b]),
                                                          df=df_pred_np[b]))
                                for b in range(batch_size)
                            ])
                        # use per-file geometry; whole batch from same trajectory avoids per-sample broadcast
                        geom = ds.get_batch_geometry(file_idx)
                        unique_fids = np.unique(file_idx)
                        if len(unique_fids) == 1:
                            # fast path: whole batch from one trajectory
                            geom_one = {k: np.asarray(v[0]) for k, v in geom.items()}
                            _, eflux_b = gyaradax_flux_integrals(df_pred_np, geom_one)
                            eflux = np.asarray(eflux_b).reshape(batch_size, -1).sum(axis=-1)
                        else:
                            # mixed-trajectory batch (boundary) — split + integrate
                            eflux = np.zeros((batch_size,), dtype=np.float64)
                            for fi in unique_fids:
                                mask = (file_idx == fi)
                                idx_b = np.where(mask)[0]
                                df_sub = df_pred_np[idx_b]
                                geom_sub = {k: np.asarray(v[idx_b[0]]) for k, v in geom.items()}
                                _, e_sub = gyaradax_flux_integrals(df_sub, geom_sub)
                                e_sub = np.asarray(e_sub).reshape(len(idx_b), -1).sum(axis=-1)
                                eflux[idx_b] = e_sub
                        for b, tid in enumerate(traj_ids):
                            if tid is None:
                                continue
                            per_traj_pred[tid].append(float(eflux[b]))
                            per_traj_tgt[tid] = float(tgt_avg_flux[b])
                    except Exception as e:
                        if batch_idx == 0:
                            print(f"[evaluate] gyaradax flux-integral failed: {e}")

            if self.is_rank0 and (batch_idx + 1) % 25 == 0:
                _el = time.time() - _t_start
                _it_s = (batch_idx + 1) / max(_el, 1e-6)
                print(f"  [eval] {batch_idx + 1}/{n_iter} batches "
                      f"({_it_s:.2f} batch/s, {_el:.1f}s elapsed)", flush=True)

            # emit the per-batch cross-section plot once for context
            if batch_idx == 0 and self.is_rank0:
                try:
                    if hasattr(ds, "denormalize") and ds.normalization is not None:
                        df_pred_d = np.asarray(ds.denormalize(int(file_idx[0]), df=np.asarray(df_pred)))
                        df_tgt_d = np.asarray(ds.denormalize(int(file_idx[0]), df=np.asarray(df_tgt)))
                    else:
                        df_pred_d = np.asarray(df_pred)
                        df_tgt_d = np.asarray(df_tgt)
                    ts_arr = np.asarray([float(samples[0].timestep)])
                    val_plots.update(
                        generate_val_plots(
                            rollout={"df": df_pred_d[0]},
                            gt={"df": df_tgt_d[0]},
                            phase="val sample",
                            ts=ts_arr,
                            to_wandb=True,
                        )
                    )
                except Exception as e:
                    print(f"[eval] cross-section plots skipped: {e}")

        running, n_acc = self._sync(running, n_acc)
        metrics = self._finalize(running, n_acc)

        # per-trajectory flux RMSE + UQ scatter
        if per_traj_pred and self.is_rank0:
            traj_ids_sorted = sorted(per_traj_pred.keys())
            pred_means = np.array([np.mean(per_traj_pred[t]) for t in traj_ids_sorted])
            pred_stds = np.array([np.std(per_traj_pred[t]) for t in traj_ids_sorted])
            tgt_vals = np.array([per_traj_tgt[t] for t in traj_ids_sorted])
            metrics["avg_flux_rmse"] = float(
                np.sqrt(np.mean((pred_means - tgt_vals) ** 2))
            )
            val_plots["avg_flux_UQ"] = avg_flux_confidence(
                pred_means, pred_stds, tgt_vals, traj_ids_sorted, to_wandb=True,
            )

        return metrics, val_plots
