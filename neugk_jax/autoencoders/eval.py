"""AE evaluator: reconstruction MSE + optional integrals via gyaradax.

Mirrors ``neugk/pinc/autoencoders/eval.py:AutoencoderEvaluator`` but
trimmed to the bits the user actually trains (recon metrics + integrals
+ cross-section plots). Linear probing is left as a follow-up.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from neugk_jax.evaluate.base import BaseEvaluator, validation_metrics


class AEEvaluator(BaseEvaluator):
    """Run ``model`` over the val set, return mean recon metrics + plot dict."""

    def __call__(
        self,
        model: Any,
        *,
        epoch: int,
        batch_size: int = 1,
        eval_integrals: bool = False,
        **kwargs,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        ds = self.val_ds
        n = len(ds)

        @eqx.filter_jit
        def fwd(m, x):
            return jax.vmap(lambda xi: m(xi)["df"])(x)

        running: dict[str, float] = {}
        n_acc = 0.0
        # match training-time placement: model replicated across devices → shard the val batch too
        local_dev = jax.local_device_count()
        data_shard = None
        if local_dev > 1:
            mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
            data_shard = NamedSharding(mesh, P("dp"))

        # plot collection — one cross-section panel per epoch (first batch)
        val_plots: dict[str, Any] = {}
        plot_drawn = False

        for start in range(0, n - batch_size + 1, batch_size):
            samples = [ds[i] for i in range(start, start + batch_size)]
            df = jnp.stack([jnp.asarray(s.df) for s in samples])
            if data_shard is not None:
                df = jax.device_put(df, data_shard)
            pred = fwd(model, df)

            geometry = None
            if eval_integrals and hasattr(ds, "get_batch_geometry"):
                fid = np.asarray([int(s.file_index) for s in samples])
                geom = ds.get_batch_geometry(fid)
                geometry = {k: jnp.asarray(v) for k, v in geom.items()}

            metrics, integrated = validation_metrics(
                preds={"df": pred},
                tgts={"df": df},
                eval_integrals=eval_integrals,
                geometry=geometry,
            )
            running, n_acc = self._accumulate(running, metrics, n_acc, n_new=batch_size)

            # one cross-section panel per epoch (first eval batch only): df + integrated phi
            if not plot_drawn and self.is_rank0:
                try:
                    from neugk_jax.evaluate.plots import generate_val_plots
                    b_idx = 0
                    fid_i = int(samples[b_idx].file_index)
                    pred_d = np.asarray(ds.denormalize(fid_i, df=np.asarray(pred[b_idx])))
                    tgt_d = np.asarray(ds.denormalize(fid_i, df=np.asarray(df[b_idx])))
                    rollout = {"df": pred_d}
                    gt = {"df": tgt_d}
                    if integrated is not None and integrated.get("phi") is not None:
                        # phi is the spectral-space (s, k_x, k_y) potential, complex-valued —
                        # plot the magnitude so matplotlib can render it
                        rollout["phi"] = np.abs(np.asarray(integrated["phi"])[b_idx])
                        gt["phi"] = np.abs(np.asarray(integrated["phi_tgt"])[b_idx])
                    panels = generate_val_plots(
                        rollout=rollout, gt=gt, phase="random draw",
                        ts=np.asarray(samples[b_idx].timestep).reshape(-1),
                    )
                    val_plots.update(panels)
                except Exception as e:
                    print(f"[evaluate] cross-section plot skipped: {e}")
                finally:
                    plot_drawn = True

        running, n_acc = self._sync(running, n_acc)
        # rename to torch's canonical keys
        finalized = self._finalize(running, n_acc)
        renamed = {
            "df_mse" if k == "df" else
            "phi_int_mse" if k == "phi_int" else
            "flux_int_mse" if k == "flux_int" else
            "flux_target_mse" if k == "flux" else k: v
            for k, v in finalized.items()
        }
        return renamed, val_plots

