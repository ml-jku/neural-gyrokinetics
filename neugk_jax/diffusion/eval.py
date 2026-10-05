"""Diffusion evaluator: sample latents, decode, integrate, per-trajectory flux RMSE.

* draw ``eval_n_samples`` flow-matching samples per validation condition
* decode and denormalize on device, integrate the heat flux of every sample
* compare against the df-mode target snapshot (``df_mse``, ``df_rel_l2``)
* aggregate the sampled fluxes per ``iteration_<id>`` trajectory (mean ± std) against the
  trajectory's ``avg_flux``: ``avg_flux_rmse``, ``avg_flux_rel_err`` and per-trajectory values
* emit the cross-section ``df`` plot and the ``avg_flux_UQ`` scatter
"""

from __future__ import annotations

import re
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.diffusion.flow_matching import dit_flow_loss, euler_sample
from neugk_jax.evaluate.base import (
    BaseEvaluator,
    accumulate,
    integrate,
    recon_metrics,
    validation_cfg,
)
from neugk_jax.training.data import stack_fields
from neugk_jax.training.ddp import replicate_local
from neugk_jax.utils import traced_jit

_TRAJ_RE = re.compile(r"iteration_\d+")


def _traj_id(path: str) -> Optional[str]:
    m = _TRAJ_RE.search(path)
    return m.group(0) if m else None


def _sample_decode(dit, ae, key, cond, n, steps, latent_scale):
    z = euler_sample(
        lambda x, t, c=None: dit(x, t, c),
        key=key,
        shape=(n, *dit.latent_shape),
        cond=cond,
        steps=steps,
        latent_scale=latent_scale,
    )
    return jax.vmap(lambda zi: ae.decode(zi)["df"])(z)


@traced_jit("diffusion_eval_step")
def diffusion_eval_step(dit, ae, key, batch, acc, norm, geom, steps: int, latent_scale: float):
    x, fids, mask = batch["df"], batch["file_index"], batch["mask"]
    pred = _sample_decode(dit, ae, key, batch.get("cond"), x.shape[0], steps, latent_scale)
    pred_d, tgt_d = norm.denormalize("df", pred, fids), norm.denormalize("df", x, fids)
    eflux = None
    if geom is not None:
        _, (_, eflux, _) = integrate(geom, fids, pred_d)
    return accumulate(acc, recon_metrics(pred, x, pred_d, tgt_d), mask), eflux, pred_d, tgt_d


@traced_jit("fm_val_step")
def fm_val_step(model, tables, batch, acc, key, latent_scale: float, use_ot: bool):
    idx, mask, cond = batch["idx"], batch["mask"], tables["cond"]
    loss = dit_flow_loss(
        model,
        tables["latents"][idx],
        None if cond is None else cond[idx],
        key,
        latent_scale=latent_scale,
        use_ot=use_ot,
        train=False,
        mask=mask,
    )
    n = jnp.sum(mask)
    return {"fm_loss": acc["fm_loss"] + loss * n, "_n": acc["_n"] + n}


class FlowLossEvaluator(BaseEvaluator):
    """Flow-matching loss over the latent validation set with a fixed noise key.

    ``tables`` holds the device ``latents`` and ``cond`` of ``val_ds`` in flat-index order;
    batches carry only their indices, so nothing is read from disk.
    """

    def __init__(
        self, cfg: Any, *, tables: dict, latent_scale: float, use_ot: bool, seed: int, **kw
    ):
        super().__init__(cfg, **kw)
        self.tables = replicate_local(self.dist, tables)
        self.latent_scale = float(latent_scale)
        self.use_ot = use_ot
        self.key = jr.fold_in(jr.PRNGKey(seed), len(self.ds))

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        model = self.local_model(model)
        acc = self.zeros(("fm_loss", "_n"))
        for plan in self.plans:
            batch = self.place({"idx": plan.indices.astype(np.int32), "mask": plan.mask})
            key = jr.fold_in(self.key, plan.number)
            acc = fm_val_step(model, self.tables, batch, acc, key, self.latent_scale, self.use_ot)
        return self.finalize(self.reduce(acc), ("fm_loss",)), {}


class DiffusionEvaluator(BaseEvaluator):
    """Sampling-based evaluator with per-trajectory flux UQ.

    ``val_ds`` serves df targets (mode "ae"); ``cond_slots`` selects the DiT conditioning
    from the dataset's condition vector. ``steps``, ``n_samples``, ``stride`` (every
    ``stride``-th validation sample is evaluated) and ``max_batches`` default to
    ``validation.eval_sample_steps`` / ``eval_n_samples`` / ``eval_stride`` /
    ``eval_max_batches``.
    """

    integrals_default = True

    def __init__(
        self,
        cfg: Any,
        *,
        val_ds: Any,
        autoencoder: Any,
        latent_scale: float,
        cond_slots: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
        n_samples: Optional[int] = None,
        stride: Optional[int] = None,
        max_batches: Optional[int] = None,
        **kwargs,
    ):
        vcfg = validation_cfg(cfg)
        stride = int(stride or vcfg.get("eval_stride", 1))
        super().__init__(
            cfg,
            val_ds=val_ds,
            indices=range(0, len(val_ds), stride),
            max_batches=max_batches if max_batches is not None else vcfg.get("eval_max_batches"),
            **kwargs,
        )
        self.ae = replicate_local(self.dist, autoencoder)
        self.latent_scale = float(latent_scale)
        self.cond_slots = cond_slots
        self.steps = int(steps or self.vcfg.get("eval_sample_steps", 50))
        self.n_samples = int(n_samples or self.vcfg.get("eval_n_samples", 1))
        self.eval_spectra = self.spectra_available(bool(self.vcfg.get("eval_spectra", False)))
        self.metric_keys = ("df_mse", "df_rel_l2")
        self.traj_ids = [_traj_id(f) for f in val_ds.files]

    def load(self, ds, indices, read):
        batch = stack_fields(read(ds, indices), ("df", "file_index", "timestep", "conditioning"))
        cond = batch.pop("conditioning", None)
        if self.cond_slots is not None:
            batch["cond"] = np.asarray(cond)[:, self.cond_slots]
        return batch

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        from neugk_jax.evaluate.plots import generate_val_plots

        model = self.local_model(model)
        geom = self.geometry if self.eval_integrals else None
        acc = self.zeros((*self.metric_keys, "_n"))
        key = jr.PRNGKey(epoch)
        fluxes, plots, spectra = [], {}, {}
        for plan, batch, _ in self.loader.iterate(self.ds, self.plans, self.load, self.place):
            for s in range(self.n_samples):
                k = jr.fold_in(jr.fold_in(key, plan.number), s)
                acc, eflux, pred_d, tgt_d = diffusion_eval_step(
                    model, self.ae, k, batch, acc, self.norm, geom, self.steps, self.latent_scale
                )
                if eflux is not None:
                    fluxes.append((plan, eflux))
                if self.eval_spectra:
                    self.spectra(spectra, pred_d, tgt_d, plan)
            if plan.number == 0 and self.is_rank0:
                gt = {"df": tgt_d[0]}
                plots.update(
                    generate_val_plots(
                        {"df": pred_d[0]}, gt, "val sample", ts=self.plot_time(batch)
                    )
                )
        metrics = self.finalize(self.reduce(acc), self.metric_keys)
        if self.eval_spectra:
            metrics.update(self.spectral_metrics(spectra))
        if self.eval_integrals:
            metrics.update(self._flux_metrics(fluxes, plots))
        return metrics, plots

    def _flux_metrics(self, fluxes: list, plots: dict) -> dict:
        from neugk_jax.evaluate.plots import avg_flux_confidence

        # per trajectory: sum, square sum and count of the sampled fluxes of every process
        traj = np.zeros((len(self.ds.files), 3), np.float64)
        for (plan, _), e in zip(fluxes, jax.device_get([e for _, e in fluxes])):
            valid = plan.mask > 0
            e = np.asarray(e, np.float64)[valid]
            np.add.at(traj, plan.fids[valid], np.stack([e, e**2, np.ones_like(e)], -1))
        traj = self.sum_process_arrays(traj)
        fids = np.nonzero(traj[:, 2] > 0)[0]
        if not len(fids):
            return {}
        flux_sum, flux_sq, n = traj[fids].T
        mean = flux_sum / n
        std = np.sqrt(np.maximum(flux_sq / n - mean**2, 0))
        tgt = np.asarray([self.ds.get_avg_flux(f) for f in fids])
        names = [self.traj_ids[f] or str(f) for f in fids]
        out = {
            "avg_flux_rmse": float(np.sqrt(np.mean((mean - tgt) ** 2))),
            "avg_flux_rel_err": float(np.mean(np.abs(mean - tgt) / np.maximum(np.abs(tgt), 1e-9))),
        }
        for t, pm, ps, gt in zip(names, mean, std, tgt):
            out[f"avg_flux_pred/{t}"], out[f"avg_flux_std/{t}"] = float(pm), float(ps)
            out[f"avg_flux_gt/{t}"] = float(gt)
        if len(tgt) > 2:
            var = float(np.var(mean))
            out["avg_flux_corr"] = float(np.corrcoef(mean, tgt)[0, 1])
            out["avg_flux_slope"] = (
                float(np.cov(mean, tgt)[0, 1] / var) if var > 0 else float("nan")
            )
        if self.is_rank0:
            plots["avg_flux_UQ"] = avg_flux_confidence(mean, std, tgt, names)
        return out
