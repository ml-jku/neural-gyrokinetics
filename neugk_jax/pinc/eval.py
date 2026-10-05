"""AE evaluator: reconstruction MSE and relative L2, optional flux integrals and spectra, cross-section plots.

Metrics: ``df_mse`` (normalized), ``df_rel_l2`` (denormalized, zf recombined); with
``validation.eval_integrals`` also ``phi_int_mse``/``phi_int_rel_l2`` and
``flux_int_mse``/``flux_int_rel_err`` of the integrals of the denormalized reconstruction
against those of the target (cached after the first epoch). ``VQVAEEvaluator`` adds the
``codebook_usage`` (fraction of codes hit) and ``codebook_perplexity`` of the validation codes.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neugk_jax.evaluate.base import BaseEvaluator, accumulate, integrate, recon_metrics, take_rows
from neugk_jax.losses import per_sample_mse, per_sample_rel_l2, rel_err
from neugk_jax.training.data import stack_fields
from neugk_jax.training.ddp import replicate_local
from neugk_jax.utils import traced_jit


def select_conditions(batch: dict, slots) -> dict:
    # dataset conditioning -> the model's condition vector (dropped for an unconditioned model)
    cond = batch.pop("conditioning", None)
    if slots is not None:
        batch["conditioning"] = cond[:, slots]
    return batch


def reconstruct(model, x, cond, keys=None, *, inference: bool) -> dict:
    def one(xi, ci, k):
        return model(*((xi,) if ci is None else (xi, ci)), key=k, inference=inference)

    return jax.vmap(one)(x, cond, keys)


@traced_jit("ae_target_integrals")
def target_integrals(x, fids, norm, geom):
    phi, (_, eflux, _) = integrate(geom, fids, norm.denormalize("df", x, fids))
    return phi, eflux


@traced_jit("ae_eval_step")
def ae_eval_step(model, batch, acc, norm, geom, tgt_int, extra=None):
    """Reconstruct one batch, add its masked metric sums to ``acc``; returns the denormalized pair.

    ``extra(pred_d, tgt_d, geom_rows)`` adds per-sample metrics of the denormalized pair.
    """
    x, fids, mask = batch["df"], batch["file_index"], batch["mask"]
    out = reconstruct(model, x, batch.get("conditioning"), inference=True)
    pred = out.pop("df")
    pred_d, tgt_d = norm.denormalize("df", pred, fids), norm.denormalize("df", x, fids)
    values = recon_metrics(pred, x, pred_d, tgt_d)
    phi = None
    if tgt_int is not None:
        phi_t, eflux_t = tgt_int
        phi, (_, eflux, _) = integrate(geom, fids, pred_d)
        values["phi_int_mse"] = per_sample_mse(phi, phi_t)
        values["phi_int_rel_l2"] = per_sample_rel_l2(phi, phi_t)
        values["flux_int_mse"] = (eflux - eflux_t) ** 2
        values["flux_int_rel_err"] = rel_err(eflux, eflux_t)
    if extra is not None:
        values.update(extra(pred_d, tgt_d, take_rows(geom, fids)))
    return accumulate(acc, values, mask), pred_d, tgt_d, phi, out


class AEEvaluator(BaseEvaluator):
    """Reconstruction metrics over the validation set; ``cond_slots`` selects the model conditions."""

    # per-sample metrics module of the denormalized pair, see ae_eval_step
    extra = None

    def __init__(self, cfg: Any, *, cond_slots=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cond_slots = cond_slots
        self.eval_spectra = self.spectra_available(bool(self.vcfg.get("eval_spectra", False)))
        keys = ["df_mse", "df_rel_l2"]
        if self.eval_integrals:
            keys += ["phi_int_mse", "phi_int_rel_l2", "flux_int_mse", "flux_int_rel_err"]
        self.metric_keys = tuple(keys)
        self._tgt_int: dict[int, tuple] = {}

    def load(self, ds, indices, read):
        fields = ("df", "file_index", "timestep", "conditioning")
        return select_conditions(stack_fields(read(ds, indices), fields), self.cond_slots)

    def observe(self, out: dict, batch: dict) -> None:
        pass

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        model = self.local_model(model)
        geom = self.geometry if self.eval_integrals or self.extra is not None else None
        acc = self.zeros((*self.metric_keys, "_n"))
        plots: dict[str, Any] = {}
        spectra: dict[int, dict] = {}
        for plan, batch, _ in self.loader.iterate(self.ds, self.plans, self.load, self.place):
            tgt_int = None
            if self.eval_integrals:
                if plan.number not in self._tgt_int:
                    self._tgt_int[plan.number] = jax.device_get(
                        target_integrals(batch["df"], batch["file_index"], self.norm, geom)
                    )
                tgt_int = self.place(self._tgt_int[plan.number])
            acc, pred_d, tgt_d, phi, out = ae_eval_step(
                model, batch, acc, self.norm, geom, tgt_int, self.extra
            )
            self.observe(out, batch)
            if self.eval_spectra:
                self.spectra(spectra, pred_d, tgt_d, plan)
            if plan.number == 0 and self.is_rank0:
                plots = self._plots(pred_d, tgt_d, phi, tgt_int, batch)
        metrics = self.finalize(self.reduce(acc), self.metric_keys)
        if self.eval_spectra:
            metrics.update(self.spectral_metrics(spectra))
        return metrics, plots

    def _plots(self, pred_d, tgt_d, phi, tgt_int, batch) -> dict[str, Any]:
        from neugk_jax.evaluate.plots import generate_val_plots

        rollout, gt = {"df": pred_d[0]}, {"df": tgt_d[0]}
        if phi is not None:
            rollout["phi"], gt["phi"] = phi[0], tgt_int[0][0]
        return generate_val_plots(rollout, gt, "random draw", ts=self.plot_time(batch))


@eqx.filter_jit
def code_histogram(hist, indices, mask):
    counts = jax.vmap(lambda i: jnp.bincount(i.reshape(-1), length=hist.shape[0]))(indices)
    return hist + jnp.sum(counts * mask[:, None], axis=0)


def codebook_metrics(hist: np.ndarray) -> dict[str, float]:
    p = hist / max(float(hist.sum()), 1.0)
    nz = p[p > 0]
    perplexity = float(np.exp(-np.sum(nz * np.log(nz))))
    return {"codebook_usage": float(np.mean(hist > 0)), "codebook_perplexity": perplexity}


class VQVAEEvaluator(AEEvaluator):
    """AE metrics plus the codebook usage and perplexity of the validation codes."""

    def observe(self, out: dict, batch: dict) -> None:
        self._hist = code_histogram(self._hist, out["vq_indices"], batch["mask"])

    def __call__(self, model: Any, *, epoch: int) -> tuple[dict[str, float], dict[str, Any]]:
        self._hist = replicate_local(self.dist, jnp.zeros((model.codebook_size,), jnp.float32))
        metrics, plots = super().__call__(model, epoch=epoch)
        hist = self.sum_process_arrays(np.asarray(jax.device_get(self._hist), np.float64))
        return {**metrics, **codebook_metrics(hist)}, plots
