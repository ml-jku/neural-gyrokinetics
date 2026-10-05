"""PINC-AE PEFT: LoRA adapters on a frozen pretrained conditioned AE, trained on df + physics.

``PINCPEFTRunner`` strictly loads ``ae_checkpoint``, adapts the linears of ``model.peft.lora``
and trains only the adapters on ``model.loss_weights`` (+ ``model.extra_loss_weights``) of
``df`` (MSE), ``phi_int`` / ``flux_int`` (per-snapshot relative L1 of the field solve of the
denormalized prediction against that of the target) and the ``kyspec`` / ``qspec`` spectral
losses.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from neugk_jax.evaluate.base import GeometryCache, take_rows
from neugk_jax.losses import recon_loss
from neugk_jax.models.lora import lora_mask
from neugk_jax.pinc.eval import AEEvaluator, reconstruct
from neugk_jax.pinc.losses import pinc_integrals, pinc_losses, pinc_terms
from neugk_jax.pinc.runner import AERunner, read_loss_weights
from neugk_jax.training.checkpoint import resolve_checkpoint
from neugk_jax.training.ddp import replicate_local

LOSS_KEYS = ("df", "phi_int", "flux_int", "kyspec", "qspec")
# served spectrum whose log1p per-mode std normalizes each spectral loss
SPECTRUM_STATS = {"kyspec": "kyspec", "qspec": "fluxspec"}


def uniform_ds(*datasets) -> float:
    """The parallel grid spacing ``ds`` shared by every trajectory of ``datasets``."""
    vals = [d.get_ds(f) for d in datasets for f in range(len(d.files))]
    if not vals or any(v is None for v in vals):
        raise ValueError("pinc losses need the 'ds' spacing in every trajectory's metadata")
    if not np.allclose(vals, vals[0]):
        raise ValueError(f"pinc losses need one 'ds' across trajectories, got {sorted(set(vals))}")
    return float(vals[0])


class PINCPEFTRunner(AERunner):
    """Adapter fine-tune of a pretrained Swin5DAE against the PINC physics losses."""

    val_metrics = ("phi_int_mse",)
    accepts_vq = True

    def setup_data(self) -> None:
        super().setup_data()
        self.loss_weights = read_loss_weights(self.cfg.model, LOSS_KEYS)
        self.ds_spacing = uniform_ds(self.train_ds, self.val_ds)
        self.mode_std = {
            k: jnp.asarray(self.train_ds.spectral_stats(src)["std"])
            for k, src in SPECTRUM_STATS.items()
            if self.loss_weights.get(k, 0.0) > 0
        }

    def build_model(self, key):
        from neugk_jax.translate import attach_ae_lora, load_or_translate

        if self.cond_slots is None:
            raise ValueError("pinc peft expects a conditioned AE (model.decoder_conditioning)")
        if not self.cfg.get("ae_checkpoint"):
            raise ValueError("pinc peft needs ae_checkpoint (pretrained AE run dir or file)")
        k_model, k_lora = jr.split(key)
        ckpt = str(resolve_checkpoint(self.cfg.ae_checkpoint))
        model = load_or_translate(super().build_model(k_model), ckpt, strict=True)
        model = attach_ae_lora(model, self.cfg.model.peft.lora, key=k_lora)
        if self.dist.is_rank0:
            n = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, lora_mask(model))))
            print(f"lora adapters: {n / 1e6:.3f}M trainable")
        return model

    def trainable_mask(self, model):
        return lora_mask(model)

    def step_context(self) -> dict:
        return {
            "norm": self.train_ds.norm,
            "geom": GeometryCache(self.train_ds).table(),
            "mode_std": self.mode_std,
        }

    def loss_fn(self, model, batch, key):
        x, fids, norm = batch["df"], batch["file_index"], batch["norm"]
        keys = jr.split(key, x.shape[0])
        pred = reconstruct(model, x, batch["conditioning"], keys, inference=False)["df"]
        losses = pinc_losses(
            take_rows(batch["geom"], fids),
            norm.denormalize("df", pred, fids),
            norm.denormalize("df", x, fids),
            ds=self.ds_spacing,
            mode_std=batch["mode_std"],
        )
        losses["df"] = recon_loss(pred, x, "mse")
        return sum(w * losses[k] for k, w in self.loss_weights.items()), losses

    def make_evaluator(self):
        metrics = PINCMetrics(
            self.mode_std,
            ds=self.ds_spacing,
            loss_keys=tuple(k for k in LOSS_KEYS[1:] if k in self.loss_weights),
        )
        return PINCEvaluator(self.cfg, metrics=metrics, **self.evaluator_kwargs())


class PINCMetrics(eqx.Module):
    """Per-sample PINC metrics of a denormalized ``(pred, target)`` df pair.

    ``phi_int_mse``, ``flux_int_l1`` (``|pflux| + |eflux - eflux_gt|``), the ``kyspec`` /
    ``qspec`` L1 and ``<term>_loss`` of the training-loss terms ``loss_keys``.
    """

    mode_std: dict
    ds: float = eqx.field(static=True)
    loss_keys: tuple[str, ...] = eqx.field(static=True)

    @property
    def keys(self) -> tuple[str, ...]:
        base = ("phi_int_mse", "flux_int_l1", "kyspec_l1", "qspec_l1")
        return base + tuple(f"{k}_loss" for k in self.loss_keys)

    def __call__(self, pred_d, tgt_d, geom) -> dict:
        p = pinc_integrals(geom, pred_d, ds=self.ds)
        t = pinc_integrals(geom, tgt_d, ds=self.ds)

        def per_sample(pi, ti):
            return pinc_terms(*jax.tree_util.tree_map(lambda a: a[None], (pi, ti)), self.mode_std)

        terms = jax.vmap(per_sample)(p, t)
        return {
            "phi_int_mse": terms["phi_int_mse"],
            "flux_int_l1": terms["flux_int_l1"],
            "kyspec_l1": jnp.mean(jnp.abs(p["kyspec"] - t["kyspec"]), axis=-1),
            "qspec_l1": jnp.mean(jnp.abs(p["qspec"] - t["qspec"]), axis=-1),
            **{f"{k}_loss": terms[k] for k in self.loss_keys},
        }


class PINCEvaluator(AEEvaluator):
    """AE reconstruction metrics and plots plus the :class:`PINCMetrics` of the reconstruction."""

    def __init__(self, cfg: Any, *, metrics: PINCMetrics, **kw):
        super().__init__(cfg, **kw)
        # the pinc metrics carry the integrals
        self.eval_integrals = False
        self.extra = replicate_local(self.dist, metrics)
        self.metric_keys = ("df_mse", "df_rel_l2", *metrics.keys)
