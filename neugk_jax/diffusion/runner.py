"""Latent flow-matching training pipeline.

Loads a translated/trained AE (frozen), precomputes latents over the
training set, then trains a DiT on Gaussian → latent flow matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from neugk_jax.autoencoders.swin5d_ae import Swin5DAE
from neugk_jax.dataset import CycloneDataset, NumpyBackend, precompute_latents
from neugk_jax.diffusion.flow_matching import (
    euler_sample,
    fm_forward_loss,
)
from neugk_jax.diffusion.dit import DiT
from neugk_jax.training.checkpoint import load_model_only
from neugk_jax.training.runner import BaseRunner
from neugk_jax.training.schedulers import warmup_cosine


class FlowMatchingRunner(BaseRunner):
    """Trains a DiT to model the latent distribution via flow matching."""

    def setup_data(self) -> None:
        cfg = self.cfg
        # conditioning is gyroswin-specific; latent diffusion stays unconditional
        if cfg.model.get("conditioning") not in (None, [], ()):
            raise ValueError(
                "`model.conditioning` is set but the diffusion workflow does not accept "
                "scalar conditioning at the model level. Drop it from the config or "
                "switch to workflow=gyroswin."
            )
        ae_path = cfg.ae_checkpoint
        if ae_path is None:
            raise ValueError("diffusion workflow requires ae_checkpoint")
        # build the AE template + load translated weights
        from scripts.translate_ckpt import build_ae_from_config
        ae_cfg = Path(ae_path).parent / "config.yaml"
        ae_template = build_ae_from_config(str(ae_cfg), key=jr.PRNGKey(0))
        self.ae = load_model_only(ae_path, ae_template)

        common = dict(
            path=cfg.dataset.path,
            fields_to_load=tuple(cfg.dataset.get("input_fields", ("df",))),
            conditions=tuple(cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))),
            mode="ae",
            backend=NumpyBackend(),
            separate_zf=cfg.dataset.get("separate_zf", False),
            normalization=cfg.dataset.get("normalization"),
            normalization_scope=cfg.dataset.get("normalization_scope", "dataset"),
            normalization_stats=cfg.dataset.get("normalization_stats"),
            offset=cfg.dataset.get("offset", 0),
        )
        self.train_ds = CycloneDataset(
            split="train",
            trajectories=cfg.dataset.training_trajectories,
            **common,
        )
        self.val_ds = CycloneDataset(
            split="val",
            trajectories=cfg.dataset.validation_trajectories,
            **common,
        )

        # encode every sample once so training is just MSE on cached latents
        def encode_fn(df_batch, cond_batch):
            return jax.vmap(lambda x: self.ae.encode(x)[0])(df_batch)

        ae_tag = Path(ae_path).stem
        precompute_latents(self.train_ds, encode_fn=encode_fn, ae_tag=ae_tag,
                           batch_size=cfg.training.get("precompute_batch", 2))
        precompute_latents(self.val_ds, encode_fn=encode_fn, ae_tag=ae_tag,
                           batch_size=cfg.training.get("precompute_batch", 2))

        # 1 / sqrt(mean variance) — matches the upstream latent_scale
        var = self.train_ds.latent_stats.var
        self.latent_scale = float(1.0 / np.sqrt(max(float(np.mean(var)), 1e-12)))
        if self.dist.is_rank0:
            print(f"latent_scale = {self.latent_scale:.4f}")

    def setup_components(self) -> None:
        cfg = self.cfg
        mcfg = cfg.model
        key = jr.PRNGKey(getattr(cfg, "seed", 0))
        grid = tuple(self.ae.bottleneck_grid_size)
        z_dim = int(self.ae.bottleneck_dim)
        self.latent_shape = (*grid, z_dim)
        self.model = DiT(
            space=len(grid),
            z_dim=z_dim,
            dim=mcfg.get("latent_dim", 512),
            grid_size=grid,
            depth=mcfg.vit.get("depth", 4),
            num_heads=mcfg.vit.get("num_heads", 8),
            n_cond=len(cfg.dataset.get("conditions", [])),
            key=key,
            mlp_ratio=mcfg.vit.get("mlp_ratio", 4.0),
            drop_path=mcfg.vit.get("drop_path", 0.0),
        )
        steps_per_epoch = max(1, len(self.train_ds) // cfg.training.batch_size)
        total = cfg.training.n_epochs * steps_per_epoch
        self.schedule = warmup_cosine(
            peak_lr=cfg.training.learning_rate,
            total_steps=total,
            steps_per_epoch=steps_per_epoch,
            n_epochs=cfg.training.n_epochs,
            min_lr=cfg.training.get("final_learning_rate", 1e-6),
        )
        wd = cfg.training.get("weight_decay", 0.0)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.get("clip_to", 1.0))
            if cfg.training.get("clip_grad", True)
            else optax.identity(),
            optax.adamw(self.schedule, weight_decay=wd) if wd > 0 else optax.adam(self.schedule),
        )
        params, _ = eqx.partition(self.model, eqx.is_array)
        self.opt_state = self.optimizer.init(params)
        self.use_ot = bool(cfg.model.get("minibatch_ot", True))

    @eqx.filter_jit
    def _train_step(self, model, opt_state, latents, cond, key):
        def loss_fn(m):
            def fwd(x, t, c):
                return m(x, t, c)
            return fm_forward_loss(fwd, latents, cond, key=key,
                                   latent_scale=self.latent_scale,
                                   use_ot=self.use_ot)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params, static = eqx.partition(model, eqx.is_array)
        g_params, _ = eqx.partition(grads, eqx.is_array)
        updates, opt_state = self.optimizer.update(g_params, opt_state, params)
        params = eqx.apply_updates(params, updates)
        model = eqx.combine(params, static)
        return model, opt_state, loss

    def train_epoch(self, epoch: int, key) -> dict:
        cfg = self.cfg
        bs = cfg.training.batch_size
        n = len(self.train_ds)
        idx_key, key = jr.split(key)
        idx = jr.permutation(idx_key, n)
        losses = []
        for start in range(0, n - bs + 1, bs):
            samples = [self.train_ds[int(idx[i])] for i in range(start, start + bs)]
            z = jnp.stack([jnp.asarray(s.df) for s in samples])
            cond = (
                jnp.stack([jnp.asarray(s.conditioning) for s in samples])
                if samples[0].conditioning is not None
                else None
            )
            step_key, key = jr.split(key)
            self.model, self.opt_state, loss = self._train_step(
                self.model, self.opt_state, z, cond, step_key,
            )
            losses.append(float(loss))
        return {"loss": sum(losses) / max(len(losses), 1)}

    def evaluate(self, epoch: int) -> dict:
        from neugk_jax.evaluate import DiffusionEvaluator

        def _sample(*, key, batch, cond=None, steps=50):
            return self.sample(key=key, batch=batch, cond=cond, steps=steps)

        # for cheap eval we also report the FM training-loss on the val set
        cfg = self.cfg
        bs = cfg.training.batch_size
        n = min(len(self.val_ds), bs * 4)
        losses = []
        key = jr.PRNGKey(epoch)
        for start in range(0, n - bs + 1, bs):
            samples = [self.val_ds[i] for i in range(start, start + bs)]
            z = jnp.stack([jnp.asarray(s.df) for s in samples])
            cond = (
                jnp.stack([jnp.asarray(s.conditioning) for s in samples])
                if samples[0].conditioning is not None
                else None
            )
            step_key, key = jr.split(key)
            losses.append(float(fm_forward_loss(
                lambda x, t, c: self.model(x, t, c),
                z, cond, key=step_key,
                latent_scale=self.latent_scale, use_ot=self.use_ot,
            )))
        out = {"fm_loss": sum(losses) / max(len(losses), 1)}

        # sample-based eval — only when explicitly enabled (slow on CPU)
        if cfg.validation.get("eval_sampling", False):
            ev = DiffusionEvaluator(
                cfg, val_ds=self.val_ds,
                autoencoder=self.ae,
                sample_fn=_sample,
                is_rank0=self.dist.is_rank0,
            )
            metrics, val_plots = ev(
                self.model, epoch=epoch,
                batch_size=bs,
                n_steps=cfg.validation.get("eval_sample_steps", 50),
                n_samples_per_traj=cfg.validation.get("eval_n_samples", 1),
                eval_integrals=cfg.validation.get("eval_integrals", True),
                max_batches=cfg.validation.get("eval_max_batches", None),
            )
            out.update(metrics)
            # hand plots to the wandb logger
            if val_plots and self.dist.is_rank0:
                self.logger.log({f"val_plots/{k}": v for k, v in val_plots.items()},
                                step=epoch)
        return out

    def sample(self, *, key, batch: int, cond: Optional[jnp.ndarray] = None, steps: int = 50):
        latents = euler_sample(
            lambda x, t, c: self.model(x, t, c),
            key=key, shape=(batch, *self.latent_shape),
            cond=cond, steps=steps, latent_scale=self.latent_scale,
        )
        return jax.vmap(self.ae.decode)(latents)
