"""AE training pipeline. Single-task MSE on df with optax AdamW + warmup-cosine."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from neugk_jax.autoencoders.swin5d_ae import Swin5DAE
from neugk_jax.dataset import CycloneDataset, KvikIOBackend, NumpyBackend
from neugk_jax.losses import df_loss
from neugk_jax.training.ddp import data_sharding, replicated
from neugk_jax.training.runner import BaseRunner
from neugk_jax.training.schedulers import warmup_cosine


def _exclude_wd_mask(model, exclude: list[str]):
    """Build a leaf-mask: True where weight decay applies (False on excluded names)."""
    paths_and_leaves = jax.tree_util.tree_leaves_with_path(model)
    def _key_str(path):
        return ".".join(str(k.name) if hasattr(k, "name") else str(k) for k in path)
    return jax.tree_util.tree_map(
        lambda p_leaf: not any(tok in _key_str(p_leaf[0]).lower() for tok in (s.lower() for s in exclude)),
        paths_and_leaves,
        is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2,
    )


class AERunner(BaseRunner):
    """Trains the Swin5DAE on cyclone data with MSE on df."""

    def setup_data(self) -> None:
        cfg = self.cfg
        # conditioning is gyroswin-specific; AE must stay unconditional
        if cfg.model.get("conditioning") not in (None, [], ()):
            raise ValueError(
                "`model.conditioning` is set but the AE workflow does not accept "
                "scalar conditioning. Drop it from the config or switch to "
                "workflow=gyroswin."
            )
        # training.amp.enable=True → train reads bf16 shards (or fp32 with on-the-fly
        # quantize when the shard is missing). validation always reads fp32.
        amp = cfg.training.get("amp", {}) or {}
        amp_enabled = bool(amp.get("enable", False))
        amp_dtype = amp.get("dtype", "bf16") if amp_enabled else None
        # legacy: ``dataset.prefer_dtype`` / ``dataset.prefer_bf16`` overrides amp
        legacy = cfg.dataset.get("prefer_dtype", None)
        if legacy is None and cfg.dataset.get("prefer_bf16", False):
            legacy = "bf16"
        train_dtype = legacy or amp_dtype
        def _backend(dt: str | None):
            if getattr(cfg.dataset, "backend", "kvikio") == "kvikio":
                return KvikIOBackend(rank=self.dist.process_id, prefer_dtype=dt)
            return NumpyBackend(prefer_dtype=dt)
        train_backend = _backend(train_dtype)
        val_backend = _backend(None)
        norm_stats = getattr(cfg.dataset, "normalization_stats", None)
        common = dict(
            path=cfg.dataset.path,
            fields_to_load=tuple(cfg.dataset.get("input_fields", ("df",))),
            conditions=tuple(cfg.dataset.get("conditions", ("itg", "dg", "s_hat", "q"))),
            mode="ae",
            separate_zf=cfg.dataset.get("separate_zf", False),
            normalization=cfg.dataset.get("normalization"),
            normalization_scope=cfg.dataset.get("normalization_scope", "dataset"),
            normalization_stats=norm_stats,
            offset=cfg.dataset.get("offset", 0),
        )
        # mirrors torch ``neugk/dataset/__init__.py:217`` — separate filters per split
        train_filters = self._omegaconf_to_dict(cfg.dataset.get("training_cond_filters"))
        eval_filters = self._omegaconf_to_dict(cfg.dataset.get("eval_cond_filters"))
        self.train_ds = CycloneDataset(
            split="train",
            trajectories=cfg.dataset.training_trajectories,
            cond_filters=train_filters,
            backend=train_backend,
            **common,
        )
        self.val_ds = CycloneDataset(
            split="val",
            trajectories=cfg.dataset.validation_trajectories,
            cond_filters=eval_filters,
            backend=val_backend,
            **common,
        )

    @staticmethod
    def _omegaconf_to_dict(node):
        if node is None:
            return None
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(node, resolve=True)
        except Exception:
            return dict(node)

    def setup_components(self) -> None:
        cfg = self.cfg
        key = jr.PRNGKey(getattr(cfg, "seed", 0))
        mcfg = cfg.model
        vit = mcfg.vit
        patch = mcfg.patch
        bn = mcfg.bottleneck
        depth = list(vit["depth"])
        in_ch = 2 * (2 if cfg.dataset.get("separate_zf", False) else 1)
        self.model = Swin5DAE(
            space=5,
            decouple_mu=mcfg.get("decouple_mu", True),
            dim=mcfg["latent_dim"],
            base_resolution=list(self.train_ds.resolution),
            in_channels=in_ch, out_channels=in_ch,
            patch_size=list(patch["patch_size"]),
            window_size=list(patch["window_size"]),
            depth=depth,
            num_heads=list(vit["num_heads"]),
            num_layers=len(depth),
            middle_depth=mcfg.get("middle_depth", 2),
            middle_num_heads=mcfg.get("middle_num_heads", 8),
            bottleneck_dim=bn.get("dim"),
            bottleneck_depth=bn.get("depth", 2),
            bottleneck_num_heads=bn.get("num_heads", 2),
            hidden_mlp_ratio=mcfg.get("hidden_mlp_ratio", 2.0),
            merging_hidden_ratio=patch.get("merging_hidden_ratio", 8.0),
            unmerging_hidden_ratio=patch.get("unmerging_hidden_ratio", 8.0),
            merging_depth=patch.get("merging_depth", 2),
            unmerging_depth=patch.get("unmerging_depth", 2),
            c_multiplier=int(patch.get("c_multiplier", 2)),
            normalized_latent=bn.get("normalized_latent", False),
            qkv_bias=vit.get("qkv_bias", False),
            qk_norm=vit.get("qk_norm", False),
            use_rpb=vit.get("use_rpb", True),
            gated_attention=vit.get("gated_attention", False),
            norm_affine=False,
            key=key,
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
        exclude = list(cfg.training.get("exclude_from_wd", []))
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.get("clip_to", 1.0))
            if cfg.training.get("clip_grad", True)
            else optax.identity(),
            optax.adamw(self.schedule, weight_decay=wd) if wd > 0 else optax.adam(self.schedule),
        )
        self.optimizer = optimizer
        params, _ = eqx.partition(self.model, eqx.is_array)
        self.opt_state = optimizer.init(params)
        self.steps_per_epoch = steps_per_epoch
        # multi-device: replicate model+opt_state on the mesh, shard data on leading axis
        if self.dist.local_device_count > 1:
            rep = replicated(self.dist.mesh)
            self.model = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, rep) if eqx.is_array(x) else x, self.model,
            )
            self.opt_state = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, rep) if eqx.is_array(x) else x, self.opt_state,
            )

    @eqx.filter_jit
    def _train_step(self, model, opt_state, batch):
        sep_zf = bool(self.cfg.dataset.get("separate_zf", False))
        def loss_fn(m):
            pred = jax.vmap(lambda x: m(x)["df"])(batch)
            return df_loss(pred, batch, separate_zf=sep_zf)

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
        idx = jr.permutation(key, n)
        losses = []
        starts = list(range(0, n - bs + 1, bs))
        # single-threaded prefetch: load batch i+1 while jax is training on batch i
        from concurrent.futures import ThreadPoolExecutor
        import time as _time

        def _load(start):
            samples = [self.train_ds[int(idx[i])] for i in range(start, start + bs)]
            return jnp.stack([jnp.asarray(s.df) for s in samples])

        ex = ThreadPoolExecutor(max_workers=1)
        pbar = starts
        if self.dist.is_rank0 and cfg.logging.get("tqdm", False):
            pbar = tqdm(starts, desc=f"epoch {epoch}")
        future = ex.submit(_load, starts[0]) if starts else None
        t_data_ms = []
        t_step_ms = []
        multi_dev = self.dist.local_device_count > 1
        data_shard = data_sharding(self.dist.mesh) if multi_dev else None
        try:
            for i, start in enumerate(pbar):
                _t = _time.perf_counter_ns()
                df = future.result()
                if multi_dev:
                    df = jax.device_put(df, data_shard)
                t_data_ms.append((_time.perf_counter_ns() - _t) / 1e6)
                if i + 1 < len(starts):
                    future = ex.submit(_load, starts[i + 1])
                _t = _time.perf_counter_ns()
                self.model, self.opt_state, loss = self._train_step(self.model, self.opt_state, df)
                lf = float(loss)  # block on device
                t_step_ms.append((_time.perf_counter_ns() - _t) / 1e6)
                losses.append(lf)
        finally:
            ex.shutdown(wait=False)
        # drop the first step from medians (jit compile dominates)
        m_data = sorted(t_data_ms[1:] or t_data_ms)[len(t_data_ms[1:] or t_data_ms) // 2] if t_data_ms else 0.0
        m_step = sorted(t_step_ms[1:] or t_step_ms)[len(t_step_ms[1:] or t_step_ms) // 2] if t_step_ms else 0.0
        if self.dist.is_rank0:
            print(f"[ae] median per-step: data_wait={m_data:.1f}ms step={m_step:.1f}ms "
                  f"(first step={t_step_ms[0]:.1f}ms incl. jit)")
        loss_logs = {
            "df_mse": sum(losses) / max(len(losses), 1),
            "total_mse": sum(losses) / max(len(losses), 1),
        }
        info = {
            "data_ms": m_data, "step_ms": m_step,
            "first_step_ms": t_step_ms[0] if t_step_ms else 0.0,
        }
        return loss_logs, info

    def evaluate(self, epoch: int):
        from neugk_jax.evaluate import AEEvaluator
        ev = AEEvaluator(self.cfg, val_ds=self.val_ds, is_rank0=self.dist.is_rank0)
        metrics, plots = ev(
            self.model,
            epoch=epoch,
            batch_size=self.cfg.training.batch_size,
            eval_integrals=self.cfg.validation.get("eval_integrals", False),
        )
        return metrics, plots
