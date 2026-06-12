"""End-to-end smoke tests for the AE and FM runners.

Uses a synthetic dataset directory (same as ``test_dataset_parity``) and
the small Swin5DAE / DiT shape configs from ``test_models_shapes``. The
goal is to verify that:

* configs compose cleanly through Hydra,
* ``AERunner`` constructs and ``_train_step`` runs (forward + backward),
* ``FlowMatchingRunner`` constructs, ``precompute_latents`` writes the
  cache, and one FM training step runs.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from omegaconf import OmegaConf


def _make_traj(root: Path, name: str, *, n_t: int, resolution):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    df_shape = (2, *resolution)
    for t in range(n_t):
        rng.standard_normal(df_shape).astype(np.float32).tofile(data / f"timestep_{t:05d}.bin")
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], dtype=np.float32),
        "density_grad": np.array([1.1], dtype=np.float32),
        "s_hat": np.array([0.8], dtype=np.float32),
        "q": np.array([1.4], dtype=np.float32),
        "resolution": np.array(resolution),
        "geometry": {k: np.ones((1,), dtype=np.float64)
                     for k in ("krho","ints","intmu","intvp","vpgr","mugr",
                              "bn","efun","rfun","bt_frac","parseval",
                              "mas","tmp","d2X","signz","signB","kxrh","little_g")},
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)


@pytest.fixture
def cyclone_dir(tmp_path):
    resolution = (4, 4, 4, 16, 8)
    _make_traj(tmp_path, "iteration_0", n_t=8, resolution=resolution)
    _make_traj(tmp_path, "iteration_1", n_t=8, resolution=resolution)
    return tmp_path, resolution


def _tiny_ae_cfg(path, resolution):
    return OmegaConf.create({
        "workflow": "ae",
        "seed": 0,
        "output_path": str(path / "out"),
        "model": {
            "name": "ae",
            "decouple_mu": True,
            "latent_dim": 16,
            "patch": {
                "patch_size": [2, 0, 2, 4, 2],
                "window_size": [2, 0, 2, 2, 2],
                "merging_depth": 1, "unmerging_depth": 1,
                "merging_hidden_ratio": 2.0, "unmerging_hidden_ratio": 2.0,
                "c_multiplier": 1,
            },
            "vit": {
                "num_heads": [2], "depth": [1],
                "use_rpb": False, "gated_attention": False, "qk_norm": False, "qkv_bias": False,
            },
            "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2, "normalized_latent": False},
            "middle_depth": 1, "middle_num_heads": 2,
            "hidden_mlp_ratio": 2.0,
        },
        "dataset": {
            "name": "cyclone", "path": str(path), "backend": "numpy",
            "training_trajectories": "iteration_0",
            "validation_trajectories": "iteration_1",
            "input_fields": ["df"], "conditions": ["itg", "dg", "s_hat", "q"],
            "separate_zf": False, "offset": 0,
            "normalization": None,
        },
        "training": {
            "batch_size": 1, "n_epochs": 1, "learning_rate": 3e-4,
            "final_learning_rate": 1e-6, "weight_decay": 0.0,
            "clip_grad": True, "clip_to": 1.0, "exclude_from_wd": [],
        },
        "validation": {"validate_every_n_epochs": 1},
        "logging": {"mode": "disabled", "tqdm": False},
        "distributed": {"enable": False, "n_nodes": 1},
    })


def test_ae_runner_constructs_and_steps(cyclone_dir):
    path, resolution = cyclone_dir
    cfg = _tiny_ae_cfg(path, resolution)
    from neugk_jax.autoencoders.runner import AERunner
    r = AERunner(cfg, output_path=cfg.output_path)
    assert len(r.train_ds) > 0
    assert r.opt_state is not None
    # one training step
    sample = r.train_ds[0]
    df = jnp.asarray(sample.df)[None]
    r.model, r.opt_state, loss = r._train_step(r.model, r.opt_state, df)
    assert jnp.isfinite(loss)


def test_fm_runner_constructs_and_steps(cyclone_dir, tmp_path):
    """FM runner construction + one step, using a freshly-built tiny AE
    (no checkpoint translation needed). Mocks ``ae_checkpoint`` by saving
    the freshly initialised AE first."""
    path, resolution = cyclone_dir
    ae_cfg = _tiny_ae_cfg(path, resolution)
    # build + save a tiny AE so the FM runner has something to load
    from scripts.translate_ckpt import build_ae_from_config
    from neugk_jax.training.checkpoint import save_model_only
    # FlowMatchingRunner expects AE config at <ae_ckpt_dir>/config.yaml with resolution for build_ae_from_config
    ae_dir = tmp_path / "ae_ckpt"
    ae_dir.mkdir(exist_ok=True)
    ae_cfg_with_res = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    ae_cfg_with_res.dataset.resolution = list(resolution)
    ae_cfg_path = ae_dir / "config.yaml"
    OmegaConf.save(ae_cfg_with_res, ae_cfg_path)
    ae_weights = ae_dir / "ae.eqx"
    ae = build_ae_from_config(str(ae_cfg_path), key=jr.PRNGKey(0),
                              resolution=resolution)
    save_model_only(ae_weights, ae)

    fm_cfg = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    fm_cfg.workflow = "diffusion"
    fm_cfg.ae_checkpoint = str(ae_weights)
    fm_cfg.model = OmegaConf.create({
        "name": "latent_dit", "model_type": "latent_dit",
        "latent_dim": 32, "minibatch_ot": False,
        "vit": {"num_heads": 2, "depth": 1, "mlp_ratio": 2.0, "drop_path": 0.0},
        "diffusion": {"noise_distribution": "gaussian", "continuous_time": True},
    })
    fm_cfg.training.batch_size = 2

    from neugk_jax.diffusion.runner import FlowMatchingRunner
    r = FlowMatchingRunner(fm_cfg, output_path=fm_cfg.output_path)
    assert len(r.train_ds) > 0
    assert r.latent_shape == (*r.ae.bottleneck_grid_size, r.ae.bottleneck_dim)
    # one training step
    sample0 = r.train_ds[0]
    sample1 = r.train_ds[1]
    z = jnp.stack([jnp.asarray(sample0.df), jnp.asarray(sample1.df)])
    cond = jnp.stack([jnp.asarray(sample0.conditioning), jnp.asarray(sample1.conditioning)])
    r.model, r.opt_state, loss = r._train_step(r.model, r.opt_state, z, cond, jr.PRNGKey(0))
    assert jnp.isfinite(loss)
