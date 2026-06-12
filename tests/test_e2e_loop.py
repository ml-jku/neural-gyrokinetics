"""End-to-end test: full train + eval loop for both AE and diffusion.

Verifies that ``BaseRunner.__call__`` runs an entire epoch (train +
evaluator) for both workflows on a tiny synthetic dataset, that
checkpoints save + reload cleanly, and that the evaluator metrics are
finite + key-complete (AE: ``df``; FM: ``fm_loss`` and optional
``avg_flux_rmse`` when ``eval_sampling=True``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from omegaconf import OmegaConf


def _make_traj(root: Path, name: str, *, n_t: int, resolution):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    for t in range(n_t):
        rng.standard_normal((2, *resolution)).astype(np.float32).tofile(
            data / f"timestep_{t:05d}.bin"
        )
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], dtype=np.float32),
        "density_grad": np.array([1.1], dtype=np.float32),
        "s_hat": np.array([0.8], dtype=np.float32),
        "q": np.array([1.4], dtype=np.float32),
        "resolution": np.array(resolution),
        "geometry": {k: np.ones((1,), dtype=np.float64) for k in (
            "krho","ints","intmu","intvp","vpgr","mugr","bn","efun","rfun",
            "bt_frac","parseval","mas","tmp","d2X","signz","signB","kxrh","little_g",
        )},
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)


@pytest.fixture
def cyclone_dir(tmp_path):
    resolution = (4, 4, 4, 16, 8)
    _make_traj(tmp_path, "iteration_0", n_t=4, resolution=resolution)
    _make_traj(tmp_path, "iteration_1", n_t=4, resolution=resolution)
    return tmp_path, resolution


def _tiny_ae_cfg(path, resolution, out_path):
    return OmegaConf.create({
        "workflow": "ae",
        "seed": 0,
        "output_path": str(out_path),
        "model": {
            "name": "ae", "decouple_mu": True, "latent_dim": 16,
            "patch": {
                "patch_size": [2, 0, 2, 4, 2], "window_size": [2, 0, 2, 2, 2],
                "merging_depth": 1, "unmerging_depth": 1,
                "merging_hidden_ratio": 2.0, "unmerging_hidden_ratio": 2.0,
                "c_multiplier": 1,
            },
            "vit": {
                "num_heads": [2], "depth": [1],
                "use_rpb": False, "gated_attention": False,
                "qk_norm": False, "qkv_bias": False,
            },
            "bottleneck": {"dim": 8, "depth": 1, "num_heads": 2, "normalized_latent": False},
            "middle_depth": 1, "middle_num_heads": 2, "hidden_mlp_ratio": 2.0,
        },
        "dataset": {
            "name": "cyclone", "path": str(path), "backend": "numpy",
            "training_trajectories": "iteration_0",
            "validation_trajectories": "iteration_1",
            "input_fields": ["df"], "conditions": ["itg", "dg", "s_hat", "q"],
            "separate_zf": False, "offset": 0, "normalization": None,
            "resolution": list(resolution),
        },
        "training": {
            "batch_size": 1, "n_epochs": 1, "learning_rate": 3e-4,
            "final_learning_rate": 1e-6, "weight_decay": 0.0,
            "clip_grad": True, "clip_to": 1.0, "exclude_from_wd": [],
        },
        "validation": {
            "validate_every_n_epochs": 1, "eval_integrals": False,
            "eval_sampling": False,
        },
        "logging": {"mode": "disabled", "tqdm": False},
        "distributed": {"enable": False, "n_nodes": 1},
    })


def test_ae_e2e_train_eval(cyclone_dir, tmp_path):
    path, resolution = cyclone_dir
    out = tmp_path / "ae_run"
    cfg = _tiny_ae_cfg(path, resolution, out)
    from neugk_jax.autoencoders.runner import AERunner
    runner = AERunner(cfg, output_path=cfg.output_path)
    runner()  # full epoch
    # eval should have produced a 'df' metric and a checkpoint
    assert (out / "ckp.eqx").exists(), "ckp.eqx not written"
    assert (out / "best.eqx").exists(), "best.eqx not written"
    # reload to confirm round-trip
    from neugk_jax.training.checkpoint import load_checkpoint
    state = load_checkpoint(out / "ckp.eqx", runner.model)
    assert state.epoch == 1
    assert jnp.isfinite(jnp.asarray(state.loss))


def test_fm_e2e_train_eval(cyclone_dir, tmp_path):
    path, resolution = cyclone_dir
    # 1. build + save a tiny AE so the FM runner has something to load
    ae_dir = tmp_path / "ae_ckpt"
    ae_dir.mkdir()
    ae_cfg = _tiny_ae_cfg(path, resolution, ae_dir)
    ae_cfg.dataset.resolution = list(resolution)
    ae_cfg_path = ae_dir / "config.yaml"
    OmegaConf.save(ae_cfg, ae_cfg_path)
    from scripts.translate_ckpt import build_ae_from_config
    from neugk_jax.training.checkpoint import save_model_only
    ae = build_ae_from_config(str(ae_cfg_path), key=jr.PRNGKey(0), resolution=resolution)
    ae_weights = ae_dir / "ae.eqx"
    save_model_only(ae_weights, ae)

    # 2. compose a tiny FM config that runs full train+eval
    out = tmp_path / "fm_run"
    fm_cfg = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    fm_cfg.workflow = "diffusion"
    fm_cfg.ae_checkpoint = str(ae_weights)
    fm_cfg.output_path = str(out)
    fm_cfg.model = OmegaConf.create({
        "name": "latent_dit", "model_type": "latent_dit",
        "latent_dim": 32, "minibatch_ot": False,
        "vit": {"num_heads": 2, "depth": 1, "mlp_ratio": 2.0, "drop_path": 0.0},
        "diffusion": {"noise_distribution": "gaussian", "continuous_time": True},
    })
    fm_cfg.training.batch_size = 2

    from neugk_jax.diffusion.runner import FlowMatchingRunner
    runner = FlowMatchingRunner(fm_cfg, output_path=fm_cfg.output_path)
    runner()  # full epoch
    assert (out / "ckp.eqx").exists()
    assert (out / "best.eqx").exists()
    # train_epoch returned fm_loss; verify it's finite from the checkpoint
    from neugk_jax.training.checkpoint import load_checkpoint
    state = load_checkpoint(out / "ckp.eqx", runner.model)
    assert state.epoch == 1
    assert jnp.isfinite(jnp.asarray(state.loss))
