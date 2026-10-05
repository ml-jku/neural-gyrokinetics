"""End-to-end smoke tests for the AE and FM runners.

Uses a synthetic dataset directory (same as ``test_dataset_parity``) and
the small Swin5DAE / DiT shape configs from ``test_models_shapes``. The
goal is to verify that:

* ``AERunner`` constructs and one jitted train step runs (forward + backward),
* ``FlowMatchingRunner`` constructs, ``precompute_latents`` writes the
  cache, and one FM step runs on latents gathered from the device table,
* each Hydra experiment preset composes.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import make_traj, tiny_ae_cfg
from omegaconf import OmegaConf


@pytest.fixture
def cyclone_dir(tmp_path):
    resolution = (4, 4, 4, 16, 8)
    make_traj(tmp_path, "iteration_0", n_t=8, resolution=resolution)
    make_traj(tmp_path, "iteration_1", n_t=8, resolution=resolution)
    return tmp_path, resolution


def test_ae_runner_constructs_and_steps(cyclone_dir):
    path, resolution = cyclone_dir
    cfg = tiny_ae_cfg(path, resolution)
    from neugk_jax.pinc.runner import AERunner

    r = AERunner(cfg, output_path=cfg.output_path)
    assert len(r.train_ds) > 0
    assert r.opt_state is not None
    from neugk_jax.training.runner import train_step

    batch = r.load_batch(r.train_ds, [0], r.loader.read)
    r.model, r.opt_state, logs = train_step(
        (batch, r.ctx, jr.PRNGKey(0)), r.model, r.opt_state, r.spec
    )
    assert jnp.isfinite(logs["total"])
    x = jnp.ones((1, 2, *resolution))
    assert r.loss_type is None and float(r.df_recon_loss(2 * x, x)) == pytest.approx(1.0, rel=1e-3)
    cfg.training.loss_type = "l1"
    assert float(AERunner(cfg, output_path=cfg.output_path).df_recon_loss(x + 2, x)) == 2.0


def test_vqvae_model_routes_to_the_vq_runner(cyclone_dir, monkeypatch):
    import importlib.util

    from helpers import tiny_vq_cfg

    import neugk_jax.pinc.runner as runners

    path, resolution = cyclone_dir
    cfg = tiny_ae_cfg(path, resolution)
    cfg.model = OmegaConf.create(tiny_vq_cfg())
    with pytest.raises(ValueError, match="vq-vae"):
        runners.AERunner(cfg, output_path=cfg.output_path)

    spec = importlib.util.spec_from_file_location(
        "neugk_jax_main", Path(__file__).resolve().parents[1] / "main.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    used = []
    for name in ("AERunner", "VQVAERunner"):
        stub = type(name, (), {"__init__": lambda self, *a, **k: used.append(type(self).__name__)})
        monkeypatch.setattr(runners, name, type(name, (stub,), {"__call__": lambda self: None}))
    for workflow in ("ae", "pinc", "vqvae"):
        entry.dispatch_runner(OmegaConf.merge(cfg, {"workflow": workflow}))
    cfg.model.model_type = "ae"
    entry.dispatch_runner(cfg)
    assert used == ["VQVAERunner"] * 3 + ["AERunner"]


def test_fm_runner_constructs_and_steps(cyclone_dir, tmp_path):
    """FM runner construction + one step, using a freshly-built tiny AE
    (no checkpoint translation needed). Mocks ``ae_checkpoint`` by saving
    the freshly initialised AE first."""
    path, resolution = cyclone_dir
    ae_cfg = tiny_ae_cfg(path, resolution)
    # build + save a tiny ae so the fm runner has something to load
    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.training.checkpoint import save_model_only

    # FlowMatchingRunner expects ae config at <ae_ckpt_dir>/config.yaml with resolution for build_ae_from_config
    ae_dir = tmp_path / "ae_ckpt"
    ae_dir.mkdir(exist_ok=True)
    ae_cfg_with_res = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    ae_cfg_with_res.dataset.resolution = list(resolution)
    ae_cfg_path = ae_dir / "config.yaml"
    OmegaConf.save(ae_cfg_with_res, ae_cfg_path)
    ae_weights = ae_dir / "ae.eqx"
    ae = build_ae_from_config(str(ae_cfg_path), key=jr.PRNGKey(0), resolution=resolution)
    save_model_only(ae_weights, ae)

    fm_cfg = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    fm_cfg.workflow = "diffusion"
    fm_cfg.ae_checkpoint = str(ae_weights)
    fm_cfg.model = OmegaConf.create(
        {
            "name": "latent_dit",
            "model_type": "latent_dit",
            "latent_dim": 32,
            "minibatch_ot": False,
            "vit": {"num_heads": 2, "depth": 1, "mlp_ratio": 2.0, "drop_path": 0.0},
            "diffusion": {"noise_distribution": "gaussian", "continuous_time": True},
        }
    )
    fm_cfg.training.batch_size = 2

    from neugk_jax.diffusion.runner import FlowMatchingRunner

    r = FlowMatchingRunner(fm_cfg, output_path=fm_cfg.output_path)
    assert len(r.train_ds) > 0
    assert r.latent_shape == (*r.ae.bottleneck_grid_size, r.ae.bottleneck_dim)
    # the latent table is device resident and matches the dataset's cached latents
    assert r.ctx["latents"].shape == (len(r.train_ds), *r.latent_shape)
    assert np.allclose(np.asarray(r.ctx["latents"][1]), r.train_ds[1].df)
    from neugk_jax.training.runner import train_step

    batch = r.load_batch(r.train_ds, [0, 1], r.loader.read)
    assert set(batch) == {"idx"}
    r.model, r.opt_state, logs = train_step(
        (batch, r.ctx, jr.PRNGKey(0)), r.model, r.opt_state, r.spec
    )
    assert jnp.isfinite(logs["fm_loss"])


@pytest.mark.parametrize("experiment", ["ae", "diffusion", "gyroswin", "pinc_revival", "vqvae"])
def test_experiment_presets_compose(experiment, monkeypatch, tmp_path):
    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("NEUGK_DATA", str(tmp_path))
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="main", overrides=[f"experiment={experiment}"])
    assert cfg.workflow == experiment.split("_")[0]
    assert cfg.training.num_workers > 0 and "logging" in cfg
    if experiment == "diffusion":
        assert list(cfg.model.conditioning) == ["itg", "dg", "s_hat", "q"]
        assert cfg.ae_checkpoint is None
    if experiment == "pinc_revival":
        assert cfg.stage == "peft" and cfg.model.legacy_swin_shortcut
        assert cfg.model.peft.lora.strategy == "attention_mlp"
    assert cfg.dataset.path == str(tmp_path)
    stats = cfg.dataset.get("normalization_stats")
    assert stats is None or (stats.startswith(str(tmp_path)) and stats.endswith("_stats.pkl"))


def test_dataset_path_is_required():
    from hydra import compose, initialize_config_dir
    from omegaconf.errors import InterpolationResolutionError

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="main", overrides=["experiment=ae"])
    if "NEUGK_DATA" in __import__("os").environ:
        pytest.skip("NEUGK_DATA is set")
    with pytest.raises(InterpolationResolutionError, match="NEUGK_DATA"):
        _ = cfg.dataset.path
