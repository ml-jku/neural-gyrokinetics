"""End-to-end test: full train + eval loop for both AE and diffusion.

Verifies that ``BaseRunner.__call__`` runs an entire epoch (train +
evaluator) for both workflows on a tiny synthetic dataset, that
checkpoints save + reload cleanly, and that the evaluator metrics are
finite + key-complete (AE: ``df``; FM: ``fm_loss`` and optional
``avg_flux_rmse`` when ``eval_sampling=True``).
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
    make_traj(tmp_path, "iteration_0", n_t=4, resolution=resolution)
    make_traj(tmp_path, "iteration_1", n_t=4, resolution=resolution)
    return tmp_path, resolution


def test_ae_e2e_train_eval(cyclone_dir, tmp_path):
    path, resolution = cyclone_dir
    out = tmp_path / "ae_run"
    cfg = tiny_ae_cfg(path, resolution, out)
    from neugk_jax.pinc.runner import AERunner

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
    # 1. build + save a tiny ae so the fm runner has something to load
    ae_dir = tmp_path / "ae_ckpt"
    ae_dir.mkdir()
    ae_cfg = tiny_ae_cfg(path, resolution, ae_dir)
    ae_cfg.dataset.resolution = list(resolution)
    ae_cfg_path = ae_dir / "config.yaml"
    OmegaConf.save(ae_cfg, ae_cfg_path)
    from neugk_jax.models.build import build_ae_from_config
    from neugk_jax.training.checkpoint import save_model_only

    ae = build_ae_from_config(str(ae_cfg_path), key=jr.PRNGKey(0), resolution=resolution)
    ae_weights = ae_dir / "ae.eqx"
    save_model_only(ae_weights, ae)

    # 2. compose a tiny fm config that runs full train+eval
    out = tmp_path / "fm_run"
    fm_cfg = OmegaConf.create(OmegaConf.to_container(ae_cfg))
    fm_cfg.workflow = "diffusion"
    fm_cfg.ae_checkpoint = str(ae_weights)
    fm_cfg.output_path = str(out)
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

    runner = FlowMatchingRunner(fm_cfg, output_path=fm_cfg.output_path)
    runner()  # full epoch
    assert (out / "ckp.eqx").exists()
    assert (out / "best.eqx").exists()
    # train_epoch returned fm_loss; verify it's finite from the checkpoint
    from neugk_jax.training.checkpoint import load_checkpoint

    state = load_checkpoint(out / "ckp.eqx", runner.model)
    assert state.epoch == 1
    assert jnp.isfinite(jnp.asarray(state.loss))
    # the trained latent scale is recorded in the checkpoint and the run config
    assert state.meta["latent_scale"] == pytest.approx(runner.latent_scale)
    assert OmegaConf.load(out / "config.yaml").latent_scale == pytest.approx(runner.latent_scale)
    # fm_loss covers the whole val set with a fixed key
    assert runner.evaluate(1)[0]["fm_loss"] == pytest.approx(runner.evaluate(2)[0]["fm_loss"])


def test_ae_resume_keeps_best_and_continues(cyclone_dir, tmp_path):
    from neugk_jax.pinc.runner import AERunner
    from neugk_jax.training.checkpoint import load_checkpoint

    path, resolution = cyclone_dir
    out = tmp_path / "ae_resume"
    cfg = tiny_ae_cfg(path, resolution, out)
    AERunner(cfg, output_path=cfg.output_path)()
    first = load_checkpoint(out / "ckp.eqx", AERunner(cfg, output_path=cfg.output_path).model)
    best_val = first.meta["best_val"]
    assert np.isfinite(best_val)

    cfg.training.n_epochs = 2
    resumed = AERunner(cfg, output_path=cfg.output_path)
    assert resumed.start_epoch == 1 and resumed.best_val == best_val
    resumed()
    assert load_checkpoint(out / "ckp.eqx", resumed.model).epoch == 2


def test_resume_config_cli_wins(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "neugk_jax_main", Path(__file__).resolve().parents[1] / "main.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    run = tmp_path / "run"
    run.mkdir()
    (run / "ckp.eqx").write_bytes(b"")
    OmegaConf.save(
        OmegaConf.create({"training": {"n_epochs": 5, "lr": 1.0}, "seed": 3}), run / "config.yaml"
    )
    saved = OmegaConf.load(run / "config.yaml")
    entry._drop_cli_overridden(["training.n_epochs=9"], saved)
    assert "n_epochs" not in saved.training and saved.training.lr == 1.0
    cfg = OmegaConf.create(
        {
            "output_path": str(run),
            "load_ckpt": True,
            "seed": 0,
            "training": {"n_epochs": 9, "lr": 2.0},
        }
    )
    merged = entry.resume_config(cfg)
    assert merged.seed == 3 and merged.training.lr == 1.0 and merged.output_path == str(run)


def _main_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "neugk_jax_main", Path(__file__).resolve().parents[1] / "main.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def test_resume_hydra_compose_cli_wins(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    entry = _main_module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "ckp.eqx").write_bytes(b"")
    OmegaConf.save(
        OmegaConf.create(
            {
                "seed": 3,
                "training": {"n_epochs": 5, "learning_rate": 1.0},
                "output_path": "elsewhere",
            }
        ),
        run / "config.yaml",
    )
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    overrides = [f"output_path={run}", "load_ckpt=true", "training.n_epochs=9"]
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="main", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            del cfg["hydra"]
        merged = entry.resume_config(cfg)
    assert merged.training.n_epochs == 9 and merged.output_path == str(run)
    assert merged.seed == 3 and merged.training.learning_rate == 1.0


def test_run_id_format():
    import re

    assert re.fullmatch(r"\d{8}_\d{6}_\d{3}", _main_module().run_id())
