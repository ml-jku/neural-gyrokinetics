"""GyroSwin runner smoke tests on a tiny synthetic next-step dataset."""

from __future__ import annotations

import pickle
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from omegaconf import OmegaConf

RES = (8, 2, 4, 10, 8)  # (vp, mu, s, x, y)


def _geometry():
    vp, mu, s, x, y = RES
    return {
        "krho": np.linspace(0.0, 0.5, y),
        "ints": np.full(s, 1.0 / s),
        "intmu": np.linspace(0.5, 1.0, mu),
        "intvp": np.full(vp, 0.2),
        "vpgr": np.linspace(-3.0, 3.0, vp),
        "mugr": np.linspace(0.1, 1.0, mu),
        "bn": np.linspace(0.9, 1.1, s),
        "efun": np.full(s, -2.0),
        "rfun": np.full(s, 0.8),
        "bt_frac": np.ones(s),
        "parseval": np.asarray([1.0] + [float(y)] * (y - 1)),
        "kxrh": np.linspace(-1.0, 1.0, x),
        "little_g": np.tile([1.0, 0.1, 1.0], (s, 1)),
        "mas": np.ones(1),
        "tmp": np.ones(1),
        "de": np.ones(1),
        "signz": np.ones(1),
        "vthrat": np.ones(1),
        "d2X": np.asarray(1.0),
        "signB": np.asarray(1.0),
        "adiabatic": np.asarray(1.0),
        "beta": np.asarray(0.0),
        "nlapar": np.asarray(0.0),
        "nlbpar": np.asarray(0.0),
    }


def _make_traj(root: Path, name: str, n_t: int, seed: int):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for t in range(n_t):
        rng.standard_normal((2, *RES)).astype(np.float32).tofile(data / f"timestep_{t:05d}.bin")
        rng.standard_normal((RES[3], RES[2], RES[4])).astype(np.float32).tofile(
            data / f"poten_{t:05d}.bin"
        )
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.random(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], np.float32),
        "density_grad": np.array([1.1], np.float32),
        "s_hat": np.array([0.8], np.float32),
        "q": np.array([1.4], np.float32),
        "resolution": np.array(RES),
        "geometry": _geometry(),
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)


@pytest.fixture
def cyclone_dir(tmp_path):
    _make_traj(tmp_path, "iteration_0", 6, 0)
    _make_traj(tmp_path, "iteration_1", 6, 1)
    return tmp_path


def _stats():
    one = {"full": {"mean": 0.1, "std": 2.0, "min": -1.0, "max": 1.0}}
    return {k: one for k in ("df", "phi", "flux", "fluxavg")}


def _cfg(path: Path, loss_weights: dict, extra: dict | None = None, scheduler: dict | None = None):
    return OmegaConf.create(
        {
            "workflow": "gyroswin",
            "seed": 0,
            "output_path": str(path / "out"),
            "model": {
                "name": "gyroswin_multi",
                "latent_dim": 16,
                "num_layers": 1,
                "decouple_mu": True,
                "conditioning": ["timestep", "itg"],
                "drop_path": 0.1,
                "loss_weights": loss_weights,
                "extra_loss_weights": extra or {},
                "loss_scheduler": scheduler or {},
                "swin": {
                    "patch_size": [2, 1, 2, 5, 2],
                    "window_size": [2, 1, 2, 2, 2],
                    "num_heads": 2,
                    "depth": 1,
                    "merging_hidden_ratio": 2.0,
                    "unmerging_hidden_ratio": 2.0,
                    "c_multiplier": 2,
                    "flux_num_heads": 2,
                    "modulation": "film",
                },
            },
            "dataset": {
                "name": "cyclone",
                "path": str(path),
                "backend": "numpy",
                "training_trajectories": "iteration_0",
                "validation_trajectories": "iteration_1",
                "input_fields": ["df"],
                "separate_zf": True,
                "real_potens": True,
                "offset": 0,
                "normalization": {k: {"type": "zscore"} for k in ("df", "phi", "flux", "fluxavg")},
                "normalization_scope": "dataset",
                "normalization_stats": _stats(),
            },
            "training": {
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-4,
                "final_learning_rate": 1e-6,
                "weight_decay": 0.0,
                "clip_grad": True,
                "clip_to": 1.0,
                "exclude_from_wd": [],
            },
            "validation": {"validate_every_n_epochs": 1, "n_eval_steps": 2, "eval_integrals": True},
            "logging": {"mode": "disabled", "tqdm": False},
        }
    )


def test_gyroswin_runner_epoch_and_eval(cyclone_dir):
    from neugk_jax.gyroswin.runner import GyroSwinRunner

    sched = {
        "phi": {
            "type": "linear",
            "start": 0.0,
            "end": 1.0,
            "start_fraction": 0.0,
            "end_fraction": 1.0,
        }
    }
    cfg = _cfg(cyclone_dir, {"df": 1.0, "phi": 0.1, "flux": 0.0, "fluxavg": 1.0}, scheduler=sched)
    r = GyroSwinRunner(cfg, output_path=cfg.output_path)
    assert r.model.flux_key == "fluxavg"
    assert r.loss_cfg.active == ("df", "phi", "fluxavg")
    logs, info = r.train_epoch(1, jr.PRNGKey(0))
    assert set(logs) == {"total", "df", "phi", "fluxavg"}
    assert all(np.isfinite(v) for v in logs.values())
    metrics, _ = r.evaluate(1)
    for k in ("df_x1", "df_x2", "phi_x1", "phi_x2", "fluxavg_x1", "df"):
        assert k in metrics and np.isfinite(metrics[k])
    assert "phi_int_x1" not in metrics


def test_gyroswin_integral_losses_train_and_eval(cyclone_dir):
    from neugk_jax.gyroswin.runner import GyroSwinRunner

    cfg = _cfg(
        cyclone_dir, {"df": 1.0, "phi": 0.1, "flux": 1.0}, extra={"phi_int": 0.1, "flux_int": 0.1}
    )
    r = GyroSwinRunner(cfg, output_path=cfg.output_path)
    assert r.loss_cfg.integrals == ("phi_int", "flux_int")
    logs, _ = r.train_epoch(1, jr.PRNGKey(0))
    assert {"phi_int", "flux_int"} <= set(logs)
    assert all(np.isfinite(v) for v in logs.values())
    metrics, _ = r.evaluate(1)
    assert np.isfinite(metrics["phi_int_x1"]) and np.isfinite(metrics["flux_int_rel_err_x2"])


def test_cross_losses_rejected(cyclone_dir):
    from neugk_jax.gyroswin.runner import GyroSwinRunner

    cfg = _cfg(cyclone_dir, {"df": 1.0}, extra={"flux_cross": 1.0})
    with pytest.raises(ValueError, match="cross losses"):
        GyroSwinRunner(cfg, output_path=cfg.output_path)


def test_weight_schedule_does_not_retrace(cyclone_dir, monkeypatch):
    import neugk_jax.gyroswin.runner as runner_mod
    from neugk_jax.training.runner import train_step
    from neugk_jax.utils import TRACE_COUNTS

    calls = []
    orig = runner_mod.compute_multi_task_loss

    def counting(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "compute_multi_task_loss", counting)
    sched = {
        "phi": {
            "type": "linear",
            "start": 0.0,
            "end": 1.0,
            "start_fraction": 0.0,
            "end_fraction": 1.0,
        }
    }
    cfg = _cfg(cyclone_dir, {"df": 1.0, "phi": 0.1}, scheduler=sched)
    r = runner_mod.GyroSwinRunner(cfg, output_path=cfg.output_path)
    losses = []
    for step, frac in enumerate((0.0, 0.5, 1.0)):
        batch = r.load_batch(r.train_ds, [0, 1], r.loader.read)
        batch.update(r.step_extras(int(frac * r.total_steps)))
        r.model, r.opt_state, logs = train_step(
            (batch, r.ctx, jr.PRNGKey(step)), r.model, r.opt_state, r.spec
        )
        losses.append(float(logs["total"]))
    assert len(calls) == 1 and TRACE_COUNTS["train_step:GyroSwinRunner"] >= 1
    assert np.all(np.isfinite(losses))


def test_gyroswin_epochs_do_not_retrace(cyclone_dir):
    from neugk_jax.gyroswin.runner import GyroSwinRunner
    from neugk_jax.utils import TRACE_COUNTS

    cfg = _cfg(
        cyclone_dir, {"df": 1.0, "phi": 0.1, "flux": 1.0}, extra={"phi_int": 0.1, "flux_int": 0.1}
    )
    r = GyroSwinRunner(cfg, output_path=cfg.output_path)
    # 3 val samples: the last batch is padded to the evaluator batch
    bs = r.evaluator.batch_size
    assert r.evaluator.plans[-1].mask.tolist() == [1.0] * (3 % bs or bs) + [0.0] * (-3 % bs)
    r.train_epoch(1, jr.PRNGKey(0))
    first, _ = r.evaluate(1)
    TRACE_COUNTS.clear()
    r.train_epoch(2, jr.PRNGKey(1))
    second, _ = r.evaluate(2)
    assert TRACE_COUNTS["train_step:GyroSwinRunner"] == 0
    assert TRACE_COUNTS["gyroswin_eval_step"] == 0
    for k in ("df_rel_l2_x1", "phi_rel_l2_x2", "df_rel_l2", "flux_int_rel_err_x1"):
        assert np.isfinite(second[k]), k


def test_drop_path_active_only_in_training(cyclone_dir):
    from neugk_jax.gyroswin.runner import GyroSwinRunner

    cfg = _cfg(cyclone_dir, {"df": 1.0, "phi": 0.1})
    cfg.model.drop_path = 0.5
    r = GyroSwinRunner(cfg, output_path=cfg.output_path)
    s = r.train_ds[0]
    x, c = jnp.asarray(s.df), jnp.asarray(s.conditioning)
    fwd = eqx.filter_jit(lambda m, k, inf: m(x, c, key=k, inference=inf))
    a = fwd(r.model, jr.PRNGKey(0), False)["df"]
    b = fwd(r.model, jr.PRNGKey(1), False)["df"]
    e0 = fwd(r.model, jr.PRNGKey(0), True)["df"]
    e1 = fwd(r.model, jr.PRNGKey(1), True)["df"]
    assert not np.allclose(a, b)
    assert np.array_equal(e0, e1)
