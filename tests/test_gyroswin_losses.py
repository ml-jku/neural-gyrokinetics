"""GyroSwin loss terms, loss-config resolution and the jittable flux-integral core."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from neugk_jax.losses import l1, relative_norm_mse
from neugk_jax.training.loss_scheduler import LossConfig, compute_multi_task_loss


def test_relative_norm_mse_matches_formula():
    rng = np.random.default_rng(0)
    p, y = rng.standard_normal((3, 4, 5)), rng.standard_normal((3, 4, 5))
    ref = np.mean([np.sum((p[b] - y[b]) ** 2) / (np.sum(y[b] ** 2) + 1e-4) for b in range(3)])
    assert float(relative_norm_mse(jnp.asarray(p), jnp.asarray(y))) == pytest.approx(ref, rel=1e-5)


def test_l1_reshapes_target():
    p = jnp.asarray([[1.0], [2.0], [4.0]])
    y = jnp.asarray([0.0, 3.0, 4.0])
    assert float(l1(p, y)) == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_multi_task_loss_forms():
    rng = np.random.default_rng(1)
    preds = {
        "df": jnp.asarray(rng.standard_normal((2, 4, 3))),
        "phi": jnp.asarray(rng.standard_normal((2, 5))),
        "fluxavg": jnp.asarray(rng.standard_normal((2, 1))),
    }
    tgts = {
        "df": jnp.asarray(rng.standard_normal((2, 4, 3))),
        "phi": jnp.asarray(rng.standard_normal((2, 5))),
        "fluxavg": jnp.asarray(rng.standard_normal(2)),
    }
    w = {"df": 1.0, "phi": 0.5, "fluxavg": 2.0}
    total, parts = compute_multi_task_loss(preds, tgts, w, ("df", "phi", "fluxavg"))
    assert float(parts["df"]) == pytest.approx(float(relative_norm_mse(preds["df"], tgts["df"])))
    assert float(parts["fluxavg"]) == pytest.approx(
        float(jnp.mean(jnp.abs(preds["fluxavg"][:, 0] - tgts["fluxavg"])))
    )
    assert float(total) == pytest.approx(
        float(parts["df"] + 0.5 * parts["phi"] + 2 * parts["fluxavg"]), rel=1e-6
    )
    _, zf = compute_multi_task_loss(preds, tgts, w, ("df",), separate_zf_loss=True)
    ref = jnp.mean((preds["df"][:, :2] - tgts["df"][:, :2]) ** 2) + relative_norm_mse(
        preds["df"][:, 2:], tgts["df"][:, 2:]
    )
    assert float(zf["df"]) == pytest.approx(float(ref), rel=1e-6)


def test_loss_config_matches_upstream_rules():
    sched = {
        "df": {"type": "linear", "start": 0, "end": 0, "start_fraction": 0, "end_fraction": 1.0},
        "phi": None,
        "flux": None,
        "fluxavg": {
            "type": "linear",
            "start": 1,
            "end": 1,
            "start_fraction": 0,
            "end_fraction": 1.0,
        },
        "phi_int": None,
        "flux_int": None,
        "phi_cross": None,
        "flux_cross": None,
    }
    cfg = LossConfig(
        {"df": 1.0, "phi": 0.1, "flux": 0.0, "fluxavg": 0.0},
        {"phi_int": 0.0, "flux_int": 0.0, "phi_cross": 0.0, "flux_cross": 0.0},
        sched,
    )
    assert cfg.outputs == ("df", "phi", "fluxavg")
    assert cfg.active == ("df", "phi", "fluxavg")
    assert cfg.flux_key == "fluxavg"
    # a schedule replaces the static weight
    weights = {k: float(v) for k, v in cfg.weights_at(7, 10).items()}
    assert weights == pytest.approx({"df": 0.0, "phi": 0.1, "fluxavg": 1.0})
    assert cfg.integrals == ()


def test_loss_config_rejects_bad_keys():
    with pytest.raises(ValueError, match="unknown loss keys"):
        LossConfig({"df": 1.0, "avgflux": 1.0})
    with pytest.raises(ValueError, match="cross losses"):
        LossConfig({"df": 1.0}, {"phi_cross": 0.5})
    with pytest.raises(ValueError, match="both flux"):
        LossConfig({"df": 1.0, "flux": 1.0, "fluxavg": 1.0})
