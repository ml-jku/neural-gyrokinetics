"""PINC-AE PEFT: physics losses, spectral and df stats, the runner and evaluator."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import RES, TARGETS, make_geometry, make_traj, tiny_ae, tiny_ae_model_cfg
from omegaconf import OmegaConf

STD = {k: jnp.full((RES[-1],), 0.5) for k in ("kyspec", "qspec")}


def test_pinc_losses_identity_gradient_and_core_agreement():
    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry
    from neugk_jax.pinc.losses import pinc_integrals, pinc_losses

    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        g = jax.tree_util.tree_map(jnp.asarray, precompute_geometry(make_geometry(RES), np.float64))
        geom = {k: np.stack([v] * 2) for k, v in g.items()}
        tgt = jnp.asarray(rng.standard_normal((2, 4, *RES)))
        ident = {k: float(v) for k, v in pinc_losses(geom, tgt, tgt, ds=0.0625).items()}
        p = pinc_integrals(geom, tgt, ds=0.0625)
        phi, (_, eflux, _) = flux_integral(g, tgt[0, :2] + tgt[0, 2:])
        q_total = float(jnp.sum(p["qspec"][0]))
    for k in ("phi_int", "kyspec", "qspec", "phi_int_mse"):
        assert ident[k] == 0.0, k
    assert ident["flux_int_l1"] == pytest.approx(ident["flux_int"], abs=1e-12)
    # only the mean |pflux| of the target remains
    assert ident["flux_int"] < 1e-12
    np.testing.assert_allclose(p["phi"][0], phi, rtol=1e-12)
    np.testing.assert_allclose([p["eflux"][0], q_total], [eflux] * 2, rtol=1e-10)

    geom32 = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float32), geom)
    tgt32 = jnp.asarray(tgt, jnp.float32)
    pred = tgt32 + 0.3 * jnp.asarray(rng.standard_normal(tgt.shape), jnp.float32)

    def total(x):
        terms = pinc_losses(geom32, x, tgt32, ds=0.0625, mode_std=STD)
        return sum(terms[k] for k in ("phi_int", "flux_int", "kyspec", "qspec"))

    grad = jax.grad(total)(pred)
    assert float(total(pred)) > 0 and np.isfinite(np.asarray(grad)).all() and grad.any()


def test_relative_and_spectral_loss_formulas():
    from neugk_jax.pinc.losses import relative_l1_snap, served_spectral_loss

    rng = np.random.default_rng(1)
    p, t = rng.standard_normal((4, 3, 5)), rng.standard_normal((4, 3, 5))
    ref = np.mean(np.abs(p - t).sum((1, 2)) / (np.abs(t).sum((1, 2)) + 1e-8))
    assert float(relative_l1_snap(jnp.asarray(p), jnp.asarray(t))) == pytest.approx(ref, rel=1e-6)
    ps, ts = np.abs(p[:, 0]), np.abs(t[:, 0])
    dl = np.log1p(ps) - np.log1p(ts)
    for std in (np.full(5, 0.7), None):
        s = np.log1p(ts).std(0, ddof=1) if std is None else std
        got = served_spectral_loss(jnp.asarray(ps), jnp.asarray(ts), std)
        assert float(got) == pytest.approx(np.mean(np.abs(dl / (s + 1e-8))), rel=1e-6)


def _make_traj(root, name, *, n_t, flux=2.0):
    rng = np.random.default_rng(len(name) + n_t)
    return make_traj(
        root,
        name,
        n_t=n_t,
        flux=np.full(n_t, flux, np.float32),
        kyspec=rng.uniform(0.0, 3.0, (n_t, RES[-1])),
        fluxspec=rng.uniform(0.0, 1.0, (n_t, RES[-1])),
    )


def test_spectral_stats_pool_trajectories(tmp_path):
    from neugk_jax.dataset import CycloneDataset, NumpyBackend

    metas = [_make_traj(tmp_path, f"iteration_{i}", n_t=6 + i) for i in range(2)]
    ds = CycloneDataset(
        path=str(tmp_path), trajectories="iteration_{0-1}", offset=2, backend=NumpyBackend()
    )
    st = ds.spectral_stats("kyspec")
    logs = np.concatenate([np.log1p(m["kyspec"][2:]) for m in metas])
    # the epsilon-count prior shifts the pooled moments by ~1e-4 relative here
    np.testing.assert_allclose(st["mean"], logs.mean(0), rtol=1e-3)
    np.testing.assert_allclose(st["std"], logs.std(0), rtol=1e-3)
    with pytest.raises(KeyError):
        ds.spectral_stats("missing_spectrum")


def _stats_pkl(path, rng=None):
    from neugk_jax.utils import RunningStats

    rms = RunningStats(prior_count=1.0)
    rng = rng or np.random.default_rng(0)
    rms.mean = rng.standard_normal((4, *RES)).astype(np.float32)
    rms.var = rng.uniform(0.5, 2.0, (4, *RES)).astype(np.float32)
    rms.min, rms.max = rms.mean - 3, rms.mean + 3
    path.write_bytes(pickle.dumps({"df": rms}))
    return rms


def test_stats_aggregate_in_stored_precision(tmp_path):
    from neugk_jax.dataset import CycloneDataset, NumpyBackend

    _make_traj(tmp_path, "iteration_0", n_t=4)
    rms = _stats_pkl(tmp_path / "stats.pkl")
    agg = (1, 3, 4, 5)
    ds = CycloneDataset(
        path=str(tmp_path),
        trajectories="iteration_0",
        separate_zf=True,
        normalization={"df": {"type": "zscore", "agg_axes": list(agg)}},
        normalization_stats=str(tmp_path / "stats.pkl"),
        backend=NumpyBackend(),
    )
    mean = rms.mean.mean(agg, keepdims=True)
    var = rms.var.mean(agg, keepdims=True) + ((rms.mean - mean) ** 2).mean(agg, keepdims=True)
    np.testing.assert_array_equal(ds.stats["df"]["full"]["mean"], mean)
    np.testing.assert_array_equal(ds.stats["df"]["full"]["std"], np.sqrt(var))


def test_pinc_runner_trains_only_adapters_and_validates(tmp_path):
    from neugk_jax.pinc.peft import PINCPEFTRunner
    from neugk_jax.training.checkpoint import save_model_only
    from neugk_jax.training.ddp import local_view, shard_batch
    from neugk_jax.training.runner import train_step

    for i, f in enumerate((2.0, 3.0, 0.5)):
        _make_traj(tmp_path, f"iteration_{i}", n_t=6, flux=f)
    _stats_pkl(tmp_path / "stats.pkl")
    (tmp_path / "base").mkdir()
    save_model_only(tmp_path / "base" / "best.eqx", tiny_ae())
    lora = {"r": 2, "lora_alpha": 0.5, "target_modules": TARGETS}
    weights = {"df": 1.0, "phi_int": 1.0, "flux_int": 1.0, "kyspec": 1.0}
    dataset = {
        "path": str(tmp_path),
        "backend": "numpy",
        "training_trajectories": "iteration_{0-2}",
        "validation_trajectories": ["iteration_1"],
        "separate_zf": True,
        "offset": 1,
        "normalization": {"df": {"type": "zscore", "agg_axes": [1, 3, 4, 5]}},
        "normalization_stats": str(tmp_path / "stats.pkl"),
        "training_cond_filters": {"last_flux": [1.0, float("inf")]},
    }
    training = {"batch_size": 2, "n_epochs": 1, "learning_rate": 1e-3, "num_workers": 1}
    cfg = OmegaConf.create(
        {
            "workflow": "pinc",
            "stage": "peft",
            "output_path": str(tmp_path / "out"),
            "ae_checkpoint": str(tmp_path / "base"),
            "model": tiny_ae_model_cfg(
                peft={"lora": lora}, loss_weights=weights, extra_loss_weights={"qspec": 0.0}
            ),
            "dataset": dataset,
            "training": training,
            "validation": {"eval_integrals": True},
            "logging": {"mode": "disabled"},
        }
    )
    r = PINCPEFTRunner(cfg, output_path=cfg.output_path)
    assert len(r.train_ds.files) == 2
    assert sum(jax.tree_util.tree_leaves(r.trainable)) == 2 * len(TARGETS)
    before = [np.asarray(v) for v in jax.tree_util.tree_leaves(local_view(r.dist, r.model))]
    batch = r.load_batch(r.train_ds, [0, 1] * r.dist.local_device_count, r.loader.read)
    model, _, logs = train_step(
        (shard_batch(r.dist, batch), r.ctx, jr.PRNGKey(0)), r.model, r.opt_state, r.spec
    )
    assert r.ds_spacing == 0.0625 and set(r.mode_std) == {"kyspec"}
    assert {"total", "df", "phi_int", "flux_int", "kyspec", "flux_int_l1"} <= set(logs)
    assert all(np.isfinite(float(v)) for v in logs.values())
    after = jax.tree_util.tree_leaves(local_view(r.dist, model))
    for m, a, b in zip(jax.tree_util.tree_leaves(r.trainable), before, after):
        assert m or np.array_equal(a, b)
    r.model = model
    metrics, _ = r.evaluate(1)
    assert set(r.evaluator.metric_keys) == set(metrics)
    assert r._val_score(metrics) == metrics["phi_int_mse"]
    assert {"df_rel_l2", "flux_int_l1", "kyspec_loss"} <= set(metrics)
    assert "qspec_loss" not in metrics and "flux_int_mse" not in metrics


def test_pinc_needs_one_ds(tmp_path):
    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    from neugk_jax.pinc.peft import uniform_ds

    _make_traj(tmp_path, "iteration_0", n_t=4)
    make_traj(tmp_path, "iteration_1", n_t=4, ds=np.float64(0.125))
    make_traj(tmp_path, "iteration_2", n_t=4, ds=None)

    def ds(trajs):
        return CycloneDataset(path=str(tmp_path), trajectories=trajs, backend=NumpyBackend())

    assert uniform_ds(ds("iteration_0")) == 0.0625
    for trajs in ("iteration_{0-1}", "iteration_2"):
        with pytest.raises(ValueError, match="'ds'"):
            uniform_ds(ds(trajs))
