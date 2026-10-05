"""Evaluator tests on a tiny synthetic dataset.

Covers the AE / diffusion evaluators: metric keys (``*_rel_l2`` next to MSE and flux
errors), flux integrals of the denormalized df, spectral metrics, geometry validation,
fixed-shape padded batches and no retracing across epochs.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import make_traj
from omegaconf import OmegaConf

from neugk_jax.evaluate import AEEvaluator, DiffusionEvaluator
from neugk_jax.utils import TRACE_COUNTS


@pytest.fixture
def tiny_setup(tmp_path):
    """A synthetic single-trajectory dataset + a tiny Swin5DAE."""
    resolution = (4, 4, 4, 16, 8)
    make_traj(tmp_path, "iteration_0", n_t=4, resolution=resolution)
    from neugk_jax.dataset import CycloneDataset, NumpyBackend

    ds = CycloneDataset(
        path=str(tmp_path),
        split="train",
        trajectories="iteration_0",
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        backend=NumpyBackend(),
        separate_zf=False,
        normalization=None,
    )
    from neugk_jax.pinc import Swin5DAE

    ae = Swin5DAE(
        space=5,
        decouple_mu=True,
        dim=16,
        base_resolution=resolution,
        in_channels=2,
        out_channels=2,
        patch_size=[2, 0, 2, 4, 2],
        window_size=[2, 0, 2, 2, 2],
        depth=[1],
        num_heads=[2],
        num_layers=1,
        bottleneck_dim=8,
        bottleneck_depth=1,
        bottleneck_num_heads=2,
        merging_depth=1,
        unmerging_depth=1,
        merging_hidden_ratio=2.0,
        unmerging_hidden_ratio=2.0,
        hidden_mlp_ratio=2.0,
        key=jr.PRNGKey(0),
    )
    return ds, ae


class Scaled(eqx.Module):
    """Stand-in reconstruction model: ``factor * x``."""

    factor: float = eqx.field(static=True)

    def __call__(self, x, *, key=None, inference=True):
        return {"df": self.factor * x}


def _cfg(**validation):
    return OmegaConf.create({"validation": validation})


def test_ae_evaluator_runs(tiny_setup):
    ds, ae = tiny_setup
    ev = AEEvaluator(_cfg(eval_integrals=False), val_ds=ds, batch_size=1)
    metrics, _ = ev(ae, epoch=1)
    assert set(metrics) == {"df_mse", "df_rel_l2"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_ae_evaluator_integrals_plain_df(tiny_setup):
    """eval_integrals on a plain 2-channel df yields finite phi/flux integral metrics."""
    ds, ae = tiny_setup
    ev = AEEvaluator(_cfg(eval_integrals=True), val_ds=ds, batch_size=1)
    metrics, plots = ev(ae, epoch=1)
    for key in (
        "df_mse",
        "df_rel_l2",
        "phi_int_mse",
        "phi_int_rel_l2",
        "flux_int_mse",
        "flux_int_rel_err",
    ):
        assert np.isfinite(metrics[key]), key
    assert any(k.startswith("df ") for k in plots) and any(k.startswith("phi ") for k in plots)


def test_ae_integrals_use_denormalized_df(tmp_path):
    """Flux errors are those of the physical df: a 2x reconstruction gives 4x the flux."""
    from neugk_jax.dataset import CycloneDataset, NumpyBackend
    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

    resolution = (4, 4, 4, 16, 8)
    make_traj(tmp_path, "iteration_0", n_t=4, resolution=resolution)
    common = dict(path=str(tmp_path), trajectories="iteration_0", backend=NumpyBackend())
    raw = CycloneDataset(**common)
    std = 3.0
    ds = CycloneDataset(
        **common,
        normalization={"df": {"type": "zscore"}},
        normalization_stats={"df": {"full": {"mean": 0.0, "std": std}}},
    )
    gt = precompute_geometry(raw.metadata[0]["geometry"])
    solve = jax.jit(flux_integral)
    eflux = np.asarray([float(solve(gt, jnp.asarray(raw[i].df))[1][1]) for i in range(len(raw))])

    ev = AEEvaluator(_cfg(eval_integrals=True), val_ds=ds, batch_size=2)
    metrics, _ = ev(Scaled(2.0), epoch=1)
    assert metrics["df_rel_l2"] == pytest.approx(1.0, rel=1e-5)
    assert metrics["flux_int_rel_err"] == pytest.approx(3.0, rel=1e-3)
    assert metrics["flux_int_mse"] == pytest.approx(float(np.mean((3 * eflux) ** 2)), rel=1e-3)
    identity, _ = ev(Scaled(1.0), epoch=2)
    assert identity["flux_int_mse"] == pytest.approx(0.0, abs=1e-10)


def test_integrals_reject_incomplete_geometry(tiny_setup):
    ds, ae = tiny_setup
    del ds.metadata[0]["geometry"]["kxrh"]
    ev = AEEvaluator(_cfg(eval_integrals=True), val_ds=ds, batch_size=1)
    with pytest.raises(KeyError, match="kxrh"):
        ev(ae, epoch=1)


def test_ae_evaluator_spectral_metrics(tiny_setup):
    """eval_spectra=True produces finite time-averaged zonal-flow and
    spectral metrics (needs the 'ds' metadata entry)."""
    ds, ae = tiny_setup
    assert ds.get_ds(0) == 0.0625
    ev = AEEvaluator(_cfg(eval_spectra=True), val_ds=ds, batch_size=2)
    metrics, _ = ev(ae, epoch=1)
    for key in ("zfphi_rl2", "zf_energy_err", "kyspec_rl2", "qspec_rl1"):
        assert key in metrics, f"missing spectral metric {key}"
        assert np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"


def test_ae_evaluator_padded_last_batch_and_no_retrace(tiny_setup):
    """3 samples: the tail batch is padded and masked, and epochs reuse one trace."""
    ds, ae = tiny_setup
    ev = AEEvaluator(_cfg(eval_integrals=True), val_ds=ds, batch_size=2)
    bs, tail = ev.batch_size, 3 % ev.batch_size or ev.batch_size
    assert [p.indices.shape for p in ev.plans] == [(bs,)] * -(-3 // bs)
    assert ev.plans[-1].mask.tolist() == [1.0] * tail + [0.0] * (bs - tail)
    full = AEEvaluator(_cfg(eval_integrals=True), val_ds=ds, batch_size=1)
    # batch-size invariance only holds without reduced-precision matmuls
    with jax.default_matmul_precision("highest"):
        m1, _ = ev(ae, epoch=1)
        TRACE_COUNTS.clear()
        m2, _ = ev(ae, epoch=2)
        assert TRACE_COUNTS["ae_eval_step"] == 0 and TRACE_COUNTS["ae_target_integrals"] == 0
        ref, _ = full(ae, epoch=1)
    for k in m1:
        assert m1[k] == pytest.approx(m2[k]) and m1[k] == pytest.approx(ref[k], rel=1e-4), k


def test_diffusion_evaluator_samples_and_scores(tiny_setup):
    """Samples decode to df, are scored against the df targets and flux-aggregated per trajectory."""
    from neugk_jax.diffusion.dit import DiT

    ds, ae = tiny_setup
    grid = tuple(ae.bottleneck_grid_size)
    dit = DiT(
        z_dim=int(ae.bottleneck_dim),
        dim=16,
        grid_size=grid,
        depth=1,
        num_heads=2,
        n_cond=2,
        key=jr.PRNGKey(1),
    )
    cfg = _cfg(eval_integrals=True, eval_sample_steps=2, eval_n_samples=2)
    ev = DiffusionEvaluator(
        cfg,
        val_ds=ds,
        autoencoder=ae,
        latent_scale=1.0,
        cond_slots=np.asarray([0, 1]),
        batch_size=2,
    )
    metrics, plots = ev(dit, epoch=1)
    TRACE_COUNTS.clear()
    ev(dit, epoch=2)
    assert TRACE_COUNTS["diffusion_eval_step"] == 0
    for key in (
        "df_mse",
        "df_rel_l2",
        "avg_flux_rmse",
        "avg_flux_rel_err",
        "avg_flux_pred/iteration_0",
        "avg_flux_std/iteration_0",
    ):
        assert np.isfinite(metrics[key]), key
    assert metrics["avg_flux_gt/iteration_0"] == pytest.approx(ds.get_avg_flux(0))
    assert "avg_flux_UQ" in plots


def test_spectral_sums_split_over_processes_match_the_whole():
    """Per-process packed sums add up to the metrics of the whole trajectory set."""
    from neugk_jax.evaluate import metrics as m

    rng = np.random.default_rng(0)

    def diag():
        return {
            "kyspec": rng.random(4),
            "qspec": rng.random(4),
            "kxspec": rng.random(6),
            **{k: rng.standard_normal(6) for k in ("zfphi", "zfflow", "zfshear")},
        }

    pred = {f: [diag() for _ in range(5)] for f in (0, 2)}
    gt = {f: [diag() for _ in range(5)] for f in (0, 2)}
    whole = {f: m.time_averaged_spectral_metrics(pred[f], gt[f]) for f in (0, 2)}
    want = {k: np.mean([whole[f][k] for f in (0, 2)]) for k in whole[0]}
    parts = [
        {f: m.spectral_sums(pred[f][sl], gt[f][sl]) for f in (0, 2)}
        for sl in (slice(0, 2), slice(2, 5))
    ]
    packed = sum(m.pack_spectral_store(p, 3, 4) for p in parts)
    got = m.merged_spectral_metrics(m.unpack_spectral_store(packed, 4))
    assert set(got) == set(want)
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-10), k
