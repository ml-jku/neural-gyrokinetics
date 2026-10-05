"""Real-data tests on the public CBC snapshot ``ml-jku/gyroswin_cbc_id_ood/iteration_8.h5``.

The h5 holds one spatial df snapshot with its geometry, conditions and GKW heat flux; no
potential is stored. Skipped without Hugging Face access.
"""

from __future__ import annotations

import importlib.util
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

needs_gyaradax = pytest.mark.skipif(
    importlib.util.find_spec("gyaradax") is None, reason="gyaradax not installed"
)


def _dataset(sample, stats=None, **kwargs):
    from neugk_jax.dataset import CycloneDataset, H5Backend

    norm = {"df": {"type": "zscore"}} if stats else None
    return CycloneDataset(
        path=sample.root,
        trajectories=[sample.name],
        backend=H5Backend(),
        normalization=norm,
        normalization_stats=stats,
        **kwargs,
    )


@pytest.fixture(scope="module")
def raw(hf_sample):
    import h5py

    with h5py.File(hf_sample.h5, "r") as f:
        return {
            "df": f["data/timestep_00000"][()],
            "meta": {k: v[()] for k, v in f["metadata"].items()},
        }


def test_h5_loader_reads_the_release_snapshot(hf_sample, raw):
    from neugk_jax.utils import separate_zf

    stats = hf_sample.stats
    ds = _dataset(hf_sample)
    assert ds.resolution == (32, 8, 16, 85, 32) and len(ds.files) == 1
    # one snapshot leaves no next-step target, so the index is empty
    assert len(ds) == 0
    meta = ds.metadata[0]
    assert float(meta["flux"][0]) == pytest.approx(float(raw["meta"]["fluxes"][0]))
    assert ds.get_ds(0) == pytest.approx(0.0625)
    s = ds.sample(0, 0)
    np.testing.assert_array_equal(s.df, raw["df"])
    assert float(s.timestep) == pytest.approx(float(raw["meta"]["timesteps"][0]))
    want = [raw["meta"][k][0] for k in ("density_grad", "ion_temp_grad", "q", "s_hat")]
    assert ds.conditions == ["dg", "itg", "q", "s_hat"]
    np.testing.assert_allclose(s.conditioning, want, rtol=1e-6)

    norm = _dataset(hf_sample, stats, separate_zf=True).sample(0, 0).df
    with open(stats, "rb") as f:
        full = pickle.load(f)["df"]["full"]
    ref = (separate_zf(raw["df"], axis=0) - full["mean"]) / full["std"]
    assert norm.shape == (4, *ds.resolution)
    np.testing.assert_allclose(norm, ref, rtol=1e-5, atol=1e-6)


def test_flux_integral_reproduces_gkw_flux(hf_sample, raw):
    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

    gt = precompute_geometry(_dataset(hf_sample).metadata[0]["geometry"])
    solve = jax.jit(flux_integral)
    phi, (pflux, eflux, _) = solve(gt, jnp.asarray(raw["df"]))
    gkw = float(raw["meta"]["fluxes"][0])
    assert float(eflux) == pytest.approx(gkw, rel=1e-4)
    # the solved-phi particle flux vanishes analytically
    assert abs(float(pflux)) <= 1e-6 * gkw
    assert phi.shape == (85, 16, 32) and np.isfinite(np.asarray(phi)).all()


@needs_gyaradax
def test_spectral_fields_sum_to_the_flux(hf_sample, raw):
    from neugk_jax.evaluate import metrics as m
    from neugk_jax.evaluate.integrals import gyaradax_spectral_fields

    ds = _dataset(hf_sample)
    geom = {k: np.asarray(v, np.float64) for k, v in ds.metadata[0]["geometry"].items()}
    df = raw["df"][None].astype(np.float64)
    (d,) = m.spectral_diagnostics(df, geom, ds.get_ds(0))
    assert d["qspec"].sum() == pytest.approx(float(raw["meta"]["fluxes"][0]), rel=1e-4)
    assert np.all(d["kyspec"] >= 0) and all(np.isfinite(d[k]).all() for k in d)
    # per-sample geometry gives the shared-geometry fields
    phi_s, ef_s = gyaradax_spectral_fields(df, geom)
    stacked = {k: np.stack([v, v]) for k, v in geom.items()}
    phi_p, ef_p = gyaradax_spectral_fields(np.concatenate([df, df]), stacked, per_sample=True)
    np.testing.assert_allclose(phi_p[1], phi_s[0], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(ef_p[0], ef_s[0], rtol=1e-10, atol=1e-14)
    same = m.time_averaged_spectral_metrics([d], [d])
    assert same["kyspec_pc"] == pytest.approx(1.0) and same["qspec_rl2"] == pytest.approx(0.0)
    assert same["zfphi_rl2"] == pytest.approx(0.0) and same["zf_energy_err"] == pytest.approx(0.0)
