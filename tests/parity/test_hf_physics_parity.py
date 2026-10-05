"""Flux integrals and spectral metrics on the public snapshot ``iteration_8.h5`` against torch."""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

needs_gyaradax = pytest.mark.skipif(
    importlib.util.find_spec("gyaradax") is None, reason="gyaradax not installed"
)


@pytest.fixture(scope="module")
def sample(hf_sample):
    from neugk_jax.dataset import CycloneDataset, H5Backend

    ds = CycloneDataset(path=hf_sample.root, trajectories=[hf_sample.name], backend=H5Backend())
    return (np.asarray(ds.sample(0, 0).df, np.float32), ds.metadata[0]["geometry"], ds.get_ds(0))


def test_flux_integral_matches_torch(sample):
    import torch
    from neugk.physics.integrals import FluxIntegral

    from neugk_jax.evaluate.integrals import flux_integral, precompute_geometry

    df, geom, _ = sample
    core = jax.jit(flux_integral)
    gt = precompute_geometry(geom)
    j_phi, (j_pf, j_ef, _) = core(gt, jnp.asarray(df))
    _, (j_pf2, j_ef2, _) = core(gt, jnp.asarray(df), j_phi)

    integ = FluxIntegral(real_potens=True)
    g_t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32)[None] for k, v in geom.items()}
    t_phi_in = torch.from_numpy(np.asarray(j_phi))[None]
    with torch.no_grad():
        t_phi, (t_pf, t_ef, _) = integ(g_t, torch.from_numpy(df)[None])
        _, (t_pf2, t_ef2, _) = integ(g_t, torch.from_numpy(df)[None], t_phi_in)
    t_phi = t_phi[0].numpy()
    assert j_phi.shape == t_phi.shape
    assert np.linalg.norm(np.asarray(j_phi) - t_phi) <= 1e-5 * np.linalg.norm(t_phi)
    assert float(j_ef) == pytest.approx(float(t_ef[0]), rel=1e-5)
    assert float(j_ef2) == pytest.approx(float(t_ef2[0]), rel=1e-5)
    assert float(j_pf2) == pytest.approx(float(t_pf2[0]), rel=1e-4, abs=1e-6 * abs(float(t_ef[0])))
    assert abs(float(j_pf) - float(t_pf[0])) <= 1e-6 * abs(float(t_ef[0]))


@needs_gyaradax
def test_spectral_metrics_match_torch(sample, monkeypatch):
    import functools

    import torch
    from neugk.pinc.eval import metrics as tmetrics

    from neugk_jax.evaluate import metrics as jmetrics

    # field-line sums (gkw kykx spectra), also where the torch tree still takes the mid slice
    monkeypatch.setattr(
        tmetrics, "diagnostics", functools.partial(tmetrics.diagnostics, aggregate="mean")
    )

    df, geom, ds_val = sample
    rng = np.random.default_rng(0)
    # four pseudo snapshots around the real one, and a perturbed "prediction" of each
    gt = np.stack([df * (1 + 0.05 * i) for i in range(4)]).astype(np.float64)
    pred = gt + 0.01 * gt.std() * rng.standard_normal(gt.shape)
    geom64 = {k: np.atleast_1d(np.asarray(v, np.float64)) for k, v in geom.items()}
    geom_t = {k: torch.as_tensor(v) for k, v in geom64.items()}

    def torch_diags(dfs):
        return [
            tmetrics.spectral_diagnostics(torch.as_tensor(d)[None], geom_t, ds=ds_val) for d in dfs
        ]

    t_gt = torch_diags(gt)
    j_gt = jmetrics.spectral_diagnostics(gt, geom64, ds_val)
    tam = tmetrics.time_averaged_spectral_metrics(torch_diags(pred), t_gt)
    jam = jmetrics.time_averaged_spectral_metrics(
        jmetrics.spectral_diagnostics(pred, geom64, ds_val), j_gt
    )
    keys = [f"{s}_{m}" for s in ("kyspec", "qspec") for m in ("pc", "sc", "rl2", "rl1", "wd")]
    keys += ["zfphi_rl2", "zfflow_rl2", "zfshear_rl2", "zf_energy_err"]
    for k in keys:
        rel = 1e-3 if k.startswith("qspec_") else 1e-4
        assert jam[k] == pytest.approx(float(tam[k]), rel=rel, abs=1e-10), k
    q_t = np.stack([np.asarray(d["qspec"]) for d in t_gt]).mean(0)
    q_j = np.stack([d["qspec"] for d in j_gt]).mean(0)
    mask = np.abs(q_j) > 1e-12 * np.abs(q_j).max()
    np.testing.assert_allclose(q_t[mask], q_j[mask], rtol=1e-4)
