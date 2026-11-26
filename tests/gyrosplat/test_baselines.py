"""The baselines on the synthetic snapshot: Fourier top-K with and without the objective,
PCA, the neural field, and (when PyWavelets is installed) the wavelet."""
import importlib

import pytest

from neugk.gyrosplat.backends.synthetic import SyntheticBackend
from neugk.gyrosplat.baselines import run_baseline
from neugk.gyrosplat.config import load_config
from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot

SHAPE = (6, 4, 5, 12, 12)


def _setup(*overrides):
    snap = make_synthetic_snapshot(SHAPE, seed=5)
    backend = SyntheticBackend(snap, band=[1, 2, 3])
    cfg = load_config(
        None,
        ["gn.steps=4", "baselines.objective_warmup=20", "verbose=false"]
        + list(overrides),
    )
    return backend, cfg


def test_fourier_and_objective_reduce_flux_error():
    backend, cfg = _setup()
    out = run_baseline(
        "fourier_gn", backend, cfg, budget=600, device="cpu", log=lambda *a: None
    )
    plain, gn = out["rows"]
    assert gn["flux_relL1_band"][0] < 0.5 * plain["flux_relL1_band"][0]
    assert plain["CR"] > 1
    full = run_baseline(
        "fourier",
        backend,
        cfg,
        budget=2 * backend.gt[0].numel(),
        device="cpu",
        log=lambda *a: None,
    )
    assert full["rows"][0]["psnr_paper_all"] > 100  # every coefficient kept: exact


def test_pca_takes_the_closest_rank():
    import math

    backend, cfg = _setup()
    out = run_baseline(
        "pca", backend, cfg, budget=900, device="cpu", log=lambda *a: None
    )
    m, n = SHAPE[0] * SHAPE[1] * SHAPE[2], SHAPE[3] * SHAPE[4]
    sizes = {r: 2 * (r * (m + n) + n + r) for r in range(1, 6)}
    closest = min(sizes.values(), key=lambda v: abs(math.log(v / 900)))
    assert out["achieved_reals"][0] == closest
    assert out["rows"][0]["psnr_paper_all"] > 10
    with pytest.raises(ValueError):
        run_baseline("pca_gn", backend, cfg, budget=900, device="cpu")


@pytest.mark.skipif(
    importlib.util.find_spec("pywt") is None, reason="PyWavelets not installed"
)
def test_wavelet_keeps_single_coefficients():
    backend, cfg = _setup("baselines.wavelet=haar")
    snr = []
    for budget in (301, 1501):
        out = run_baseline(
            "wavelet", backend, cfg, budget=budget, device="cpu", log=lambda *a: None
        )
        assert out["achieved_reals"][0] == budget
        snr.append(out["rows"][0]["H_snr_dB"][0])
    assert snr[1] > snr[0]


def test_plain_splats_spend_the_budget_on_16_real_atoms():
    backend, cfg = _setup("fit.steps=20")
    out = run_baseline(
        "splats", backend, cfg, budget=800, device="cpu", log=lambda *a: None
    )
    assert out["n_atoms"] == [800 // 16]
    assert out["rows"][-1]["stored_reals"]["atoms"] == 16 * (800 // 16)


def test_neural_field_fits_the_budget_and_the_field():
    from neugk.gyrosplat.baselines.neural_field import parameter_count, solve_width

    backend, cfg = _setup(
        "baselines.nf.compile=false", "baselines.nf.epochs=6", "baselines.nf.batch=256"
    )
    width, n_params = solve_width(SHAPE, 1500, cfg.baselines.nf.layers)
    assert n_params <= 1500 < parameter_count(SHAPE, width + 5, cfg.baselines.nf.layers)
    out = run_baseline(
        "neural_field", backend, cfg, budget=1500, device="cpu", log=lambda *a: None
    )
    assert out["achieved_reals"][0] == n_params + 4
    assert out["rows"][0]["H_snr_dB"][0] > 3.0
