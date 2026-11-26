"""The byte-matched tables and the appendix experiments from results/experiments, pinned to
the printed values of the paper."""
import os
from collections import Counter

import pytest
import torch

from neugk.gyrosplat.eval.diagnostics import VIEWS, flux_views, relative_l1, view_errors
from neugk.gyrosplat.eval.paper import experiments as X
from neugk.gyrosplat.eval.paper import load_rows

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "neugk/gyrosplat/results")
ROWS = load_rows(os.path.join(RESULTS, "rows/gkw_rows.jsonl"))
EXP = X.load_experiments(os.path.join(RESULTS, "experiments"))


def test_files_cover_the_experiments():
    sizes = {k: len(v) for k, v in EXP.items() if isinstance(v, list)}
    assert sizes == {
        "seeds": 1248,
        "byte_matched": 468,
        "rate_distortion": 5928,
        "views": 663,
        "heldout_modes": 18,
        "latent_refinement": 576,
        "damping": 132,
        "timing": 252,
        "operator_cost": 9,
    }
    seeds = Counter((r["method"], r["tier"]) for r in EXP["seeds"])
    assert seeds[("gyrosplat", 302)] == seeds[("neural_field_physics", 142)] == 5 * 39
    assert seeds[("splats", 142)] == seeds[("neural_field", 302)] == 3 * 39
    assert len(X.load_restarts(EXP)) == 8


def values(tier, method):
    v = X._summaries(EXP, ROWS, tier, method)
    return tuple(
        v[k]["value"] for k in ("psnr_f", "psnr_phi", "flux_rel_l1", "kappa_rel")
    )


@pytest.mark.parametrize(
    "tier,method,expected",
    [
        (302, "gyrosplat", ("37.63", "27.69", "5.2e-07", "0.984")),
        (142, "gyrosplat", ("38.21", "29.24", "3.7e-08", "0.973")),
        (142, "neural_field_physics", ("38.16", "27.56", "0.105", "0.892")),
        (302, "fourier", ("35.25", "21.36", "0.289", "4.290")),
        (302, "wavelet_haar_objective", ("12.90", "-15.08", "1.8e-07", "15.814")),
        (302, "autoencoder", ("35.49", "15.42", "0.131", "0.864")),
    ],
)
def test_table_1_and_3_at_equal_bytes(tier, method, expected):
    got = values(tier, method)
    printed = (
        f"{got[0]:.2f}",
        f"{got[1]:.2f}",
        f"{got[2]:.1e}" if got[2] < 1e-3 else f"{got[2]:.3f}",
        f"{got[3]:.3f}",
    )
    assert printed == expected


def test_seed_variation_and_paired_gain():
    s = X.summary(X.samples(EXP, ROWS, 302, "gyrosplat"), "psnr_phi")
    assert (s["seeds"], round(s["std"], 2)) == (5, 0.04)
    s = X.summary(X.samples(EXP, ROWS, 142, "neural_field_physics"), "psnr_phi")
    assert round(s["std"], 2) == 0.25
    g = X.paired_gain(EXP, 302)
    assert round(g["median"], 2) == 1.16 and [round(x, 2) for x in g["interval"]] == [
        0.75,
        1.53,
    ]
    assert (g["positive_snapshots"], g["snapshots"], g["trajectories"]) == (35, 39, 27)
    g = X.paired_gain(EXP, 142)
    assert round(g["median"], 2) == 2.13 and g["positive_snapshots"] == 38


def test_rate_distortion():
    v = X.rd_summary(EXP, "gyrosplat", 1000, "psnr_phi")
    assert (round(v["value"], 2), round(v["std"], 2), v["seeds"]) == (22.24, 0.30, 3)
    assert (
        round(X.rd_summary(EXP, "neural_field_physics", 800, "psnr_phi")["value"], 2)
        == 23.86
    )
    f = X.rd_summary(EXP, "fourier_objective", 100, "flux_rel_l1")
    assert f["std"] is None and round(f["value"], 3) == 0.002
    assert "| 1000× | 89,188 | 999.3 |" in X.rd_budget_table(EXP)


def test_diagnostics_outside_the_objective():
    assert round(X.view_medians(EXP, 302, "gyrosplat")["Q_vpar_mu"], 3) == 0.433
    assert (
        round(X.view_medians(EXP, 302, "fourier_objective")["Q_pointwise"], 3) == 2.874
    )
    table = X.heldout_table(EXP)
    assert (
        "| Even | 2.5×10⁻⁸ | 0.212 | -0.06 |" in table
        and "| Odd | 1.4×10⁻⁷ | 0.334 | -0.09 |" in table
    )
    table = X.restart_table(EXP)
    assert "| GyroSplat | 2.4 | 22 | 8, 1 non-finite |" in table
    assert "| NF + physics | 12.2 | 7 | 8 |" in table


def test_frozen_decoders_stability_and_cost():
    med = X.latent_medians(EXP)
    assert round(med[("autoencoder", "adam")]["psnr_phi"], 2) == 26.57
    assert round(med[("autoencoder", "gauss_newton")]["psnr_phi"], 2) == 10.20
    assert (
        "| VQ-VAE | 22.13 → 23.34 | 0.083 → 0.051 | 34.41 → 34.95 |"
        in X.latent_steps_table(EXP)
    )
    assert (
        "| 302× | 5.1×10⁻⁷ | 3.6×10⁻⁵ | 39/39 | 0/39 | 36/39 | 21.45 |"
        in X.final_fit_table(ROWS)
    )
    assert (
        "| 1.0×10⁻⁴ | 8 | 5.7×10⁻⁷ | 3.6×10⁻⁵ | -0.005 | -0.957 |"
        in X.damping_table(EXP)
    )
    assert "| GyroSplat | 302× | 18 | 169 [167–172] |" in X.timing_table(EXP)
    assert "| Final GN refinement | 11 | 24 |" in X.stage_table(EXP)
    ops = X.operator_cost(EXP)
    assert (
        round(ops["geometry_bytes"] / 1e6, 2) == 3.61
        and round(ops["jacobian_bytes_142"] / 1e6, 2) == 3.89
    )
    assert "| GyroSplat | 36% | 18% | 34% | 15% |" in X.relative_error_table(EXP, ROWS)


def test_every_table_renders_in_both_forms():
    tables = [
        lambda latex: X.main_table(EXP, ROWS, latex=latex),
        lambda latex: X.refinement_table(EXP, ROWS, latex=latex),
        lambda latex: X.db4_table(EXP, latex=latex),
        lambda latex: X.seed_dispersion_table(EXP, ROWS, latex=latex),
        lambda latex: X.regime_table(EXP, ROWS, latex=latex),
        lambda latex: X.bytes_table(EXP, latex=latex),
        lambda latex: X.rd_table(EXP, "kappa_rel", latex=latex),
        lambda latex: X.views_table(EXP, latex=latex),
        lambda latex: X.latent_table(EXP, ROWS, latex=latex),
    ]
    for make in tables:
        md, tex = make(False), make(True)
        assert (
            md.startswith("| ")
            and tex.startswith(r"\begin{tabular}")
            and r"\bottomrule" in tex
        )
    assert r"\boldsymbol{27.69}" in X.main_table(EXP, ROWS, latex=True)


def test_flux_views():
    torch.manual_seed(0)
    integrand = torch.randn(4, 3, 5, 6, 8)
    v = flux_views(integrand)
    total = integrand[..., 1:].double().sum()
    for k in ("Q_ky", "Q_kx", "Q_s", "Q_vpar_mu"):
        assert torch.allclose(v[k].sum(), total)
    assert v["Q_kx"].shape == (6,) and v["Q_vpar_mu"].shape == (4, 3)
    assert relative_l1(v["Q_s"], v["Q_s"]) == 0.0

    class Backend:  # the two methods view_errors needs
        integrand_gt = integrand

        def to_real(self, decoded):
            return [decoded]

        def readouts(self, field):
            return None, None, field

    err = view_errors(Backend(), integrand * 1.1)
    assert set(err) == set(VIEWS) and all(abs(e - 0.1) < 1e-6 for e in err.values())
