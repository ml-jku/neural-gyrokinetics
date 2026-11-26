"""The tables of the paper from the stored rows, pinned to the printed values."""
import glob
import json
import os
from collections import Counter

import pytest

from neugk.gyrosplat.eval.paper import (
    dispersion_table,
    families_table,
    ladder_table,
    load_rows,
    medians,
    package_rows,
    per_class_table,
    quantile,
    rows_of,
    sign_flips,
    storage_table,
    transfer_table,
    wavelet_table,
)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "neugk/gyrosplat/results")
ROWS = load_rows(os.path.join(RESULTS, "rows/gkw_rows.jsonl"))
CONFIGS = [
    os.path.join(ROOT, "configs/gyrosplat/recipe", f) for f in ("gkw_302x.yaml", "gkw_142x.yaml")
]


def printed(m, flux_digits=4):
    return (
        round(m["psnr_f"], 2),
        round(m["psnr_phi"], 2),
        float(f"{m['flux_rel_l1']:.3g}")
        if m["flux_rel_l1"] < 1e-3
        else round(m["flux_rel_l1"], flux_digits),
        round(m["kappa_rel"], 3),
    )


def test_rows_cover_the_roster():
    assert len(ROWS) == 2106
    assert len({r["snapshot"] for r in ROWS}) == 39
    classes = Counter(
        r["class"] for r in ROWS if r["method"] == "gyrosplat" and r["tier"] == 302
    )
    assert classes == {"streamer": 12, "mixed": 12, "strong_zonal": 15}
    groups = Counter(
        (r["tier"], r["method"] or (r["recipe"], r["stage"])) for r in ROWS
    )
    assert len(groups) == 54 and set(groups.values()) == {39}


@pytest.mark.parametrize(
    "tier,method,expected",
    [
        (302, "gyrosplat", (37.59, 27.71, 5.14e-7, 0.985)),
        (142, "gyrosplat", (38.20, 29.24, 4.49e-8, 0.971)),
        (302, "fourier", (35.90, 23.62, 0.2104, 3.659)),
        (142, "splats", (41.43, 9.60, 0.1608, 0.961)),
        (302, "autoencoder", (35.49, 15.42, 0.1308, 0.864)),
        (142, "sz3", (38.42, -12.23, 0.5367, 4.349)),
        (302, "wavelet_haar_objective", (22.15, -11.49, 8.16e-8, 4.682)),
        (142, "fourier_objective", (37.38, 27.73, 0.0014, 2.582)),
    ],
)
def test_table_1_and_3_medians_match_the_paper(tier, method, expected):
    assert printed(medians(rows_of(ROWS, tier, method))) == expected


@pytest.mark.parametrize(
    "recipe,stage,cr,expected",
    [
        ("importance_init", 0, 302.2, (39.06, 8.95, 0.4022, 0.986)),
        ("deployed", 0, 302.2, (37.59, 27.73, 1.38e-4, 0.984)),
        ("deployed", 1, 302.2, (37.59, 27.71, 5.14e-7, 0.985)),
        ("s7_pack", 0, 604.4, (37.59, 27.39, 0.0016, 0.985)),
        ("fp32match", 0, 604.7, (36.77, 25.37, 1.42e-4, 1.043)),
        ("noky_full", 0, 302.2, (35.43, -26.81, 0.0357, 1.739)),
        ("all_gabor_full", 0, 302.3, (36.86, 10.22, 7.53e-8, 1.347)),
    ],
)
def test_ladder_medians_match_the_paper(recipe, stage, cr, expected):
    m = medians(rows_of(ROWS, 302, recipe=recipe, stage=stage))
    assert round(m["CR"], 1) == cr
    assert printed(m) == expected


def test_sign_flips_per_class_and_quartiles_match_the_paper():
    assert sign_flips(rows_of(ROWS, 302, "wavelet_db8")) == 3
    assert sign_flips(rows_of(ROWS, 302, "wavelet_haar")) == 35
    assert sign_flips(rows_of(ROWS, 142, "wavelet_db4")) == 38
    assert sign_flips(rows_of(ROWS, 302, "zfp")) == 5
    assert sign_flips(rows_of(ROWS, 302, "gyrosplat")) == 0
    ours = rows_of(ROWS, 302, "gyrosplat")
    streamer = medians([r for r in ours if r["class"] == "streamer"])
    zonal = medians([r for r in ours if r["class"] == "strong_zonal"])
    assert printed(streamer) == (37.21, 26.91, 1.35e-6, 1.037)
    assert printed(zonal) == (37.59, 30.79, 3.41e-7, 0.965)
    psnr = [r["psnr_f"] for r in ours]
    assert (round(quantile(psnr, 0.25), 2), round(quantile(psnr, 0.75), 2)) == (
        37.08,
        38.35,
    )
    flux = [r["flux_rel_l1"] for r in ours]
    assert (
        round(quantile(flux, 0.25) * 1e7, 2) == 0.67
        and round(quantile(flux, 0.75) * 1e7, 2) == 24.06
    )
    assert round(medians(rows_of(ROWS, 302, "zfp"))["CR"], 1) == 304.8


def test_tables_render_in_both_forms():
    md = families_table(ROWS)
    assert "| GyroSplat (ours) | 37.59 | **27.71** | **5.14×10⁻⁷** | **0.985** |" in md
    tex = families_table(ROWS, latex=True, cite=True)
    assert (
        r"\boldsymbol{27.71}" in tex
        and r"5.14\!\times\!10^{-7}" in tex
        and r"\citep{pinc2026}" in tex
    )
    assert "604.4×" in ladder_table(ROWS, extended=True)
    assert "1.38×10⁻⁴" in ladder_table(ROWS)
    assert "35/39" in wavelet_table(ROWS)
    assert "**8.16×10⁻⁸**" in transfer_table(ROWS)
    assert "37.21" in per_class_table(ROWS)
    assert "5.14 [0.67, 24.06]×10⁻⁷" in dispersion_table(ROWS)
    accounting = storage_table(CONFIGS)
    assert "| 3,254 | 526 | 6,936 | 73,732 | 302.2× |" in accounting
    assert "| 6,912 | 1,119 | 14,745 | 156,673 | 142.2× |" in accounting


def test_package_rows_from_result_files(tmp_path):
    from neugk.gyrosplat.cli import main

    cfg = os.path.join(ROOT, "configs/gyrosplat/recipe/synthetic.yaml")
    common = [
        "--dataset",
        "synthetic",
        "--config",
        cfg,
        "--device",
        "cpu",
        "verbose=false",
    ]
    main(["fit", *common, "--out", str(tmp_path / "gyrosplat_302x_s.json")])
    main(
        [
            "baseline",
            "--method",
            "fourier_gn",
            *common,
            "--out",
            str(tmp_path / "fourier_gn_302x_s.json"),
        ]
    )
    main(
        [
            "ablate",
            "--control",
            "noky",
            *common,
            "--out",
            str(tmp_path / "ablation_noky_302x_s.json"),
        ]
    )
    rows = package_rows(sorted(glob.glob(str(tmp_path / "*.json"))))
    kinds = {(r["method"], r["recipe"], r["stage"]) for r in rows}
    assert {
        (None, "deployed", 0),
        ("gyrosplat", "deployed", 1),
        (None, "s7_pack", 0),
    } <= kinds
    assert {
        ("fourier", None, None),
        ("fourier_objective", None, None),
        (None, "noky", 0),
    } <= kinds
    assert all(r["tier"] == 302 for r in rows)
    for r in rows:
        assert (
            r["psnr_f"] is not None
            and r["flux_rel_l1"] is not None
            and r["kappa_rel"] is not None
        )
    with open(tmp_path / "gyrosplat_302x_s.json") as fh:
        assert json.load(fh)["config"]["carriers"]["share_envelopes"] is True


def test_figures_render():
    import matplotlib

    matplotlib.use("Agg")
    from neugk.gyrosplat.eval.paper import flux_spectrum_figure, scatter_figure

    assert scatter_figure(ROWS, 302).axes
    assert flux_spectrum_figure(ROWS, "iteration_280_t25", 302).axes
