"""Every ladder control runs on the synthetic snapshot and returns a scored row."""
from pathlib import Path

import pytest

from neugk.gyrosplat.eval.ablation import CONTROLS, control_config, run_control
from neugk.gyrosplat.backends.synthetic import SyntheticBackend
from neugk.gyrosplat.config import load_config
from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot

CFG = str(Path(__file__).parents[2] / "configs/gyrosplat/recipe/synthetic.yaml")


def backend():
    snap = make_synthetic_snapshot((8, 4, 6, 16, 16), seed=0)
    return SyntheticBackend(snap, band=[1, 2, 3, 4], device="cpu")


def test_unknown_control_is_refused():
    with pytest.raises(ValueError):
        control_config(load_config(CFG), "not_a_control")


def test_controls_apply_their_overrides():
    cfg = load_config(CFG)
    assert control_config(cfg, "noky").carriers.kind == "none"
    assert control_config(cfg, "fp32match").n_equiv == 2168
    assert control_config(cfg, "s7_pack").storage_bits == 16
    assert control_config(cfg, "all_gabor").carriers.fraction == 1
    assert control_config(cfg, "deployed") == cfg


@pytest.mark.parametrize(
    "control",
    [
        "deployed",
        "floor",
        "gabor_init",
        "field_solve",
        "all_gabor",
        "all_gabor_full",
        "s7_pack",
        "noky_full",
    ],
)
def test_control_runs_and_scores(control):
    cfg = load_config(
        CFG, ["verbose=false", "fit.steps=10", "physics.epochs=2", "gn.steps=2"]
    )
    if control in ("noky_full",):
        cfg = load_config(
            CFG, ["verbose=false", "fit.steps=10", "physics.epochs=2", "gn.steps=2"]
        )
    out = run_control(control, backend(), cfg, device="cpu")
    assert out["control"] == control and len(out["rows"]) == (
        2 if control == "deployed" else 1
    )
    row = out["rows"][-1]
    assert row["psnr_paper_all"] > 0 and row["CR"] > 1
    if control in ("all_gabor", "all_gabor_full"):
        assert out["n_atoms"][0] == out["n_carriers"][0] > 0
    if control == "field_solve":
        assert out["config"]["carriers"]["cofit"] == "none"
    assert set(CONTROLS) >= {"floor", "deployed", "flux_refinement_all_splats"}
