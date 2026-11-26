"""Shared geometry must be a single trainable parameter and survive storage exactly."""
from pathlib import Path
import zlib

import pytest
import torch

from neugk.gyrosplat.model.atoms import make_grid, subgrids
from neugk.gyrosplat.backends.synthetic import SyntheticBackend
from neugk.gyrosplat.baselines import budget_from_recipe
from neugk.gyrosplat.model.carriers import graft_tied_gabor
from neugk.gyrosplat.config import CarrierConfig, load_config
from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot
from neugk.gyrosplat.model.init import field_init
from neugk.gyrosplat.model.pack import load_model, packed_parameters, save_model, stored_reals
from neugk.gyrosplat.recipe import compress
from neugk.gyrosplat.model.synth import _synth_block, render, to_separable
from neugk.gyrosplat.model.train import routed_cofit
from neugk.gyrosplat.model.normalize import normalize

SHAPE = (4, 3, 4, 6, 8)


def pair():
    snap = make_synthetic_snapshot(SHAPE, seed=8)
    grid = make_grid(SHAPE)
    base = field_init(3, to_separable(snap.fields[0]), grid, SHAPE, seed=3)
    cfg = CarrierConfig(tied_k=3, flux_gate=0.0, share_envelopes=False)
    independent = graft_tied_gabor(base, grid, 2, cfg, [1, 2, 3], seed=5)
    torch.manual_seed(9)
    independent.amps.data.normal_()
    shared = independent.share_carrier_envelopes(3)
    return snap, independent, shared, subgrids(grid, SHAPE)


def differentiable_render(b, vel, phys):
    V, Pre, Pim = b.factors(vel, phys)
    re, im = _synth_block(V, Pre, Pim, b.amps[:, 0], b.amps[:, 1])
    return torch.stack([re, im], -1)


def test_shared_geometry_gradient_is_sum_over_carriers():
    _, independent, shared, (vel, phys) = pair()
    expected = differentiable_render(independent, vel, phys)
    actual = differentiable_render(shared, vel, phys)
    assert torch.equal(actual, expected)
    torch.manual_seed(1)
    weights = torch.randn_like(actual)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    assert shared.mu.shape[0] == 5 and shared.N == 9
    for name in ["mu", "L_phys_raw", "L_vel_raw"]:
        full = getattr(independent, name).grad
        reduced = torch.cat([full[:3], full[3:].reshape(2, 3, -1).sum(1)])
        torch.testing.assert_close(
            getattr(shared, name).grad, reduced, rtol=3e-5, atol=1e-5
        )
    torch.testing.assert_close(shared.amps.grad, independent.amps.grad)


def test_routed_training_keeps_one_geometry_and_roundtrips(tmp_path):
    snap, _, shared, (vel, phys) = pair()
    backend = SyntheticBackend(snap, band=[1, 2, 3])
    _, stats = normalize(snap.fields[0], "zonal")
    start = shared.mu.detach().clone()
    routed_cofit(
        [shared],
        [stats],
        [vel],
        [phys],
        backend,
        CarrierConfig(share_envelopes=True),
        epochs=3,
        chunk=2048,
    )
    assert not torch.equal(start, shared.mu)
    assert shared.geometry_carrier_mask.tolist() == [False] * 3 + [True] * 2
    for p in [shared.mu, shared.L_phys_raw, shared.L_vel_raw]:
        expanded = shared.expanded_geometry(p)[3:].reshape(2, 3, -1)
        assert torch.equal(expanded, expanded[:, :1].expand_as(expanded))
    assert torch.count_nonzero(shared.ky[:3]) == 0
    cfg = load_config(None, ["carriers.share_envelopes=true"])
    path = tmp_path / "shared.pt"
    save_model(str(path), [shared], [stats], cfg)
    banks, _, _, _ = load_model(str(path))
    loaded = banks[0]
    assert loaded.envelope_group_size == 3 and loaded.mu.shape[0] == 5
    assert torch.equal(render(shared, vel, phys), render(loaded, vel, phys))
    n = sum(p.numel() for p in shared.parameters())
    assert n == 3 * 17 + 2 * (14 + 3 * 3)
    assert stored_reals([shared])["total"] == n
    assert len(zlib.decompress(packed_parameters([shared]))) == 4 * n


def test_lossy_compaction_is_rejected_and_subset_preserves_render():
    _, independent, shared, (vel, phys) = pair()
    independent.mu.data[4, 0] += 0.1
    with pytest.raises(ValueError, match="divergent"):
        independent.share_carrier_envelopes(3)
    ids = torch.tensor([0, 4, 7])
    subset = shared.subset(ids)
    expected = shared.amps.detach().clone()
    shared.amps.data.zero_()
    shared.amps.data[ids] = expected[ids]
    assert torch.equal(render(subset, vel, phys), render(shared, vel, phys))


@pytest.mark.parametrize("preset,count", [(302, 73732), (142, 156673)])
def test_shared_presets_count_one_envelope_per_group(preset, count):
    cfg = load_config(
        str(Path(__file__).parents[2] / f"configs/gyrosplat/recipe/gkw_{preset}x.yaml"),
        ["carriers.share_envelopes=true"],
    )
    assert budget_from_recipe(cfg) == count


def test_shared_recipe_all_stages_and_quantization():
    snap = make_synthetic_snapshot(SHAPE, seed=6)
    backend = SyntheticBackend(snap, band=[1, 2, 3, 4])
    cfg = load_config(
        str(Path(__file__).parents[2] / "configs/gyrosplat/recipe/synthetic.yaml"),
        [
            "carriers.share_envelopes=true",
            "fit.steps=10",
            "physics.epochs=2",
            "carriers.routed_steps=2",
            "carriers.routed_steps_hard=3",
            "gn.steps=3",
            "verbose=false",
        ],
    )
    model = compress(backend, cfg, device="cpu")
    b = model.banks[0]
    assert b.N == 58 and b.mu.shape[0] == 36
    assert [r["tag"] for r in model.info["rows"]] == [
        "fit",
        "+ gauss-newton",
        "+ 16-bit parameters",
    ]
    assert all(
        r["storage_accounting"] == "shared_envelopes_v1" for r in model.info["rows"]
    )
    assert (
        model.info["rows"][-1]["stored_reals"]["total"]
        == (25 * 17 + 11 * (14 + 9)) / 2 + 4
    )
