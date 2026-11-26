"""Storage must retain independently fitted carrier geometry."""
import zlib

import numpy as np
import pytest
import torch

from neugk.gyrosplat.model.atoms import AtomBank
from neugk.gyrosplat.baselines import budget_from_recipe
from neugk.gyrosplat.config import load_config
from neugk.gyrosplat.model.pack import packed_bytes, packed_parameters, stored_reals
from neugk.gyrosplat.recipe import Model, _accounting, budget
from neugk.gyrosplat.model.normalize import NormStats


def bank():
    # Two base atoms and one group of three carriers with DIFFERENT fitted envelopes.
    return AtomBank(
        torch.arange(25).reshape(5, 5).float(),
        torch.arange(30).reshape(5, 6).float(),
        torch.arange(15).reshape(5, 3).float(),
        torch.arange(10).reshape(5, 2).float(),
        torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]),
        n_base=2,
        shape=(2, 2, 2, 2, 2),
    )


@pytest.mark.parametrize("bits,dtype", [(32, np.float32), (16, np.float16)])
def test_group_hint_cannot_discard_independent_envelopes(bits, dtype):
    b = bank()
    blob = packed_parameters([b], bits, tied_k=3)
    assert blob == packed_parameters([b], bits)
    values = np.frombuffer(zlib.decompress(blob), dtype=dtype)
    offset = 0
    for section in (slice(0, 2), slice(2, 5)):
        for field in (b.mu, b.L_phys_raw, b.L_vel_raw, b.amps, b.ky):
            expected = field.detach()[section].numpy().astype(dtype).ravel()
            np.testing.assert_array_equal(
                values[offset : offset + expected.size], expected
            )
            offset += expected.size
    assert offset == values.size == 5 * 17
    assert len(zlib.decompress(blob)) == stored_reals([b], bits=bits)["total"] * 4
    assert packed_bytes([b], bits, tied_k=3) == len(blob)


def test_plain_base_wavenumber_omission_is_checked():
    b = bank()
    assert stored_reals([b], reals_per_atom=16)["total"] == 2 * 16 + 3 * 17
    blob = packed_parameters([b], reals_per_atom=16)
    assert len(zlib.decompress(blob)) == (2 * 16 + 3 * 17) * 4
    b.ky.data[0] = 1
    with pytest.raises(ValueError, match="nonzero"):
        packed_parameters([b], reals_per_atom=16)
    with pytest.raises(ValueError, match="17 reals"):
        stored_reals([b], reals_per_carrier=5)


@pytest.mark.parametrize(
    "preset,base,carriers,reals",
    [
        (302, 3254, 3682, 73728),
        (142, 6912, 7833, 156669),
    ],
)
def test_preset_sizes_and_shared_envelope_budgets(preset, base, carriers, reals):
    from pathlib import Path

    cfg = load_config(
        str(Path(__file__).parents[2] / f"configs/gyrosplat/recipe/gkw_{preset}x.yaml")
    )
    nb, groups = budget(cfg)
    assert (nb, groups * cfg.carriers.tied_k) == (base, carriers)
    assert budget_from_recipe(cfg) == reals + 4


def test_accounting_includes_normalization_and_does_not_mutate_model():
    from types import SimpleNamespace

    b = bank()
    before = {k: v.clone() for k, v in b.state_dict().items()}
    model = Model([b], [NormStats(0.0, 1.0, 0.0, 1.0)])
    row = _accounting(model, SimpleNamespace(raw_reals=10000), load_config(None))
    assert row["stored_reals"] == {"atoms": 85.0, "normalization": 4, "total": 89.0}
    assert row["CR"] == 10000 / 89
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in before.items())
