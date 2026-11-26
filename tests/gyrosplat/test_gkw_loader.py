"""The GKW loader reads both metadata layouts: the flat one and the nested one."""
import pickle

import numpy as np
import torch

from neugk.gyrosplat.datasets.gkw import load_gkw_snapshot

RES = (4, 2, 4, 5, 4)


def _write(root, name, meta, field):
    d = root / f"{name}_ifft_realpotens"
    (d / "data").mkdir(parents=True)
    with open(d / "metadata.pkl", "wb") as fh:
        pickle.dump(meta, fh)
    field.astype(np.float32).tofile(d / "data" / "timestep_00003.bin")


def test_flat_and_nested_metadata_load_the_same_snapshot(tmp_path):
    rng = np.random.default_rng(0)
    field = rng.standard_normal((2, *RES))
    geometry = {
        "ints": np.full((1, RES[2]), 0.25),
        "krho": rng.random((1, RES[4])),
        "adiabatic": np.array(1.0),
    }
    flat = {"resolution": RES, "geometry": geometry}
    nested = {
        "schema": 3,
        "layout": {"resolution": RES},
        "geometry": {**geometry, "ds": np.array(0.25)},
    }
    _write(tmp_path, "flat", flat, field)
    _write(tmp_path, "nested", nested, field)

    a = load_gkw_snapshot(str(tmp_path), "flat", 3, device="cpu")
    b = load_gkw_snapshot(str(tmp_path), "nested", 3, device="cpu")

    assert a.fields[0].shape == (2, *RES)
    assert torch.equal(a.fields[0], b.fields[0])
    assert a.meta["ds"] == b.meta["ds"] == 0.25
    assert set(a.meta["geom"]) == set(b.meta["geom"])
    for k in a.meta["geom"]:
        assert torch.equal(a.meta["geom"][k], b.meta["geom"][k])
