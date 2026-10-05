"""Dataset shape and behaviour tests on a synthetic kvikio-style directory."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from neugk_jax.dataset import CycloneDataset, NumpyBackend


def _make_synthetic_traj(
    root: Path, name: str, *, n_t: int, resolution, drop_keys=(), extra_meta=None
):
    """Write a fake trajectory directory: metadata.pkl + N data/.bin files."""
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    df_shape = (2, *resolution)
    # write per-timestep .bin (float32, contiguous, size = prod(df_shape))
    for t in range(n_t):
        arr = rng.standard_normal(df_shape).astype(np.float32)
        arr.tofile(data / f"timestep_{t:05d}.bin")

    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64),
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], dtype=np.float32),
        "density_grad": np.array([1.1], dtype=np.float32),
        "s_hat": np.array([0.8], dtype=np.float32),
        "q": np.array([1.4], dtype=np.float32),
        "resolution": np.array(resolution),
        "geometry": {
            k: np.ones((1,), dtype=np.float64)
            for k in (
                "krho",
                "ints",
                "intmu",
                "intvp",
                "vpgr",
                "mugr",
                "bn",
                "efun",
                "rfun",
                "bt_frac",
                "parseval",
                "mas",
                "tmp",
                "d2X",
                "signz",
                "signB",
                "kxrh",
                "little_g",
            )
        },
        "df_mean": np.zeros(1, dtype=np.float32),
        "df_std": np.ones(1, dtype=np.float32),
        "df_min": np.full(1, -3.0, dtype=np.float32),
        "df_max": np.full(1, 3.0, dtype=np.float32),
    }
    for k in drop_keys:
        meta.pop(k, None)
    meta.update(extra_meta or {})
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    return traj


@pytest.fixture
def synthetic_dir(tmp_path):
    resolution = (2, 2, 2, 4, 2)  # tiny (vp, mu, s, x, y)
    _make_synthetic_traj(tmp_path, "iteration_001", n_t=8, resolution=resolution)
    _make_synthetic_traj(tmp_path, "iteration_002", n_t=8, resolution=resolution)
    return tmp_path, resolution


def test_dataset_construction(synthetic_dir):
    root, res = synthetic_dir
    ds = CycloneDataset(
        path=str(root),
        split="train",
        trajectories=["iteration_001", "iteration_002"],
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        offset=0,
        bundle_seq_length=1,
        backend=NumpyBackend(),
    )
    assert len(ds.files) == 2
    assert ds.resolution == res
    # per-file samples = n_t - bundle_seq_length*2 + 1 = 8 - 2 + 1 = 7
    assert len(ds) == 7 * 2


def test_getitem_shapes(synthetic_dir):
    root, res = synthetic_dir
    ds = CycloneDataset(
        path=str(root),
        split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        normalization={"df": {"type": "zscore"}},
        normalization_scope="dataset",
        backend=NumpyBackend(),
    )
    s = ds[0]
    assert s.df.shape == (2, *res)
    assert s.df.dtype == np.float32
    assert s.conditioning.shape == (4,)
    assert s.itg.shape == ()  # scalar after squeeze
    assert s.flux.shape == ()
    assert s.timestep.shape == ()
    assert int(s.file_index) == 0


def test_stack_fields_batches(synthetic_dir):
    from neugk_jax.training.data import stack_fields

    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root),
        split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        backend=NumpyBackend(),
    )
    batch = stack_fields(
        [ds[i] for i in range(3)], ("df", "conditioning", "flux", "file_index", "phi")
    )
    assert batch["df"].shape == (3, 2, *ds.resolution)
    assert batch["conditioning"].shape == (3, 4)
    assert batch["flux"].shape == (3,)
    assert batch["file_index"].dtype == np.int32 and "phi" not in batch


def test_normalize_denormalize_roundtrip(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root),
        split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        normalization={"df": {"type": "zscore"}},
        normalization_scope="dataset",
        backend=NumpyBackend(),
    )
    # craft a synthetic field of known mean/std and roundtrip through the table
    x = np.full((2, *ds.resolution), 5.0, dtype=np.float32)
    z = ds.norm.normalize("df", x, 0)
    assert np.allclose(ds.norm.denormalize("df", z, 0), x, atol=1e-5)


def test_separate_zf_doubles_channels(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root),
        split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        separate_zf=True,
        backend=NumpyBackend(),
    )
    s = ds[0]
    # df channels doubled by separate_zf
    assert s.df.shape[0] == 4


def _assert_meta_equal(a: dict, b: dict):
    assert set(a) == set(b)
    for k in a:
        if k == "geometry":
            assert set(a[k]) == set(b[k])
            for gk in a[k]:
                assert np.array_equal(np.asarray(a[k][gk]), np.asarray(b[k][gk])), gk
        else:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_npz_metadata_matches_pkl(tmp_path):
    from neugk_jax.dataset.backend import load_meta, save_meta

    res = (2, 2, 2, 4, 2)
    traj = _make_synthetic_traj(tmp_path, "iteration_001", n_t=4, resolution=res)
    backend = NumpyBackend()
    meta_pkl = backend.read_metadata(str(traj))

    # convert the trajectory to npz-only metadata and read again
    base = str(traj / "metadata")
    save_meta(base, load_meta(base), ".npz")
    os.remove(traj / "metadata.pkl")
    meta_npz = backend.read_metadata(str(traj))

    _assert_meta_equal(meta_pkl, meta_npz)
    # geometry defaults filled on both routes
    g = meta_npz["geometry"]
    assert "ffun" in g
    # missing flags default to electrostatic with adiabatic electrons
    assert {k: float(g[k]) for k in ("adiabatic", "de", "beta", "nlapar", "nlbpar")} == {
        "adiabatic": 1.0,
        "de": 1.0,
        "beta": 0.0,
        "nlapar": 0.0,
        "nlbpar": 0.0,
    }
    # resolution special-cased back to a tuple of ints
    assert tuple(meta_npz["resolution"]) == res


def test_backend_exists(tmp_path):
    from neugk_jax.dataset.backend import load_meta, save_meta

    res = (2, 2, 2, 4, 2)
    backend = NumpyBackend()

    pkl_traj = _make_synthetic_traj(tmp_path, "iteration_001", n_t=2, resolution=res)
    assert backend.exists(str(pkl_traj))

    npz_traj = _make_synthetic_traj(tmp_path, "iteration_002", n_t=2, resolution=res)
    base = str(npz_traj / "metadata")
    save_meta(base, load_meta(base), ".npz")
    os.remove(npz_traj / "metadata.pkl")
    assert backend.exists(str(npz_traj))

    empty = tmp_path / "iteration_003_ifft_realpotens"
    empty.mkdir()
    assert not backend.exists(str(empty))


def test_missing_required_field_excludes_trajectory(tmp_path):
    res = (2, 2, 2, 4, 2)
    _make_synthetic_traj(tmp_path, "iteration_001", n_t=8, resolution=res)
    _make_synthetic_traj(tmp_path, "iteration_002", n_t=8, resolution=res, drop_keys=("s_hat",))
    with pytest.warns(UserWarning, match="s_hat"):
        ds = CycloneDataset(
            path=str(tmp_path),
            split="train",
            trajectories=["iteration_001", "iteration_002"],
            backend=NumpyBackend(),
        )
    assert len(ds.files) == 1
    assert "iteration_001" in ds.files[0]
    # remaining trajectory still indexes and serves samples
    assert ds[0].df is not None


def test_missing_cond_filter_field_excludes_trajectory(tmp_path):
    res = (2, 2, 2, 4, 2)
    _make_synthetic_traj(
        tmp_path,
        "iteration_001",
        n_t=8,
        resolution=res,
        extra_meta={"beta": np.array([0.5], dtype=np.float32)},
    )
    _make_synthetic_traj(tmp_path, "iteration_002", n_t=8, resolution=res)
    ds = CycloneDataset(
        path=str(tmp_path),
        split="train",
        trajectories=["iteration_001", "iteration_002"],
        cond_filters={"beta": (0.0, 1.0)},
        backend=NumpyBackend(),
    )
    # iteration_002 lacks the filter field -> excluded rather than crash
    assert len(ds.files) == 1
    assert "iteration_001" in ds.files[0]


def test_normalized_field_without_stats_raises(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root),
        split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        normalization={"df": {"type": "zscore"}, "flux": {"type": "zscore"}},
        normalization_stats={"df": {"full": {"mean": 0.0, "std": 1.0}}},
        backend=NumpyBackend(),
    )
    ds.norm.normalize("df", np.zeros((2, *ds.resolution), np.float32), 0)
    with pytest.raises(KeyError, match="flux"):
        ds.norm.normalize("flux", np.float32(1.0), 0)
