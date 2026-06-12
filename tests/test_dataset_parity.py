"""Dataset shape + behaviour tests against a synthetic KvikIO-style directory.

We don't need real cyclone preprocessed data to test the loader plumbing:
a temp directory with a hand-built ``metadata.pkl`` and a few
``timestep_NNNNN.bin`` files exercises every code path. A separate parity
test against the actual upstream torch loader runs only when
``NEUGK_CYCLONE_PATH`` points at real data.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from neugk_jax.dataset import CycloneDataset, NumpyBackend


def _make_synthetic_traj(root: Path, name: str, *, n_t: int, resolution):
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
            for k in ("krho", "ints", "intmu", "intvp", "vpgr", "mugr",
                     "bn", "efun", "rfun", "bt_frac", "parseval",
                     "mas", "tmp", "d2X", "signz", "signB",
                     "kxrh", "little_g")
        },
        "df_mean": np.zeros(1, dtype=np.float32),
        "df_std": np.ones(1, dtype=np.float32),
        "df_min": np.full(1, -3.0, dtype=np.float32),
        "df_max": np.full(1, 3.0, dtype=np.float32),
    }
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
        path=str(root), split="train",
        trajectories=["iteration_001", "iteration_002"],
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        offset=0, bundle_seq_length=1,
        backend=NumpyBackend(),
    )
    assert len(ds.files) == 2
    assert ds.resolution == res
    # per-file samples = file_num_timesteps - bundle_seq_length*2 + 1 = 8 - 2 + 1 = 7
    assert len(ds) == 7 * 2


def test_getitem_shapes(synthetic_dir):
    root, res = synthetic_dir
    ds = CycloneDataset(
        path=str(root), split="train",
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


def test_collate_batches(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root), split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        backend=NumpyBackend(),
    )
    batch = CycloneDataset.collate([ds[i] for i in range(3)])
    assert batch.df.shape == (3, 2, *ds.resolution)
    assert batch.conditioning.shape == (3, 4)
    assert batch.flux.shape == (3,)


def test_normalize_denormalize_roundtrip(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root), split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        normalization={"df": {"type": "zscore"}},
        normalization_scope="dataset",
        backend=NumpyBackend(),
    )
    # craft a synthetic field of known mean/std and roundtrip through normalize
    x = np.full((2, *ds.resolution), 5.0, dtype=np.float32)
    z = ds.normalize(0, df=x)
    y = ds.denormalize(0, df=z)
    assert np.allclose(y, x, atol=1e-5)


def test_separate_zf_doubles_channels(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root), split="train",
        trajectories=["iteration_001"],
        fields_to_load=("df",),
        separate_zf=True,
        backend=NumpyBackend(),
    )
    s = ds[0]
    # df channels doubled by separate_zf
    assert s.df.shape[0] == 4


def test_get_batch_geometry(synthetic_dir):
    root, _ = synthetic_dir
    ds = CycloneDataset(
        path=str(root), split="train",
        trajectories=["iteration_001", "iteration_002"],
        backend=NumpyBackend(),
    )
    file_idx = np.array([0, 1, 0])
    g = ds.get_batch_geometry(file_idx)
    # every key batched to size 3
    for k, v in g.items():
        assert v.shape[0] == 3, f"{k} not batched"




@pytest.mark.skipif(
    not os.environ.get("NEUGK_CYCLONE_PATH"),
    reason="set NEUGK_CYCLONE_PATH to run torch-loader parity (needs real data)",
)
@pytest.mark.parametrize("separate_zf", [False, True])
def test_byte_equal_to_torch_loader(separate_zf):
    """Compare ``df`` bytes against the upstream torch ``CycloneAEDataset``.

    Parameterised over ``separate_zf`` so we catch divergences in either
    the raw read path or the channel-axis pre-processing.

    Required env vars::
        NEUGK_CYCLONE_PATH=/local00/bioinf/galletti/preprocessed_kvikio
        NEUGK_CYCLONE_TRAJS=iteration_{0-1}
    """
    import sys
    sys.path.insert(0, "/system/user/publicwork/galletti/git/neural-gyrokinetics-gitlab")
    from neugk.dataset.cyclone_diff import CycloneAEDataset as TorchAE
    from neugk.dataset.backend import KvikIOBackend as TorchKvikIO

    path = os.environ["NEUGK_CYCLONE_PATH"]
    trajectories = os.environ.get("NEUGK_CYCLONE_TRAJS", "iteration_{0-1}")

    t_ds = TorchAE(
        backend=TorchKvikIO(rank=0, use_kvikio=False),
        path=path, split="train",
        trajectories=trajectories,
        partial_holdouts={},
        fields_to_load=["df"],
        probe_targets=[],
        conditions=["itg", "dg", "s_hat", "q"],
        normalization=None,
        offset=0,
        bundle_seq_length=1,
        spatial_ifft=True,
        real_potens=True,
        separate_zf=separate_zf,
    )
    j_ds = CycloneDataset(
        path=path, split="train",
        trajectories=trajectories,
        fields_to_load=("df",),
        conditions=("itg", "dg", "s_hat", "q"),
        mode="ae",
        backend=NumpyBackend(),
        separate_zf=separate_zf,
    )
    assert len(t_ds) == len(j_ds), f"lengths differ: torch={len(t_ds)} jax={len(j_ds)}"
    # match by (file basename, timestep) — ordering differs between stacks (torch: set, jax: sorted)
    t_lookup = {
        (os.path.basename(t_ds.files[fid]), int(t_idx)): flat
        for flat, (fid, t_idx) in t_ds.flat_index_to_file_and_tstep.items()
    }
    for i in (0, 1, len(j_ds) // 2, len(j_ds) - 1):
        j_fid, j_t = j_ds.flat_index_to_file_and_tstep[i]
        key = (os.path.basename(j_ds.files[j_fid]), int(j_t))
        ti = t_lookup[key]
        ts = t_ds[ti]
        js = j_ds[i]
        t_df = np.asarray(ts.df)
        j_df = np.asarray(js.df)
        assert t_df.shape == j_df.shape, (
            f"shape mismatch at jax-idx {i} (separate_zf={separate_zf}): "
            f"torch={t_df.shape} jax={j_df.shape}"
        )
        # post-separate_zf path subtracts a mean ⇒ tiny rounding allowed
        atol = 1e-6 if separate_zf else 0.0
        diff = np.abs(t_df - j_df).max()
        assert diff <= atol, (
            f"df differs at jax-idx {i} key={key} "
            f"(separate_zf={separate_zf}), max|diff|={diff}"
        )
        t_cond = np.asarray(ts.conditioning) if ts.conditioning is not None else None
        j_cond = np.asarray(js.conditioning)
        if t_cond is not None:
            assert np.array_equal(t_cond, j_cond), f"conditioning differs at {key}"
