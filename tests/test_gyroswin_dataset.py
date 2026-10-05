"""CycloneDataset ``mode="next"``: next-step index rule, normalization, timestep condition."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from neugk_jax.dataset import CycloneDataset, NumpyBackend

RES = (2, 2, 2, 4, 2)


def _make_traj(root: Path, name: str, *, n_t: int, seed: int):
    traj = root / f"{name}_ifft_realpotens"
    data = traj / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # every df/phi entry of step t equals t + seed/100, so a read identifies (trajectory, step)
    for t in range(n_t):
        np.full((2, *RES), t + seed / 100, np.float32).tofile(data / f"timestep_{t:05d}.bin")
        np.full((RES[3], RES[2], RES[4]), -(t + seed / 100), np.float32).tofile(
            data / f"poten_{t:05d}.bin"
        )
    meta = {
        "timesteps": np.arange(n_t, dtype=np.float64) * 0.5 + seed,
        "flux": rng.standard_normal(n_t).astype(np.float32),
        "ion_temp_grad": np.array([2.3], np.float32),
        "density_grad": np.array([1.1], np.float32),
        "s_hat": np.array([0.8], np.float32),
        "q": np.array([1.4], np.float32),
        "resolution": np.array(RES),
        "geometry": {"krho": np.ones((1,))},
    }
    with open(traj / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    return meta


@pytest.fixture
def two_trajs(tmp_path):
    metas = [_make_traj(tmp_path, f"iteration_{i}", n_t=6 + i, seed=10 * (i + 1)) for i in range(2)]
    return tmp_path, metas


def _stats(flux_mean=1.0, flux_std=2.0, avg_mean=-1.0, avg_std=4.0):
    def one(m, s):
        return {
            "full": {
                "mean": np.asarray(m, np.float32),
                "std": np.asarray(s, np.float32),
                "min": np.asarray(m - s, np.float32),
                "max": np.asarray(m + s, np.float32),
            }
        }

    return {
        "df": one(0.5, 2.0),
        "phi": one(-0.5, 3.0),
        "flux": one(flux_mean, flux_std),
        "fluxavg": one(avg_mean, avg_std),
    }


_NORM = {k: {"type": "zscore"} for k in ("df", "phi", "flux", "fluxavg")}


def test_next_step_index_never_crosses_trajectories(two_trajs):
    root, metas = two_trajs
    ds = CycloneDataset(
        path=str(root),
        trajectories=["iteration_0", "iteration_1"],
        fields_to_load=("df", "phi"),
        mode="next",
        backend=NumpyBackend(),
        offset=1,
    )
    # per file: (n_t - offset) - 2 * bundle + 1 samples
    assert len(ds) == (6 - 1 - 1) + (7 - 1 - 1)
    for i in range(len(ds)):
        s = ds[i]
        fid, t = int(s.file_index), int(s.timestep_index)
        tag = (10 * (fid + 1)) / 100
        assert np.allclose(s.df, t + 1 + tag)
        assert np.allclose(s.y_df, t + 2 + tag)
        assert np.allclose(s.y_phi, -(t + 2 + tag))
        assert s.y_flux == pytest.approx(metas[fid]["flux"][t + 2])
        assert s.y_fluxavg == pytest.approx(np.mean(metas[fid]["flux"][1:][-80:]))
        assert s.timestep == pytest.approx(metas[fid]["timesteps"][t + 1])


def test_next_step_tail_offset_keeps_rollout_targets(two_trajs):
    root, _ = two_trajs
    ds = CycloneDataset(
        path=str(root),
        trajectories=["iteration_0"],
        fields_to_load=("df",),
        mode="next",
        backend=NumpyBackend(),
        tail_offset=2,
        split="val",
    )
    assert len(ds) == 6 - 2 - 1
    last = int(ds[len(ds) - 1].timestep_index)
    # the tail keeps full rollouts: every step capped by num_ts has a target on disk
    steps = ds.num_ts(0) - last - 1
    assert steps >= 2
    for t in range(steps):
        ds.get_target(0, last + t)
    with pytest.raises(FileNotFoundError):
        ds.get_target(0, last + steps)


def test_rollout_cap_with_subsample(two_trajs):
    root, _ = two_trajs
    ds = CycloneDataset(
        path=str(root),
        trajectories=["iteration_1"],
        fields_to_load=("df",),
        mode="next",
        backend=NumpyBackend(),
        tail_offset=2,
        split="val",
        subsample=2,
    )
    t0 = [int(ds[i].timestep_index) for i in range(len(ds))]
    assert t0 == [0, 2]
    # num_ts and timestep_index are both raw, so no sample loses its rollout
    assert all(ds.num_ts(0) - t - 1 >= 2 for t in t0)


def test_next_step_normalization(two_trajs):
    root, metas = two_trajs
    ds = CycloneDataset(
        path=str(root),
        trajectories=["iteration_0"],
        fields_to_load=("df", "phi"),
        mode="next",
        backend=NumpyBackend(),
        normalization=_NORM,
        normalization_stats=_stats(),
    )
    raw = CycloneDataset(
        path=str(root),
        trajectories=["iteration_0"],
        fields_to_load=("df", "phi"),
        mode="next",
        backend=NumpyBackend(),
    )[2]
    s = ds[2]
    assert np.allclose(s.df, (raw.df - 0.5) / 2.0)
    assert np.allclose(s.y_df, (raw.y_df - 0.5) / 2.0)
    assert np.allclose(s.y_phi, (raw.y_phi + 0.5) / 3.0)
    assert s.y_flux == pytest.approx((metas[0]["flux"][3] - 1.0) / 2.0)
    assert s.y_fluxavg == pytest.approx((np.mean(metas[0]["flux"][1:][-80:]) + 1.0) / 4.0)
    assert np.allclose(ds.norm.denormalize("fluxavg", s.y_fluxavg, 0), raw.y_fluxavg, atol=1e-6)


def test_timestep_condition(two_trajs):
    root, metas = two_trajs
    ds = CycloneDataset(
        path=str(root),
        trajectories=["iteration_0", "iteration_1"],
        fields_to_load=("df",),
        mode="next",
        backend=NumpyBackend(),
        conditions=("timestep", "itg"),
    )
    assert len(ds.files) == 2
    assert ds.conditions == ["itg", "timestep"]
    s = ds[3]
    assert s.conditioning[1] == pytest.approx(
        metas[int(s.file_index)]["timesteps"][int(s.timestep_index)]
    )
    assert ds.get_timestep(int(s.file_index), int(s.timestep_index)) == pytest.approx(
        s.conditioning[1]
    )


def test_scale_shift_batches_per_trajectory_stats(two_trajs):
    root, _ = two_trajs
    stats = _stats()
    stats["flux"] = {
        0: {"mean": np.ones(1, np.float32), "std": np.full(1, 2.0, np.float32)},
        1: {"mean": np.zeros(1, np.float32), "std": np.full(1, 3.0, np.float32)},
    }
    ds = CycloneDataset(
        path=str(root),
        trajectories=["iteration_0", "iteration_1"],
        mode="next",
        backend=NumpyBackend(),
        normalization=_NORM,
        normalization_stats=stats,
        normalization_scope="trajectory",
    )
    scale, shift = ds.norm.scale_shift("flux", np.asarray([0, 1, 0]))
    assert scale.shape == (3,) and np.allclose(scale, [2.0, 3.0, 2.0])
    assert np.allclose(shift, [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="normalization_scope"):
        CycloneDataset(
            path=str(root),
            trajectories=["iteration_0"],
            backend=NumpyBackend(),
            normalization=_NORM,
            normalization_stats=stats,
            normalization_scope="sample",
        )
