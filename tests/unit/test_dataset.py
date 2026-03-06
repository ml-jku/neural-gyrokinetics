import pytest
import torch
import numpy as np
import os
import contextlib
from unittest.mock import MagicMock, patch
from neugk.dataset.cyclone import CycloneDataset, CycloneSample, LinearCycloneDataset
from neugk.dataset.backend import DataBackend, H5Backend, KvikIOBackend


class MockBackend(DataBackend):
    def is_valid(self, path):
        return True

    def exists(self, path):
        return True

    def format_path(self, path, *args, **kwargs):
        return path

    def read_metadata(self, path, fields=None):
        return {
            "timesteps": np.arange(10),
            "resolution": (8, 4, 4, 4, 4),
            "df_mean": np.zeros((2, 8, 4, 4, 4, 4)),
            "df_std": np.ones((2, 8, 4, 4, 4, 4)),
            "df_min": -np.ones((2, 8, 4, 4, 4, 4)),
            "df_max": np.ones((2, 8, 4, 4, 4, 4)),
            "phi_mean": np.zeros((2, 4, 4, 4)),
            "phi_std": np.ones((2, 4, 4, 4)),
            "phi_min": -np.ones((2, 4, 4, 4)),
            "phi_max": np.ones((2, 4, 4, 4)),
            "flux_mean": np.zeros(1),
            "flux_std": np.ones(1),
            "flux_min": -np.ones(1),
            "flux_max": np.ones(1),
            "fluxes": np.random.randn(10),
            "ion_temp_grad": np.array([1.0]),
            "density_grad": np.array([1.0]),
            "s_hat": np.array([1.0]),
            "q": np.array([1.0]),
            "geometry": {"krho": np.zeros(4)},
        }

    @contextlib.contextmanager
    def open(self, path):
        yield MagicMock()

    @contextlib.contextmanager
    def create(self, path):
        yield MagicMock()

    def read_df(self, f, t_str, shape, active_keys=None, rank=0):
        return np.random.randn(*shape)

    def read_phi(self, f, t_str, shape, rank=0):
        # shape is (nx, ns, ny)
        return np.random.randn(2, *shape)

    def write_metadata(self, f, metadata):
        pass

    def write_df(self, f, timestamp, df):
        pass

    def write_phi(self, f, timestamp, phi):
        pass


@pytest.fixture
def mock_dataset():
    backend = MockBackend()
    with patch(
        "neugk.dataset.cyclone.os.listdir", return_value=["traj1", "traj2"]
    ), patch("neugk.dataset.cyclone.os.path.exists", return_value=True), patch(
        "neugk.dataset.cyclone.dist.is_initialized", return_value=False
    ):

        ds = CycloneDataset(
            backend=backend,
            path="/mock/path",
            fields_to_load=["df", "phi", "flux"],
            normalization={
                "df": {"type": "zscore", "agg_axes": []},
                "phi": {"type": "zscore", "agg_axes": []},
                "flux": {"type": "zscore", "agg_axes": []},
            },
            normalization_scope="dataset",
            val_ratio=0.5,
            num_workers=1,
        )
    return ds


def test_dataset_initialization(mock_dataset):
    assert len(mock_dataset.files) > 0
    assert mock_dataset.length > 0


def test_dataset_getitem(mock_dataset):
    sample = mock_dataset[0]
    assert isinstance(sample, CycloneSample)
    assert sample.df.shape == (2, 8, 4, 4, 4, 4)
    assert sample.phi.shape == (2, 4, 4, 4)


def test_recompute_stats(mock_dataset):
    with patch("neugk.dataset.cyclone.os.path.exists", return_value=False), patch(
        "neugk.dataset.cyclone.os.replace"
    ), patch("pickle.dump"), patch("builtins.open", MagicMock()):
        stats = mock_dataset._recompute_stats(keys=["df"])
        assert "df" in stats
        assert stats["df"].mean.shape == (2, 1, 1, 1, 1, 1)


def test_dataset_separate_zf(mock_dataset):
    mock_dataset.separate_zf = True
    sample = mock_dataset.__getitem__(0, get_normalized=False)
    # df shape: (2, ...) -> (4, ...) because [zf, x-zf]
    assert sample.df.shape[0] == 4


def test_dataset_collate(mock_dataset):
    samples = [mock_dataset[0], mock_dataset[0]]
    batch = mock_dataset.collate(samples)
    assert batch.df.shape == (2, 2, 8, 4, 4, 4, 4)  # (B, C, ...)


def test_get_timesteps(mock_dataset):
    file_idx = torch.tensor([0])
    ts = mock_dataset.get_timesteps(file_idx)
    assert ts.shape[0] == 1


def test_linear_dataset():
    backend = MockBackend()
    # Use trajectories directly to avoid os.listdir issue
    ds = LinearCycloneDataset(
        backend=backend,
        path="/mock/path",
        trajectories=["traj1"],
        fields_to_load=["df", "phi", "flux"],
    )
    sample = ds[0]
    assert sample.y_df is None
    assert sample.df is not None


# --- Backend Tests ---


def test_h5_backend_format_path():
    backend = H5Backend()
    path = "test.h5"

    # spatial_ifft=False
    assert backend.format_path(path, spatial_ifft=False) == "test.h5"

    # spatial_ifft=True, real_potens=True
    assert (
        backend.format_path(path, spatial_ifft=True, real_potens=True)
        == "test_ifft_realpotens.h5"
    )

    # spatial_ifft=True, split_into_bands=2
    assert (
        backend.format_path(path, spatial_ifft=True, split_into_bands=2)
        == "test_ifft_separate_zf_2bands.h5"
    )


def test_kvikio_backend_format_path():
    backend = KvikIOBackend(use_kvikio=False)
    path = "test.h5"

    # Base path should strip .h5
    assert backend.format_path(path, spatial_ifft=False) == "test"

    # spatial_ifft=True, real_potens=True
    assert (
        backend.format_path(path, spatial_ifft=True, real_potens=True)
        == "test_ifft_realpotens"
    )
