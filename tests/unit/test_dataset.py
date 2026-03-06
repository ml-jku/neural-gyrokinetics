
import pytest
import torch
import numpy as np
import os
import contextlib
from unittest.mock import MagicMock, patch
from neugk.dataset.cyclone import CycloneDataset, CycloneSample
from neugk.dataset.backend import DataBackend

class MockBackend(DataBackend):
    def is_valid(self, path): return True
    def exists(self, path): return True
    def format_path(self, path, *args, **kwargs): return path
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
    
    def read_df(self, f, t_str, shape, active_keys=None):
        return np.random.randn(*shape)
    
    def read_phi(self, f, t_str, shape):
        return np.random.randn(2, *shape)
        
    def write_metadata(self, f, metadata): pass
    def write_df(self, f, timestamp, df): pass
    def write_phi(self, f, timestamp, phi): pass

@pytest.fixture
def mock_dataset():
    backend = MockBackend()
    with patch("os.listdir", return_value=["traj1", "traj2"]), \
         patch("os.path.exists", return_value=True), \
         patch("neugk.dataset.cyclone.dist.is_initialized", return_value=False):
        
        ds = CycloneDataset(
            backend=backend,
            path="/mock/path",
            fields_to_load=["df", "phi", "flux"],
            normalization={"df": {"type": "zscore", "agg_axes": []}, 
                           "phi": {"type": "zscore", "agg_axes": []},
                           "flux": {"type": "zscore", "agg_axes": []}},
            normalization_scope="dataset",
            val_ratio=0.5,
            num_workers=1
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
    with patch("os.path.exists", return_value=False), \
         patch("os.replace"), \
         patch("pickle.dump"), \
         patch("builtins.open", MagicMock()):
        stats = mock_dataset._recompute_stats(keys=["df"])
        assert "df" in stats
        assert stats["df"].mean.shape == (2, 1, 1, 1, 1, 1)
