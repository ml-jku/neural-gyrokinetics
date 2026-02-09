from typing import Optional

import h5py
import os
import pickle
import tqdm
import hashlib
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import QuantileTransformer

import torch
import numpy as np

from neugk.dataset import CycloneAEDataset


class ChannelWiseGaussianizer:
    def __init__(self, n_quantiles: int = 10000, epsilon: float = 1e-5):
        self.n_quantiles = n_quantiles
        self.epsilon = epsilon
        self.qtrans = []
        self.n_channels = None

    def fit(self, data: np.ndarray, channel_dim=1):
        data_swapped = np.moveaxis(data, channel_dim, 0)
        self.n_channels = data_swapped.shape[0]
        self.qtrans = [
            QuantileTransformer(
                n_quantiles=self.n_quantiles,
                output_distribution="normal",
                subsample=None,
            )
            for _ in range(self.n_channels)
        ]

        for c in range(self.n_channels):
            self.qtrans[c].fit(data_swapped[c].reshape(-1, 1))
        return self

    def transform(self, x: np.ndarray, channel_dim: int = 0):
        device = None
        if isinstance(x, torch.Tensor):
            device, dtype = x.device, x.dtype
            x = x.detach().cpu().numpy()

        x = np.moveaxis(x, channel_dim, 0)
        shape = x[0].shape
        for c in range(self.n_channels):
            x[c] = self.qtrans[c].transform(x[c].reshape(-1, 1)).reshape(shape)
        x = np.moveaxis(x, 0, channel_dim)

        if device is not None:
            x = torch.from_numpy(x).to(device, dtype=dtype)
        return x

    def inverse_transform(self, x: np.ndarray, channel_dim: int = 0):
        device = None
        if isinstance(x, torch.Tensor):
            device, dtype = x.device, x.dtype
            x = x.clone().cpu().numpy()

        x = np.moveaxis(x, channel_dim, 0)
        shape = x[0].shape
        for c in range(self.n_channels):
            x[c] = self.qtrans[c].inverse_transform(x[c].reshape(-1, 1)).reshape(shape)
        x = np.moveaxis(x, 0, channel_dim)

        if device is not None:
            x = torch.from_numpy(x).to(device, dtype=dtype)
        return x


class CycloneAEDatasetGaussianized(CycloneAEDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussianizer = self.get_gaussianizers(self.offsets[0])

    def get_gaussianizers(self, offset: int = 0):
        def load_raw_sample(t_idx):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]
            with h5py.File(self.files[file_index], "r") as f:
                sample = self._load_data(f, file_index, t_index)

            x = sample["x"]
            if self.separate_zf:
                x = self._separate_zf(x)
            return x

        n_samples = max(5, int(self.length * 0.05))
        indices = np.random.choice(self.length, n_samples, replace=False)
        indices.sort()
        traj_hash = hashlib.sha256("".join(sorted(self.files)).encode()).hexdigest()[:8]
        stats_dump_pkl = os.path.join(
            self.dir, f"gauss_df_offset{offset}_{traj_hash}_1pct_stats.pkl"
        )

        if os.path.exists(stats_dump_pkl):
            with open(stats_dump_pkl, "rb") as f:
                gauss = pickle.load(f)
        else:
            batch_data = []
            with ThreadPoolExecutor(self.num_workers) as executor:
                results = list(
                    tqdm.tqdm(
                        executor.map(load_raw_sample, indices),
                        total=len(indices),
                        desc=f"Loading 5% subsample for df gaussianization",
                    )
                )
                batch_data = np.stack([r for r in results if r is not None], axis=0)

            gauss = ChannelWiseGaussianizer(n_quantiles=1000)
            gauss.fit(batch_data, channel_dim=1)

            with open(stats_dump_pkl, "wb") as f:
                pickle.dump(gauss, f)

        return gauss

    def normalize(
        self,
        file_index: int,
        df: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        flux: Optional[torch.Tensor] = None,
    ):
        if df is not None:
            return self.gaussianizer.transform(df, channel_dim=0)

        if phi is not None:
            scale, shift = self._get_scale_shift(file_index, "phi", phi)
            return (phi - shift) / scale

        if flux is not None:
            scale, shift = self._get_scale_shift(file_index, "flux", flux)
            # return (flux - shift) / scale
            return flux

    def denormalize(
        self,
        file_index: int,
        df: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        flux: Optional[torch.Tensor] = None,
    ):
        if df is not None:
            return self.gaussianizer.inverse_transform(df, channel_dim=0)

        if phi is not None:
            scale, shift = self._get_scale_shift(file_index, "phi", phi)
            return phi * scale + shift

        if flux is not None:
            scale, shift = self._get_scale_shift(file_index, "flux", flux)
            # return flux * scale + shift
            return flux
