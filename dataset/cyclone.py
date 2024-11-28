import os
import h5py
from typing import Type

import torch
import numpy as np
from torch.utils.data import Dataset
import random


class CycloneDataset(Dataset):
    def __init__(
        self,
        path: str = "/system/user/publicdata/gyrokinetics_preprocessed",
        split: str = "train",
        dtype: Type = None,
        random_seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        assert split in ["train", "val", "test"]
        self.dtype = torch.float32 if dtype is None else dtype

        self.dir = path
        self.trajectories_fnames = os.listdir(path)
        self.files = [
            os.path.join(self.dir, f_name) for f_name in self.trajectories_fnames
        ]

        # shuffle and split files into training and validation sets
        random.seed(random_seed)
        random.shuffle(self.files)
        # ensure at least one sample in validation and test sets
        perm = set(np.random.permutation(len(self.files)))
        val_idx = random.sample(list(perm), max(1, int(val_ratio * len(self.files))))
        perm = perm - set(val_idx)
        test_idx = random.sample(list(perm), max(1, int(test_ratio * len(self.files))))
        perm = perm - set(test_idx)
        train_idx = list(perm)

        if split == "train":
            self.files = [self.files[i] for i in train_idx]
        if split == "val":
            self.files = [self.files[i] for i in val_idx]
        if split == "test":
            self.files = [self.files[i] for i in test_idx]

        # get total number of samples (assuming no bundling and assuming constant number of timesteps accross files)
        with h5py.File(self.files[0], "r") as f:
            # read the timesteps
            timesteps = f["metadata/timesteps"][:]
            n_timesteps = len(timesteps) - 1

        self.length = int(n_timesteps) * len(self.files)
        self.n_samples_per_file = n_timesteps
        self.bounds = None

    def __getitem__(self, index):
        # calculate file index and remainder for time index in file
        file_index = int(index // self.n_samples_per_file)
        t_index = int(index % self.n_samples_per_file)

        # load data
        with h5py.File(self.files[file_index], "r") as f:
            # read the 'metadata/timesteps' dataset
            timestep = f["metadata/timesteps"][t_index]

            # read the input
            name = "timestep_" + str(t_index).zfill(2)
            x = f[f"data/{name}"][:]

            # read the gt output (next timestep)
            name_gt = "timestep_" + str(t_index + 1).zfill(2)
            gt = f[f"data/{name_gt}"][:]

        # normalization
        if self.bounds:
            x, gt = self._normalize_sample(x, gt)
            return (
                torch.tensor(x).type(self.dtype),
                torch.tensor(gt).type(self.dtype),
                torch.tensor(timestep).type(self.dtype),
            )
        else:
            return (
                torch.tensor(x).type(self.dtype),
                torch.tensor(gt).type(self.dtype),
                torch.tensor(timestep).type(self.dtype),
            )

    def get_bounds(self):
        min_ = np.array([np.inf, np.inf])
        max_ = np.array([-np.inf, -np.inf])
        for f in self.files:
            with h5py.File(f, "r") as file:
                timesteps = file["metadata/timesteps"][:]
                for idx, timestep in enumerate(timesteps):
                    name = "timestep_" + str(idx).zfill(2)
                    data = file[f"data/{name}"][:]
                    # Update min and max across the first dimension
                    min_ = np.minimum(min_, np.min(data, axis=(1, 2, 3, 4, 5)))
                    max_ = np.maximum(max_, np.max(data, axis=(1, 2, 3, 4, 5)))
        return [min_, max_]

    def normalize(self, bounds):
        self.bounds = bounds

    def _normalize_sample(self, x, y):
        # reshape min_ and max_ to be broadcastable
        min_ = self.bounds[0][:, None, None, None, None, None]
        max_ = self.bounds[0][:, None, None, None, None, None]

        # eps for numerical stability
        eps = 1e-10
        return 2 * (x - min_) / (max_ - min_ + eps), 2 * (y - min_) / (
            max_ - min_ + eps
        )

    def __len__(self):
        return self.length
