import os
import h5py
from typing import Type

import torch
import numpy as np
from torch.utils.data import Dataset, default_collate as collate_fn
import random


class CycloneDataset(Dataset):
    def __init__(
        self,
        path: str = "/system/user/publicdata/gyrokinetics_preprocessed",
        split: str = "train",
        normalize: bool = True,
        dtype: Type = None,
        random_seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        in_memory: bool = False,
    ):
        assert split in ["train", "val", "test"]
        self.dtype = torch.float32 if dtype is None else dtype
        self.normalize = normalize
        self.in_memory = in_memory
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
        if test_ratio != 0:
            test_idx = random.sample(
                list(perm), max(1, int(test_ratio * len(self.files)))
            )
            perm = perm - set(test_idx)
        train_idx = list(perm)

        if split == "train":
            self.files = [self.files[i] for i in train_idx]
        if split == "val":
            self.files = [self.files[i] for i in val_idx]
        if split == "test":
            self.files = [self.files[i] for i in test_idx]

        print(self.files)
        # get total number of samples (assuming no bundling or pushforward and assuming constant number of timesteps accross files)
        with h5py.File(self.files[0], "r") as f:
            # read the timesteps
            timesteps = f["metadata/timesteps"][:]
            n_timesteps = len(timesteps) - 1

        self.length = int(n_timesteps) * len(self.files)
        self.n_samples_per_file = n_timesteps
        # self.bounds = None

        if self.in_memory:
            # load all timesteps into a dict of dicts
            self.data = {}
            for file_idx, file in enumerate(self.files):
                file_dict = {}
                with h5py.File(file, "r") as f:
                    # read the 'metadata/timesteps' dataset
                    timesteps = torch.from_numpy(f["metadata/timesteps"][:]).to(
                        dtype=self.dtype
                    )
                    file_dict["metadata/timesteps"] = timesteps
                    # read in all the data points
                    for t_index in range(len(file_dict["metadata/timesteps"])):
                        name = "timestep_" + str(t_index).zfill(2)
                        x = torch.from_numpy(f[f"data/{name}"][:]).to(dtype=self.dtype)
                        file_dict[f"data/{name}"] = x
                self.data[file_idx] = file_dict

    def __getitem__(self, index):
        # calculate file index and remainder for time index in file
        file_index = int(index // self.n_samples_per_file)
        t_index = int(index % self.n_samples_per_file)

        if self.in_memory:
            timestep = self.data[file_index]["metadata/timesteps"][t_index]
            # read the input
            name = "timestep_" + str(t_index).zfill(2)
            x = self.data[file_index][f"data/{name}"]

            # read the gt output (next timestep)
            name_gt = "timestep_" + str(t_index + 1).zfill(2)
            gt = self.data[file_index][f"data/{name_gt}"]

        else:
            with h5py.File(self.files[file_index], "r") as f:
                # read the 'metadata/timesteps' dataset
                # timestep = f["metadata/timesteps"][t_index]
                timestep = t_index

                # read the input
                name = "timestep_" + str(t_index).zfill(2)
                x = f[f"data/{name}"][:]

                # read the gt output (next timestep)
                name_gt = "timestep_" + str(t_index + 1).zfill(2)
                gt = f[f"data/{name_gt}"][:]

        # TODO find better way!
        if self.normalize:
            x = (x - x.mean()) / x.std()
            gt = (gt - gt.mean()) / gt.std()

        return (
            torch.tensor(x).type(self.dtype),
            torch.tensor(timestep).type(self.dtype),
            torch.tensor(gt).type(self.dtype),
            torch.tensor(file_index).type(self.dtype),  # accessory information
        )

    def get_at_time(self, file_idx: torch.Tensor, timestep: torch.Tensor):
        updated_index = (file_idx * self.n_samples_per_file + timestep).long()
        return collate_fn([self.__getitem__(idx) for idx in updated_index.tolist()])

    def __len__(self):
        return self.length

    def num_ts(self, file_id: int):
        return 49  # TODO can change?
