import os
import h5py
from typing import Type
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import Dataset

import random


@dataclass
class CycloneSample:
    x: torch.Tensor
    y: torch.Tensor
    timestep: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    # TODO: add more fields (e.g. params that we can use for conditioning)


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
        n_eval_steps: int = 1,
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
        self.files = self.files[:2]
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
            # 1 step training
            self.n_eval_steps = 1
        if split == "val":
            self.files = [self.files[i] for i in val_idx]
            self.n_eval_steps = n_eval_steps
        if split == "test":
            self.files = [self.files[i] for i in test_idx]

        # get total number of samples (assuming no bundling or pushforward and assuming constant number of timesteps accross files)
        with h5py.File(self.files[0], "r") as f:
            # read the timesteps
            timesteps = f["metadata/timesteps"][:]
            n_timesteps = len(timesteps) - self.n_eval_steps

        self.length = int(n_timesteps) * len(self.files)
        self.n_samples_per_file = n_timesteps
        # needed to access the last few targets when doing validation
        self.n_samples_per_file_val = n_timesteps + self.n_eval_steps

        if self.in_memory:
            # load all timesteps into a dict of dicts
            self.data = {}
            for file_idx, file in enumerate(self.files):
                file_dict = {}
                with h5py.File(file, "r") as f:
                    # read the 'metadata/timesteps' dataset
                    timesteps = f["metadata/timesteps"][:]
                    file_dict["metadata/timesteps"] = timesteps
                    # read in all the data points
                    for t_index in range(len(file_dict["metadata/timesteps"])):
                        name = "timestep_" + str(t_index).zfill(2)
                        x = f[f"data/{name}"][:]
                        file_dict[f"data/{name}"] = x
                self.data[file_idx] = file_dict

    def __getitem__(self, index, validating=False):
        # calculate file index and remainder for time index in file
        if not validating:
            file_index = int(index // self.n_samples_per_file)
            t_index = int(index % self.n_samples_per_file)
        else:
            file_index = int(index // self.n_samples_per_file_val)
            t_index = int(index % self.n_samples_per_file_val)

        if self.in_memory:
            x, timestep, gt = self._load_data(self.data[file_index], t_index)
        else:
            with h5py.File(self.files[file_index], "r") as f:
                x, timestep, gt = self._load_data(f, t_index)

        # TODO find better way!
        if self.normalize:
            x = (x - x.mean()) / x.std()
            gt = (gt - gt.mean()) / gt.std()

        # return (
        #     torch.tensor(x).type(self.dtype),
        #     torch.tensor(timestep).type(self.dtype),
        #     torch.tensor(gt).type(self.dtype),
        #     torch.tensor(file_index).type(self.dtype),  # accessory information
        # )
        return CycloneSample(
            x=torch.tensor(x, dtype=self.dtype),
            y=torch.tensor(gt, dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=self.dtype),
            timestep_index=torch.tensor(t_index, dtype=self.dtype),
        )

    def _load_data(self, data, t_index):
        timestep = data["metadata/timesteps"][t_index]

        # read the input
        name = "timestep_" + str(t_index).zfill(2)
        x = data[f"data/{name}"][:]

        # read the gt output (next timestep)
        name_gt = "timestep_" + str(t_index + 1).zfill(2)
        gt = data[f"data/{name_gt}"][:]

        return x, timestep, gt

    def get_at_time(self, file_idx: torch.Tensor, timestep_idx: torch.Tensor):
        updated_index = (file_idx * self.n_samples_per_file + timestep_idx).long()
        return self.collate(
            [self.__getitem__(idx, validating=True) for idx in updated_index.tolist()]
        )

    def get_timesteps_only(self, file_idx: torch.Tensor, timestep_idx: torch.Tensor):
        # file_idx: (B,)
        # timestep_idx: (B, N)
        file_idx = file_idx.cpu().long()
        timestep_idx = timestep_idx.cpu().long()

        B = file_idx.shape[0]
        if timestep_idx.dim() == 1:
            # If timestep_idx is also (B,) we can handle it directly
            flat_file_idx = file_idx
            flat_timestep_idx = timestep_idx
        else:
            # For the (B, N) case:
            N = timestep_idx.shape[1]
            # Expand file_idx to match the shape of timestep_idx
            # file_idx: (B,) -> file_idx_expanded: (B, N)
            file_idx_expanded = file_idx.unsqueeze(1).expand(B, N)

            # Flatten both to 1D
            flat_file_idx = file_idx_expanded.flatten()  # (B*N,)
            flat_timestep_idx = timestep_idx.flatten()  # (B*N,)

        # Compute the linear index
        updated_index = (
            flat_file_idx * self.n_samples_per_file + flat_timestep_idx
        ).long()

        timesteps_list = []
        for idx in updated_index.tolist():
            # use n_samples_per_file_val, because we need to access the last timesteps
            file_index = int(idx // self.n_samples_per_file_val)
            t_index = int(idx % self.n_samples_per_file_val)

            if self.in_memory:
                timesteps_array = self.data[file_index]["metadata/timesteps"]
                step_value = timesteps_array[t_index]
            else:
                with h5py.File(self.files[file_index], "r") as f:
                    timesteps_array = f["metadata/timesteps"][:]
                    step_value = timesteps_array[t_index]

            timesteps_list.append(torch.tensor(step_value, dtype=self.dtype))

        # Stack into a single tensor
        timesteps_tensor = torch.stack(timesteps_list)

        # Now reshape to the original shape
        # If original was just (B,), then no reshape needed.
        if timestep_idx.dim() > 1:
            timesteps_tensor = timesteps_tensor.view(B, -1)

        return timesteps_tensor

    def __len__(self):
        return self.length

    def num_ts(self, file_id: int):
        return self.n_samples_per_file

    def collate(self, batch):
        # batch is a list of CycloneSamples
        return CycloneSample(
            x=torch.stack([sample.x for sample in batch]),
            y=torch.stack([sample.y for sample in batch]),
            timestep=torch.stack([sample.timestep for sample in batch]),
            file_index=torch.stack([sample.file_index for sample in batch]),
            timestep_index=torch.stack([sample.timestep_index for sample in batch]),
        )
