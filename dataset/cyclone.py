import os
import h5py
from typing import Type, Optional, List, Tuple
from dataclasses import dataclass
from warnings import warn

from einops import rearrange
import torch
import numpy as np
from torch.utils.data import Dataset

import random


@dataclass
class CycloneSample:
    x: torch.Tensor
    y: torch.Tensor
    y_poten: torch.Tensor
    y_flux: torch.Tensor
    timestep: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    # TODO: add more fields (e.g. params that we can use for conditioning)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self


class CycloneDataset(Dataset):
    def __init__(
        self,
        path: str = "/system/user/publicdata/gyrokinetics_preprocessed",
        split: str = "train",
        active_keys: Optional[List[str]] = None,
        trajectories: List = None,
        normalization: Optional[str] = "zscore",
        spatial_ifft: bool = True,
        dtype: Type = None,
        random_seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        in_memory: bool = False,
        input_seq_length: int = 1,
        target_seq_length: int = 1,
        n_eval_steps: int = 1,
    ):
        assert split in ["train", "val", "test"]
        self.dtype = torch.float32 if dtype is None else dtype

        active_keys = active_keys if active_keys is not None else ["re", "im"]
        assert all([a in ["re", "im"] for a in active_keys])
        self.active_keys = np.array([{"re": 0, "im": 1}[k] for k in active_keys])
        assert normalization in ["zscore", "minmax", "none", None]
        self.normalization = normalization if normalization != "none" else None

        self.spatial_ifft = spatial_ifft
        if spatial_ifft and not in_memory:
            warn("`spatial_ifft` is only applied when `in_memory=True`.")

        self.in_memory = in_memory
        self.dir = path

        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length
        assert (
            self.input_seq_length == self.target_seq_length
        ), "Currently, only same length of input and target is supported!"

        if trajectories is None:
            self.trajectories_fnames = os.listdir(path)
            self.files = [
                os.path.join(self.dir, f_name) for f_name in self.trajectories_fnames
            ]
            # shuffle and split files into training and validation sets
            random.seed(random_seed)
            random.shuffle(self.files)
            # ensure at least one sample in validation and test sets
            perm = set(np.random.permutation(len(self.files)))
            val_idx = random.sample(
                list(perm), max(1, int(val_ratio * len(self.files)))
            )
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
                self.n_eval_steps = 1 * self.target_seq_length
            if split == "val":
                self.files = [self.files[i] for i in val_idx]
                self.n_eval_steps = n_eval_steps * self.target_seq_length
            if split == "test":
                self.files = [self.files[i] for i in test_idx]
        else:
            self.files = [os.path.join(self.dir, f_name) for f_name in trajectories]
            if split == "train":
                # 1 step training
                self.n_eval_steps = 1 * self.target_seq_length
            if split == "val":
                self.n_eval_steps = n_eval_steps * self.target_seq_length

        # get total number of samples (assuming no bundling or pushforward and assuming constant number of timesteps accross files)
        with h5py.File(self.files[0], "r") as f:
            # read the timesteps
            timesteps = f["metadata/timesteps"][:]
            self.n_samples_per_file_val = len(timesteps)
            n_timesteps = (
                len(timesteps) - self.n_eval_steps - self.input_seq_length + 1
            )  # TODO: check if this is correct, we should return a sample like [b, c, t, ...]

        self.length = n_timesteps * len(self.files)
        self.n_samples_per_file = n_timesteps
        # needed to access the last few targets when doing validation

        if self.in_memory:
            # load all timesteps into a dict of dicts
            self.data = {}
            for file_idx, file in enumerate(self.files):
                file_dict = {}
                with h5py.File(file, "r") as f:
                    # read the 'metadata/timesteps' dataset
                    timesteps = f["metadata/timesteps"][:]
                    file_dict["metadata/timesteps"] = timesteps
                    fluxes = f["metadata/fluxes"][:]
                    file_dict["metadata/fluxes"] = fluxes
                    # read in all the data points
                    for t_index in range(len(file_dict["metadata/timesteps"])):
                        k_name = "timestep_" + str(t_index).zfill(5)
                        poten_name = "poten_" + str(t_index).zfill(5)
                        k_data = f[f"data/{k_name}"][:]
                        poten_data = f[f"data/{poten_name}"][:]
                        if self.spatial_ifft:
                            # invert fft on spatial
                            k_data = np.moveaxis(k_data, 0, -1)
                            k_data = k_data.copy().view(dtype=np.complex64)
                            k_data = np.fft.ifftn(k_data, axes=(3, 4))
                            k_data = np.stack([k_data.real, k_data.imag]).squeeze()
                        # cache samples
                        file_dict[f"data/{k_name}"] = k_data
                        file_dict[f"data/{poten_name}"] = poten_data
                self.data[file_idx] = file_dict

        with h5py.File(self.files[0], "r") as f:
            self.resolution = f["metadata/resolution"][:]

    def __getitem__(self, index, validating=False):
        # calculate file index and remainder for time index in file
        if not validating:
            file_index = int(index // self.n_samples_per_file)
            t_index = int(index % self.n_samples_per_file)
        else:
            file_index = int(index // self.n_samples_per_file_val)
            t_index = int(index % self.n_samples_per_file_val)

        if self.in_memory:
            x, timestep, gt_poten, gt_flux, gt = self._load_data(
                self.data[file_index], t_index
            )
        else:
            with h5py.File(self.files[file_index], "r") as f:
                x, timestep, gt_poten, gt_flux, gt = self._load_data(f, t_index)

        if self.normalization is not None:
            x = self._normalize(x)
            gt = self._normalize(gt)

        return CycloneSample(
            x=torch.tensor(x, dtype=self.dtype),
            y=torch.tensor(gt, dtype=self.dtype),
            y_poten=torch.tensor(gt_poten, dtype=self.dtype),
            y_flux=torch.tensor(gt_flux, dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=self.dtype),
            timestep_index=torch.tensor(t_index, dtype=self.dtype),
        )

    def _load_data(self, data, t_index) -> Tuple[np.ndarray, float, float, np.ndarray]:
        timestep = data["metadata/timesteps"][t_index]
        x = []
        gt = []
        gt_poten = []
        gt_flux = []
        for i in range(self.input_seq_length):
            # read the input
            k_name = "timestep_" + str(t_index + i).zfill(5)
            k = data[f"data/{k_name}"][:]
            # select only active re/im parts
            x.append(k[self.active_keys])
        for i in range(self.target_seq_length):
            # read the gt output (next timestep)
            k_name_gt = "timestep_" + str(t_index + self.input_seq_length + i).zfill(5)
            poten_name_gt = "poten_" + str(t_index + self.input_seq_length + i).zfill(5)
            k_gt = data[f"data/{k_name_gt}"][:]
            poten_gt = data[f"data/{poten_name_gt}"][:]
            # select only active re/im parts
            gt.append(k_gt[self.active_keys])
            gt_poten.append(poten_gt)
            flux = data["metadata/fluxes"][t_index + self.input_seq_length + i]

            gt_flux.append(flux)

        # stack to shape (c, t, v1, v2, s, x, y)
        x = np.stack(x, axis=1)
        gt = np.stack(gt, axis=1)
        # stack to shape (x, s, y)
        gt_poten = np.stack(gt_poten, axis=0)
        gt_flux = np.array(gt_flux).squeeze()
        if self.input_seq_length == 1 and self.target_seq_length == 1:
            # sqeeze out time if we only have 1 timestep
            x = x.squeeze(axis=1)
            gt = gt.squeeze(axis=1)
        return x, timestep, gt_poten, gt_flux, gt

    def _normalize(self, x):
        # TODO proper normalization
        if self.normalization == "zscore":
            x_mean = x.mean((1, 2, 3, 4, 5), keepdims=True)
            x_std = x.std((1, 2, 3, 4, 5), keepdims=True)
            return (x - x_mean) / x_std
        if self.normalization == "minmax":
            x_min = x.min((1, 2, 3, 4, 5), keepdims=True)
            x_max = x.max((1, 2, 3, 4, 5), keepdims=True)
            return (x - x_min) / (x_max - x_min)

        return x

    def get_at_time(
        self,
        file_idx: torch.Tensor,
        timestep_idx: torch.Tensor,
        to_fourier: bool = True,
    ):
        updated_index = (file_idx * self.n_samples_per_file + timestep_idx).long()
        sample = self.collate(
            [self.__getitem__(idx, validating=True) for idx in updated_index.tolist()]
        )
        # TODO move somewhere else?
        if self.spatial_ifft and to_fourier:
            if sample.y.ndim == 8:
                sample.y = rearrange(sample.y, "t b c ... -> c t b ...")
            else:
                sample.y = rearrange(sample.y, "b c ... -> c b ...")

            sample.y = torch.complex(real=sample.y[0], imag=sample.y[1])
            sample.y = torch.fft.fftn(sample.y, dim=(-2, -1))
            sample.y = torch.stack([sample.y.real, sample.y.imag]).squeeze()

            if sample.y.ndim == 8:
                sample.y = rearrange(sample.y, "c t b ... -> t b c ...")
            else:
                sample.y = rearrange(sample.y, "c b ... -> b c ...")

        return sample

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
            # flat_file_idx * self.n_samples_per_file + flat_timestep_idx * self.target_seq_length
            flat_file_idx * self.n_samples_per_file
            + flat_timestep_idx
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
            y_poten=torch.stack([sample.y_poten for sample in batch]),
            y_flux=torch.stack([sample.y_flux for sample in batch]),
            timestep=torch.stack([sample.timestep for sample in batch]),
            file_index=torch.stack([sample.file_index for sample in batch]),
            timestep_index=torch.stack([sample.timestep_index for sample in batch]),
        )
