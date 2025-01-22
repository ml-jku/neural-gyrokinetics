from typing import Type, Optional, List, Tuple, Dict, Sequence
import re
import os
import h5py
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict

from einops import rearrange
import torch
import numpy as np
from torch.utils.data import Dataset

import random

from utils import expand_as


@dataclass
class CycloneSample:
    x: torch.Tensor
    y: torch.Tensor
    y_poten: torch.Tensor
    y_flux: torch.Tensor
    timestep: torch.Tensor
    itg: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    # TODO: add more fields (e.g. params that we can use for conditioning)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self


class CycloneDataset(Dataset):
    """Dataset for preprocessed gyrokinetics datasets.

    Args:
        path (str): Path to the preprocessed .h5 files
        split (str): 'train' or 'val'. When set to 'val', it enables partial holdouts.
        active_keys (optional, list): List of channels to use ('re', 'im').
        trajectories (optional, list): Optional list with picked trajectories.
        partial_holdouts (optional, dict): TODO
        normalization (optional, str): Normalization type ('none', 'zscore', 'minmax').
        normalization_scope (optional, str): Normalization scope ('sample', 'dataset').
        spatial_ifft (bool): Whether to use IFFT data (must be preprocessed).
        cond_filters (optional, dict): Filters on conditioning params (as a dictionary
                    of {cond_name: [(cond_min, cond_max), ... (cond_min, cond_max)]})
        dtype (type): Cast returned samples to this torch dtype.
        random_seed (int): Seed initialization for random splits.
        val_ratio (float): Validation data ration for random splits.
        in_memory (bool): Whether to load the dataset in RAM.
        bundle_seq_length (int): Extra temporal steps to append to returned samples.
    """

    def __init__(
        self,
        path: str = "/restricteddata/ukaea/gyrokinetics/preprocessed",
        split: str = "train",
        active_keys: Optional[List[str]] = None,
        trajectories: Optional[List[str]] = None,
        partial_holdouts: Optional[dict] = None,
        normalization: Optional[str] = "zscore",
        normalization_scope: str = "sample",
        spatial_ifft: bool = True,
        cond_filters: Optional[Dict[str, Sequence]] = None,
        dtype: Type = torch.float32,
        random_seed: int = 42,
        val_ratio: float = 0.1,
        in_memory: bool = False,
        bundle_seq_length: int = 1,
    ):
        self.partial_holdouts = partial_holdouts if partial_holdouts is not None else {}
        assert split in ["train", "val"]
        self.dtype = dtype

        active_keys = active_keys if active_keys is not None else ["re", "im"]
        assert all([a in ["re", "im"] for a in active_keys])
        self.active_keys = np.array([{"re": 0, "im": 1}[k] for k in active_keys])
        assert normalization in ["zscore", "minmax", "none", None]
        assert normalization_scope in ["sample", "dataset"]
        self.normalization = normalization if normalization != "none" else None
        self.normalization_scope = normalization_scope
        self.cond_filters = cond_filters

        self.in_memory = in_memory
        self.dir = path

        self.bundle_seq_length = bundle_seq_length

        if trajectories is None:
            self.trajectories_fnames = os.listdir(path)
            self.files = [
                os.path.join(self.dir, f_name)
                for f_name in self.trajectories_fnames
                if ".h5" in f_name
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
            train_idx = list(perm)
            if split == "train":
                self.files = [self.files[i] for i in train_idx]
            if split == "val":
                self.files = [self.files[i] for i in val_idx]
        else:
            if split == "val" and partial_holdouts:
                self.files = []
                for key in partial_holdouts.keys():
                    self.files.append(os.path.join(self.dir, key))
            else:
                self.files = [os.path.join(self.dir, f_name) for f_name in trajectories]

        # if set, load preprocessed files with ifft
        self.spatial_ifft = spatial_ifft
        if spatial_ifft:
            self.files = [
                f if "_ifft.h5" in f else re.sub(r"\.h5$", "_ifft.h5", f)
                for f in self.files
            ]

        # remove duplicates
        self.files = list(set(self.files))
        for f in self.files:
            assert os.path.isfile(f), f"Trajectory  '{f}' does not exist!"

        # filter files based on conditioning
        if self.cond_filters:
            self.files = [f for f in self.files if self._conditioning_filter(f)]
        
        if len(self.files) == 0:
            raise RuntimeError(f"No trajectories found! Active filters: {cond_filters}")

        # get metadata (samples per file and normalization stats)
        self.file_num_samples = []
        self.file_num_timesteps = []
        self.steps_per_file = {}
        self.dataset_stats = defaultdict(dict)
        for file_idx, file_path in enumerate(self.files):
            # check for holdout samples
            filename = os.path.split(file_path)[-1]
            n_tail_holdout = self.partial_holdouts.get(filename)
            with h5py.File(file_path, "r") as f:
                # read the timesteps
                if n_tail_holdout:
                    if split == "train":
                        timesteps = f["metadata/timesteps"][:-n_tail_holdout]
                    else:
                        timesteps = f["metadata/timesteps"][-n_tail_holdout:]
                        self.steps_per_file[file_idx] = len(f["metadata/timesteps"][:])
                else:
                    timesteps = f["metadata/timesteps"][:]
                # This only works for 1 step training (with pf and rollout aswell)
                n_samples = len(timesteps) - self.bundle_seq_length * 2 + 1
                self.file_num_samples.append(n_samples)
                self.file_num_timesteps.append(len(timesteps))
                # normalization stats
                self.dataset_stats[file_idx]["mean"] = f["metadata/k_mean"][:]
                self.dataset_stats[file_idx]["std"] = f["metadata/k_std"][:]
                self.dataset_stats[file_idx]["min"] = f["metadata/k_min"][:]
                self.dataset_stats[file_idx]["max"] = f["metadata/k_max"][:]

        self.cumulative_samples = np.cumsum([0] + self.file_num_samples)
        self.length = self.cumulative_samples[-1]

        # create mapping from from file and ts index to flat index and vice versa
        self.flat_index_to_file_and_tstep = {}
        self.file_and_tstep_to_flat_index = {}
        for file_idx in range(len(self.files)):
            idxs_before = self.cumulative_samples[file_idx]
            for tstep_idx in range(self.file_num_samples[file_idx]):
                flat_idx = idxs_before + tstep_idx
                self.flat_index_to_file_and_tstep[flat_idx] = (file_idx, tstep_idx)
                self.file_and_tstep_to_flat_index[(file_idx, tstep_idx)] = flat_idx

        self.offsets = [0 for _ in range(len(self.files))]
        # calculate offsets if we are in a partial holdout validation dataset
        if split == "val" and self.partial_holdouts:
            for file_idx, file in enumerate(self.files):
                filename = os.path.split(file)[-1]
                n_tail_holdout = self.partial_holdouts.get(filename)
                if n_tail_holdout:
                    self.offsets[file_idx] = (
                        self.steps_per_file[file_idx] - n_tail_holdout
                    )

        if self.in_memory:
            # load all timesteps into a dict of dicts
            self.data = {}
            for file_idx, file in enumerate(self.files):
                file_dict = {}
                # check for holdout samples
                filename = os.path.split(file)[-1]
                n_tail_holdout = self.partial_holdouts.get(filename)
                with h5py.File(file, "r") as f:
                    # read the timesteps and fluxes dataset
                    timesteps = f["metadata/timesteps"][:]
                    fluxes = f["metadata/fluxes"][:]
                    file_dict["metadata/timesteps"] = timesteps
                    file_dict["metadata/fluxes"] = fluxes
                    itg = f["metadata/ion_temp_grad"][:]
                    file_dict["metadata/ion_temp_grad"] = itg
                    # read in all the data points
                    if n_tail_holdout:
                        if split == "train":
                            _range = range(timesteps[:-n_tail_holdout])
                        else:
                            _range = range(timesteps[-n_tail_holdout:])
                    else:
                        _range = range(len(file_dict["metadata/timesteps"]))
                    for t_index in tqdm(_range):
                        original_t_index = t_index + self.offsets[file_idx]
                        k_name = "timestep_" + str(original_t_index).zfill(5)
                        poten_name = "poten_" + str(original_t_index).zfill(5)
                        k_data = f[f"data/{k_name}"][:]
                        poten_data = f[f"data/{poten_name}"][:]
                        # cache samples
                        file_dict[f"data/{k_name}"] = k_data
                        file_dict[f"data/{poten_name}"] = poten_data
                self.data[file_idx] = file_dict

        # TODO assume same resolution across all files
        with h5py.File(self.files[0], "r") as f:
            self.resolution = f["metadata/resolution"][:]

    def __getitem__(self, index: int) -> CycloneSample:
        """
        Args:
            index (int): Flat index with dataset ordering.

        Returns:
            CycloneSample: dataclass
                - x (torch.Tensor): model input, shape `(c, bundle, v1, v2, s, x, y)`
                - y (torch.Tensor): target, shape `(c, bundle, v1, v2, s, x, y)`
                - y_poten (torch.Tensor): target potential field, shape `(x, s, y)`
                - y_flux (torch.Tensor): target flux, shape `()`
                - timestep (torch.Tensor): physical timestep, shape `()`
                - itg (torch.Tensor): ion temperature gradient, shape `()`
                - file_index (torch.Tensor): accessory file index, shape `()`
                - timestep_index (torch.Tensor): accessory timestep index, shape `()`
        """
        # lookup file index and time index from flat index
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if self.in_memory:
            sample = self._load_data(self.data[file_index], file_index, t_index)
        else:
            with h5py.File(self.files[file_index], "r") as f:
                sample = self._load_data(f, file_index, t_index)
        # k-fields
        x, gt = sample["x"], sample["gt"]
        # accessory fields
        poten, flux = sample["gt_poten"], sample["gt_flux"]
        # conditioning fields
        timestep, itg = sample["timestep"], sample["itg"]
        if self.normalization is not None:
            x = self._normalize(x, file_index)
            gt = self._normalize(gt, file_index)

        return CycloneSample(
            x=torch.tensor(x, dtype=self.dtype),
            y=torch.tensor(gt, dtype=self.dtype),
            y_poten=torch.tensor(poten, dtype=self.dtype),
            y_flux=torch.tensor(flux, dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            itg=torch.tensor(itg, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=self.dtype),
            timestep_index=torch.tensor(t_index, dtype=self.dtype),
        )

    def _load_data(
        self, data, file_index, t_index
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        original_t_index = t_index + self.offsets[file_index]
        x = []
        gt = []
        gt_poten = []
        gt_flux = []
        for i in range(self.bundle_seq_length):
            # read the input
            k_name = "timestep_" + str(original_t_index + i).zfill(5)
            k = data[f"data/{k_name}"][:]
            # select only active re/im parts
            x.append(k[self.active_keys])
        for i in range(self.bundle_seq_length):
            # read the gt output (next timestep)
            k_name_gt = "timestep_" + str(
                original_t_index + self.bundle_seq_length + i
            ).zfill(5)
            poten_name_gt = "poten_" + str(
                original_t_index + self.bundle_seq_length + i
            ).zfill(5)
            k_gt = data[f"data/{k_name_gt}"][:]
            poten_gt = data[f"data/{poten_name_gt}"][:]
            # select only active re/im parts
            gt.append(k_gt[self.active_keys])
            gt_poten.append(poten_gt)
            flux = data["metadata/fluxes"][
                original_t_index + self.bundle_seq_length + i
            ]

            gt_flux.append(flux)

        sample = {}
        # stack to shape (c, t, v1, v2, s, x, y)
        x = np.stack(x, axis=1)
        gt = np.stack(gt, axis=1)
        if self.bundle_seq_length == 1:
            # sqeeze out time if we only have 1 timestep
            x = x.squeeze(axis=1)
            gt = gt.squeeze(axis=1)
        sample["x"] = x
        sample["gt"] = gt
        # stack to shape (x, s, y)
        sample["gt_poten"] = np.stack(gt_poten, axis=0)
        sample["gt_flux"] = np.array(gt_flux).squeeze()
        sample["timestep"] = data["metadata/timesteps"][t_index]
        sample["itg"] = data["metadata/ion_temp_grad"][:].squeeze()
        return sample

    def _normalize(self, x, file_index):
        shift, scale = 0.0, 1.0
        if self.normalization_scope == "sample":
            if self.normalization == "zscore":
                shift = x.mean((1, 2, 3, 4, 5), keepdims=True)
                scale = x.std((1, 2, 3, 4, 5), keepdims=True)
            if self.normalization == "minmax":
                x_min = x.min((1, 2, 3, 4, 5), keepdims=True)
                x_max = x.max((1, 2, 3, 4, 5), keepdims=True)
                shift = x_min
                scale = x_max - x_min
        if self.normalization_scope == "dataset":
            if self.normalization == "zscore":
                shift = expand_as(self.dataset_stats[file_index]["mean"], x)
                scale = expand_as(self.dataset_stats[file_index]["std"], x)
            if self.normalization == "minmax":
                x_min = expand_as(self.dataset_stats[file_index]["min"], x)
                x_max = expand_as(self.dataset_stats[file_index]["max"], x)
                shift = x_min
                scale = x_max - x_min

        return (x - shift) / scale

    def _conditioning_filter(self, fname: str) -> bool:
        with h5py.File(fname, "r") as f:
            for cond_name, cond_range in self.cond_filters.items():
                if f"metadata/{cond_name}" in f:
                    cond = f[f"metadata/{cond_name}"][:]
                    if not isinstance(cond_range[0], Sequence):
                        cond_range = [cond_range]
                    # check filters
                    if not any(min_ <= cond <= max_ for min_, max_ in cond_range):
                        return False
                else:
                    raise UserWarning(f"`{cond_name}` not found in metadata {fname}.")

            return True

    def get_at_time(
        self,
        file_idx: torch.Tensor,
        timestep_idx: torch.Tensor,
        to_fourier: bool = False,
    ):
        # Compute the flat indices from the file indices and time indices
        updated_index = [
            self.file_and_tstep_to_flat_index[i]
            for i in zip(file_idx.tolist(), timestep_idx.tolist())
        ]
        sample = self.collate([self.__getitem__(idx) for idx in updated_index])
        # TODO move somewhere else?
        if self.spatial_ifft and to_fourier:
            if sample.y.ndim == 8:
                sample.y = rearrange(sample.y, "t b c ... -> c t b ...")
            else:
                sample.y = rearrange(sample.y, "b c ... -> c b ...")

            sample.y = torch.complex(real=sample.y[0], imag=sample.y[1])
            sample.y = torch.fft.fftn(sample.y, dim=(-2, -1))
            sample.y = torch.stack([sample.y.real, sample.y.imag])

            if sample.y.ndim == 8:
                sample.y = rearrange(sample.y, "c t b ... -> t b c ...")
            else:
                sample.y = rearrange(sample.y, "c b ... -> b c ...")

        return sample

    def get_timesteps(
        self, file_idx: torch.Tensor, timestep_idx: Optional[torch.Tensor] = None
    ):
        # file_idx: (B,)
        file_idx = file_idx.cpu().long()
        # all timesteps in file
        if timestep_idx is None:
            timestep_idx = torch.stack(
                [
                    torch.arange(self.num_ts(file_idx[i]) - 1)
                    for i in range(file_idx.shape[0])
                ],
                dim=0,
            )
        # timestep_idx: (B, N)
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

        # Compute the flat indices from the file indices and time indices
        updated_index = [
            self.file_and_tstep_to_flat_index[i]
            for i in zip(flat_file_idx.tolist(), flat_timestep_idx.tolist())
        ]

        timesteps_list = []
        for idx in updated_index:
            # lookup file index and time index
            file_index, t_index = self.flat_index_to_file_and_tstep[idx]

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

    def num_ts(self, file_idx: int):
        return self.file_num_timesteps[file_idx]

    def collate(self, batch):
        # batch is a list of CycloneSamples
        return CycloneSample(
            x=torch.stack([sample.x for sample in batch]),
            y=torch.stack([sample.y for sample in batch]),
            y_poten=torch.stack([sample.y_poten for sample in batch]),
            y_flux=torch.stack([sample.y_flux for sample in batch]),
            timestep=torch.stack([sample.timestep for sample in batch]),
            itg=torch.stack([sample.itg for sample in batch]),
            file_index=torch.stack([sample.file_index for sample in batch]),
            timestep_index=torch.stack([sample.timestep_index for sample in batch]),
        )
