from typing import Type, Optional, List, Tuple, Dict, Sequence
import re
import os
import h5py
from dataclasses import dataclass
from collections import defaultdict
import tqdm
import copy
import torch
import numpy as np
from torch.utils.data import Dataset

import random

from dataset.utils import RunningMeanStd
from utils import expand_as


@dataclass
class CycloneSample:
    df: torch.Tensor
    y_df: torch.Tensor
    phi: torch.Tensor
    y_phi: torch.Tensor
    y_flux: torch.Tensor
    timestep: torch.Tensor
    itg: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    # TODO: add more fields (e.g. params that we can use for conditioning)

    def pin_memory(self):
        if self.df is not None:
            self.df = self.df.pin_memory()
            self.y_df = self.y_df.pin_memory()
        if self.phi is not None:
            self.phi = self.phi.pin_memory()
            self.y_phi = self.y_phi.pin_memory()
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
        bundle_seq_length (int): Extra temporal steps to append to returned samples.
    """

    def __init__(
        self,
        path: str = "/restricteddata/ukaea/gyrokinetics/preprocessed",
        split: str = "train",
        active_keys: Optional[List[str]] = None,
        input_fields: Optional[List[str]] = ["df"],
        trajectories: Optional[List[str]] = None,
        partial_holdouts: Optional[dict] = None,
        normalization: Optional[str] = "zscore",
        normalization_scope: str = "sample",
        normalization_stats: Optional[Dict[str, float]] = None,
        spatial_ifft: bool = True,
        cond_filters: Optional[Dict[str, Sequence]] = None,
        dtype: Type = torch.float32,
        random_seed: int = 42,
        val_ratio: float = 0.1,
        bundle_seq_length: int = 1,
        subsample: int = 1,
        log_transform: bool = False,
        split_into_bands: int = None,
        minmax_beta1: float = 8,
        minmax_beta2: float = 4,
        offset: int = 0,
        separate_zf: bool = False,
    ):
        self.input_fields = input_fields
        self.partial_holdouts = partial_holdouts if partial_holdouts is not None else {}
        assert split in ["train", "val"]
        self.dtype = dtype
        active_keys = active_keys if active_keys is not None else ["re", "im"]
        assert all([a in ["re", "im"] for a in active_keys])
        if split_into_bands:
            self.active_keys = np.arange(split_into_bands * 2)
        else:
            self.active_keys = np.array([{"re": 0, "im": 1}[k] for k in active_keys])
        assert normalization in ["zscore", "minmax", "none", None]
        assert normalization_scope in ["sample", "dataset", "per_mode", "trajectory"]
        self.normalization = normalization if normalization != "none" else None
        if normalization_stats is not None:
            self.norm_stats = normalization_stats
        self.normalization_scope = normalization_scope
        self.cond_filters = cond_filters
        self.subsample = subsample
        self.apply_log = log_transform
        self.minmax_beta1 = minmax_beta1
        self.minmax_beta2 = minmax_beta2
        self.separate_zf = separate_zf

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
            if split_into_bands and normalization_scope != "per_mode":
                n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
                self.files = [
                    (
                        f
                        if f"_ifft_separate_zf{n_bands_tag}.h5" in f
                        else re.sub(r"\.h5$", f"_ifft_separate_zf{n_bands_tag}.h5", f)
                    )
                    for f in self.files
                ]
            elif normalization_scope == "per_mode":
                self.files = [
                    (
                        f
                        if "_ifft_separate_zf_per_mode.h5" in f
                        else re.sub(r"\.h5$", "_ifft_separate_zf_per_mode.h5", f)
                    )
                    for f in self.files
                ]
            else:
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
        if normalization_stats is None:
            self.norm_stats = defaultdict(dict)
        per_file_t_indexes = []
        stats: Dict[str, RunningMeanStd] = {}
        for f_id, f_path in enumerate(self.files):
            # check for holdout samples
            filename = os.path.split(f_path)[-1]
            if spatial_ifft:
                if split_into_bands and normalization_scope != "per_mode":
                    n_bands_tag = (
                        f"_{split_into_bands}bands" if split_into_bands else ""
                    )
                    filename = filename.replace(f"_ifft_separate_zf{n_bands_tag}", "")
                elif normalization_scope == "per_mode":
                    filename = filename.replace("_ifft_separate_zf_per_mode", "")
                else:
                    filename = filename.replace("_ifft", "")
            n_tail_holdout = self.partial_holdouts.get(filename)
            with h5py.File(f_path, "r") as f:
                # read the timesteps
                if n_tail_holdout:
                    if split == "train":
                        timesteps = f["metadata/timesteps"][offset:-n_tail_holdout]
                        orig_t_index = np.arange(len(timesteps))[offset::subsample]
                        timesteps = timesteps[orig_t_index]
                    else:
                        timesteps = f["metadata/timesteps"][offset:-n_tail_holdout:]
                        orig_t_index = np.arange(len(timesteps))[::subsample]
                        timesteps = timesteps[orig_t_index]
                        self.steps_per_file[f_id] = len(
                            f["metadata/timesteps"][:][offset::subsample]
                        )
                else:
                    timesteps = f["metadata/timesteps"][:][offset:]
                    orig_t_index = np.arange(len(timesteps))[::subsample]
                    timesteps = timesteps[orig_t_index]
                # This only works for 1 step training (with pf and rollout aswell)
                n_samples = len(timesteps) - self.bundle_seq_length * 2 + 1
                self.file_num_samples.append(n_samples)
                self.file_num_timesteps.append(len(timesteps))
                per_file_t_indexes.append(orig_t_index)
                if self.norm_stats is not None:
                    # normalization stats
                    for k in self.input_fields:
                        self.norm_stats[f_id][f"{k}_mean"] = f[f"metadata/{k}_mean"][:]
                        self.norm_stats[f_id][f"{k}_std"] = f[f"metadata/{k}_std"][:]
                        self.norm_stats[f_id][f"{k}_min"] = f[f"metadata/{k}_min"][:]
                        self.norm_stats[f_id][f"{k}_max"] = f[f"metadata/{k}_max"][:]
                        if k not in stats and normalization_scope == "dataset":
                            stats[k] = RunningMeanStd(
                                shape=self.norm_stats[f_id][f"{k}_mean"].shape
                            )
                        stats[k].update(
                            self.norm_stats[f_id][f"{k}_mean"],
                            self.norm_stats[f_id][f"{k}_std"] ** 2,
                            self.norm_stats[f_id][f"{k}_min"],
                            self.norm_stats[f_id][f"{k}_max"],
                            count=len(timesteps),
                        )

        self.cumulative_samples = np.cumsum([0] + self.file_num_samples)
        self.length = self.cumulative_samples[-1]
        self.offsets = [offset for _ in range(len(self.files))]
        # calculate offsets if we are in a partial holdout validation dataset
        if split == "val" and self.partial_holdouts:
            for file_idx, file in enumerate(self.files):
                filename = os.path.split(file)[-1]
                if split_into_bands and normalization_scope != "per_mode":
                    n_bands_tag = (
                        f"_{split_into_bands}bands" if split_into_bands else ""
                    )
                    filename = filename.replace(f"_ifft_separate_zf{n_bands_tag}", "")
                elif normalization_scope == "per_mode":
                    filename = filename.replace("_ifft_separate_zf_per_mode", "")
                else:
                    filename = filename.replace("_ifft", "")
                n_tail_holdout = self.partial_holdouts.get(filename)
                if n_tail_holdout:
                    self.offsets[file_idx] = (
                        self.steps_per_file[file_idx] - n_tail_holdout
                    )

        # create mapping from from file and ts index to flat index and vice versa
        self.flat_index_to_file_and_tstep = {}
        self.file_and_tstep_to_flat_index = {}
        for file_idx in range(len(self.files)):
            idxs_before = self.cumulative_samples[file_idx]
            for tstep_idx in range(self.file_num_samples[file_idx]):
                flat_idx = idxs_before + tstep_idx
                t_index = per_file_t_indexes[file_idx][tstep_idx]
                self.flat_index_to_file_and_tstep[flat_idx] = (file_idx, t_index)
                self.file_and_tstep_to_flat_index[(file_idx, t_index)] = flat_idx

        if (
            offset > 0
            and normalization_scope == "dataset"
            and normalization_stats is None
            or (
                separate_zf
                and normalization_scope == "dataset"
                and normalization_stats is None
            )
        ):
            # overwrite dataset-wide stats with new stats for data after offset only
            stats = self._get_norm_stats()

        if normalization_scope == "dataset" and normalization_stats is None:
            for k in self.input_fields:
                self.norm_stats["full"][f"{k}_mean"] = stats[k].mean
                self.norm_stats["full"][f"{k}_std"] = stats[k].var ** (1 / 2)
                self.norm_stats["full"][f"{k}_min"] = stats[k].min
                self.norm_stats["full"][f"{k}_max"] = stats[k].max

        # TODO assume same resolution across all files
        with h5py.File(self.files[0], "r") as f:
            self.resolution = f["metadata/resolution"][:]
            self.phi_resolution = (255, 16, 32)  # TODO

    def _separate_zf(self, x):
        nky = x.shape[-1]
        zf = np.repeat(x.mean(axis=-1, keepdims=True), repeats=nky, axis=-1)
        x = np.concatenate([zf, x - zf], axis=0)
        return x

    def _get_norm_stats(self):
        stats = None
        for t_idx in tqdm.trange(
            0, self.length, 2, desc="Re-computing normalization stats"
        ):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]

            with h5py.File(self.files[file_index], "r") as f:
                sample = self._load_data(f, file_index, t_index)

            x = sample["x"]
            y = sample["gt"]
            if self.separate_zf:
                x = self._separate_zf(x)
                y = self._separate_zf(y)

            if stats is None:
                stats = RunningMeanStd(
                    shape=x.mean((1, 2, 3, 4, 5), keepdims=True).shape
                )

            stats.update(
                x.mean((1, 2, 3, 4, 5), keepdims=True),
                x.var((1, 2, 3, 4, 5), keepdims=True),
                x.min((1, 2, 3, 4, 5), keepdims=True),
                x.max((1, 2, 3, 4, 5), keepdims=True),
            )

            stats.update(
                y.mean((1, 2, 3, 4, 5), keepdims=True),
                y.var((1, 2, 3, 4, 5), keepdims=True),
                y.min((1, 2, 3, 4, 5), keepdims=True),
                y.max((1, 2, 3, 4, 5), keepdims=True),
            )

        return stats

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneSample:
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

        with h5py.File(self.files[file_index], "r") as f:
            sample = self._load_data(f, file_index, t_index)
        # k-fields
        x, gt = sample["x"], sample["gt"]
        if self.separate_zf:
            x = self._separate_zf(x)
            gt = self._separate_zf(gt)
        # accessory fields
        poten, gt_poten, flux = sample["poten"], sample["gt_poten"], sample["gt_flux"]
        # conditioning fields
        timestep, itg = sample["timestep"], sample["itg"]
        if self.normalization is not None and get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)
                gt = (gt - shift) / scale

            if poten is not None:
                poten, shift, scale = self.normalize(file_index, phi=poten)
                gt_poten = (gt_poten - shift) / scale

        return CycloneSample(
            df=torch.tensor(x, dtype=self.dtype) if x is not None else None,
            y_df=torch.tensor(gt, dtype=self.dtype) if gt is not None else None,
            phi=(torch.tensor(poten, dtype=self.dtype) if poten is not None else None),
            y_phi=(
                torch.tensor(gt_poten, dtype=self.dtype)
                if gt_poten is not None
                else None
            ),
            y_flux=torch.tensor(flux, dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            itg=torch.tensor(itg, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
        )

    def _load_data(
        self, data, file_index, t_index
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        original_t_index = t_index + self.offsets[file_index]
        x = []
        gt = []
        poten = []
        gt_poten = []
        gt_flux = []
        for i in range(self.bundle_seq_length):
            # read the input
            k_name = "timestep_" + str(original_t_index + i).zfill(5)
            phi_name = "poten_" + str(original_t_index + i).zfill(5)
            # read the gt output (next timestep)
            k_name_gt = "timestep_" + str(
                original_t_index + self.bundle_seq_length + i
            ).zfill(5)
            phi_name_gt = "poten_" + str(
                original_t_index + self.bundle_seq_length + i
            ).zfill(5)
            if "df" in self.input_fields:
                k = data[f"data/{k_name}"][:]
                k_gt = data[f"data/{k_name_gt}"][:]
                # select only active re/im parts
                x.append(k[self.active_keys])
                gt.append(k_gt[self.active_keys])
            if "phi" in self.input_fields:
                phi = data[f"data/{phi_name}"][:]
                phi_gt = data[f"data/{phi_name_gt}"][:]
                poten.append(phi)
                gt_poten.append(phi_gt)

            # target flux
            flux = data["metadata/fluxes"][
                original_t_index + self.bundle_seq_length + i
            ]

            gt_flux.append(flux)

        sample = {}
        if "df" in self.input_fields:
            # stack to shape (c, t, v1, v2, s, x, y)
            x = np.stack(x, axis=1)
            gt = np.stack(gt, axis=1)
            if self.bundle_seq_length == 1:
                # sqeeze out time if we only have 1 timestep
                x = x.squeeze(axis=1)
                gt = gt.squeeze(axis=1)
        else:
            x, gt = None, None
        if "phi" in self.input_fields:
            # stack to shape (c, t, x, s, y)
            poten = np.stack(poten, axis=1)
            gt_poten = np.stack(gt_poten, axis=1)
            if self.bundle_seq_length == 1:
                poten = poten.squeeze(axis=1)
                gt_poten = gt_poten.squeeze(axis=1)
        else:
            poten, gt_poten = None, None

        sample["x"] = x
        sample["gt"] = gt
        sample["poten"] = poten
        sample["gt_poten"] = gt_poten
        sample["gt_flux"] = np.array(gt_flux).squeeze()
        sample["timestep"] = data["metadata/timesteps"][original_t_index]
        sample["itg"] = data["metadata/ion_temp_grad"][:].squeeze()
        return sample

    def normalize(
        self,
        file_index: int,
        df: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
    ):
        if df is not None:
            field = "df"
            x = df
        elif phi is not None:
            field = "phi"
            x = phi
        else:
            raise ValueError  

        scale, shift = self._get_scale_shift(file_index, field, x)
        return (x - shift) / scale, shift, scale

    def denormalize(
        self,
        file_index: int,
        df: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
    ):
        if df is not None:
            field = "df"
            x = df
        elif phi is not None:
            field = "phi"
            x = phi
        else:
            raise ValueError  
        
        scale, shift = self._get_scale_shift(file_index, field, x)
        return x * scale + shift

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
    
    def _get_scale_shift(self, file_index: int, field: str, x) -> Tuple:
        shift, scale = np.array(0.0), np.array(1.0)
        if self.normalization_scope == "sample":
            # NOTE no denormalization with sample scope
            pass
        else:
            if self.normalization_scope == "dataset":
                key = "full"
            if self.normalization_scope == "trajectory":
                key = file_index

            if self.normalization == "zscore":
                shift = expand_as(self.norm_stats[key][f"{field}_mean"], x)
                scale = expand_as(self.norm_stats[key][f"{field}_std"], x)
            if self.normalization == "minmax":
                x_min = expand_as(self.norm_stats[key][f"{field}_min"], x)
                x_max = expand_as(self.norm_stats[key][f"{field}_max"], x)
                scale = (x_max - x_min) / self.minmax_beta1
                shift = x_min + scale * self.minmax_beta2
        if isinstance(x, torch.Tensor):
            scale = torch.from_numpy(scale).to(x.device)
            shift = torch.from_numpy(shift).to(x.device)
        return scale, shift

    def get_at_time(
        self,
        file_idx: torch.Tensor,
        timestep_idx: torch.Tensor,
        get_normalized: bool = True,
    ):
        # Compute the flat indices from the file indices and time indices
        updated_index = [
            self.file_and_tstep_to_flat_index[i]
            for i in zip(file_idx.tolist(), timestep_idx.tolist())
        ]
        sample = self.collate(
            [self.__getitem__(idx, get_normalized) for idx in updated_index]
        )
        return sample

    def get_timesteps(
        self, file_idx: torch.Tensor, timestep_idx: Optional[torch.Tensor] = None
    ):
        # file_idx: (B,)
        if isinstance(file_idx, int):
            file_idx = torch.tensor([file_idx])
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
            self.file_and_tstep_to_flat_index[(fidx, tidx * self.subsample)]
            for fidx, tidx in zip(flat_file_idx.tolist(), flat_timestep_idx.tolist())
        ]

        timesteps_list = []
        for idx in updated_index:
            # lookup file index and time index
            file_index, t_index = self.flat_index_to_file_and_tstep[idx]

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

    def get_fluxes(self, file_index: int):
        with h5py.File(self.files[file_index], "r") as f:
            fluxes = f["metadata/fluxes"][:]
        return torch.tensor(fluxes[1:])  # discard flux at t=0

    def get_avg_flux(self, file_index: List[int]):
        avg_fluxes = []
        for f in file_index:
            fluxes = self.get_fluxes(f)
            fluxes = fluxes[len(fluxes) // 2 :]  # TODO naive crop linear phase
            avg_fluxes.append(sum(fluxes) / len(fluxes))
        return torch.tensor(avg_fluxes)

    def get_flux_seq(
        self, timestep_index: List[int], file_index: List[int], window: int = 10
    ):
        flux_seq = []
        for f, t in zip(file_index, timestep_index):
            fluxes = self.get_fluxes(f)
            start_idx = max(0, t - window)
            windowed_flux = fluxes[start_idx:t]
            if len(windowed_flux) < window:
                pad_size = window - len(windowed_flux)
                windowed_flux = torch.cat([torch.zeros(pad_size), windowed_flux])

            flux_seq.append(windowed_flux)
        return torch.stack(flux_seq, 0)  # (b t)

    def __len__(self):
        return self.length

    def num_ts(self, file_idx: int):
        return self.file_num_timesteps[file_idx]

    def collate(self, batch):
        # batch is a list of CycloneSamples
        return CycloneSample(
            df=(
                torch.stack([sample.df for sample in batch])
                if batch[0].df is not None
                else None
            ),
            y_df=(
                torch.stack([sample.y_df for sample in batch])
                if batch[0].y_df is not None
                else None
            ),
            phi=(
                torch.stack([sample.phi for sample in batch])
                if batch[0].phi is not None
                else None
            ),
            y_phi=(
                torch.stack([sample.y_phi for sample in batch])
                if batch[0].y_phi is not None
                else None
            ),
            y_flux=torch.stack([sample.y_flux for sample in batch]),
            timestep=torch.stack([sample.timestep for sample in batch]),
            itg=torch.stack([sample.itg for sample in batch]),
            file_index=torch.stack([sample.file_index for sample in batch]),
            timestep_index=torch.stack([sample.timestep_index for sample in batch]),
        )
