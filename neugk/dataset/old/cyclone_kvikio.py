from typing import Tuple, Union, Optional, List, Dict, Sequence, Type

import os
import re
import random
import pickle
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tqdm

import torch
import kvikio
import cupy as cp
from torch.utils._pytree import tree_map

from neugk.utils import RunningMeanStd
from neugk.dataset.cyclone import CycloneDataset, CycloneSample


def read_cupy_bin(file: str, shape: tuple, rank: int = 0, use_kvikio: bool = True):
    if use_kvikio:
        n_elements = np.prod(shape)
        with cp.cuda.Device(rank):
            gpu_array = cp.empty(n_elements, dtype=cp.float32)
            with kvikio.CuFile(file, "r") as f:
                f.read(gpu_array)
        return torch.from_dlpack(gpu_array.reshape(shape))

    else:
        cpu_array = np.fromfile(file, dtype=np.float32)
        return torch.from_numpy(cpu_array.reshape(shape))


class KvikioCycloneDataset(CycloneDataset):
    """Subclass of CycloneDataset configured to read kvikio files instead of h5py."""

    def __init__(
        self,
        path: str = "/restricteddata/ukaea/gyrokinetics/preprocessed_kvikio",
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
        tail_offset: int = 0,
        separate_zf: bool = False,
        decouple_mu: bool = False,
        num_workers: int = 4,
        real_potens: bool = False,
        timestep_std_filter: float = None,
        timestep_std_offset: int = 80,
        rank: int = 0,
        load_with_kvikio: bool = True,
    ):
        self.input_fields = input_fields
        self.partial_holdouts = partial_holdouts if partial_holdouts is not None else {}
        assert split in ["train", "val"]
        self.split = split
        self.dtype = dtype
        active_keys = active_keys if active_keys is not None else ["re", "im"]
        assert all([a in ["re", "im"] for a in active_keys])
        self.active_keys = np.array([{"re": 0, "im": 1}[k] for k in active_keys])
        assert normalization in ["zscore", "minmax", "none", None]
        assert normalization_scope in ["sample", "dataset", "trajectory"]
        normalization = normalization if normalization != "none" else None
        self.normalization = normalization
        if normalization_stats is not None:
            self.stats = normalization_stats
        self.normalization_scope = normalization_scope
        self.cond_filters = cond_filters
        self.subsample = subsample
        self.apply_log = log_transform
        self.minmax_beta1 = minmax_beta1
        self.minmax_beta2 = minmax_beta2
        self.separate_zf = separate_zf
        self.decouple_mu = decouple_mu
        self.num_workers = num_workers
        self.dir = path
        self.bundle_seq_length = bundle_seq_length
        self.timestep_std_filter = timestep_std_filter
        self.timestep_std_offset = timestep_std_offset

        self.rank = rank
        self.load_with_kvikio = load_with_kvikio

        assert not (offset > 0 and timestep_std_filter is not None), (
            f"Cannot use both offset={offset} and timestep_std_filter={timestep_std_filter}. "
            "Set offset=0 when using adaptive std-based filtering."
        )

        # with specified files / pattern
        if trajectories is not None:
            if split == "val" and partial_holdouts:
                self.files = []
                for key in partial_holdouts.keys():
                    self.files.append(os.path.join(self.dir, key))
            else:
                if isinstance(trajectories, str):
                    # pattern for incremental naming
                    match = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", trajectories)
                    if not match:
                        trajectories = [trajectories]
                    prefix, ranges_str, suffix = match.groups()
                    # parse ranges
                    traj_numbers = []
                    for part in ranges_str.split(","):
                        if "-" in part:
                            start, end = map(int, part.split("-"))
                            traj_numbers.extend(range(start, end + 1))
                        else:
                            traj_numbers.append(int(part))

                    trajectories = [f"{prefix}{num}{suffix}" for num in traj_numbers]
                # for compatibility clean up .h5
                trajectories = [t.replace(".h5", "") for t in trajectories]
                self.files = [os.path.join(self.dir, f_name) for f_name in trajectories]

        # take all files in path
        if trajectories is None:
            self.trajectories_fnames = os.listdir(path)
            # changed to look for directories, not .h5 files
            self.files = [
                os.path.join(self.dir, f_name)
                for f_name in self.trajectories_fnames
                if os.path.isdir(os.path.join(self.dir, f_name))
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

        # if set, load preprocessed files with ifft
        self.spatial_ifft = spatial_ifft
        if spatial_ifft:
            real_potens_tag = "_realpotens" if real_potens else ""
            self.files = [
                (f if f"_ifft{real_potens_tag}" in f else f + f"_ifft{real_potens_tag}")
                for f in self.files
            ]

        # remove duplicates
        self.files = list(set(self.files))
        for f in self.files:
            assert os.path.isdir(f), f"Trajectory directory '{f}' does not exist!"

        # get metadata (samples per file and normalization stats)
        self.metadata = {}
        self.file_num_samples = []
        self.file_num_timesteps = []
        self.steps_per_file = {}
        if normalization_stats is None:
            self.stats = {k: defaultdict(dict) for k in input_fields}
        per_file_t_indexes = []
        stats: Dict[str, RunningMeanStd] = {}
        for f_id, f_path in enumerate(self.files):
            # check for holdout samples
            filename = os.path.split(f_path)[-1]
            if spatial_ifft:
                # optional preprocessing filename tags
                filename = filename.replace(f"_ifft_separate_zf", "")
                real_potens_tag = "_realpotens" if real_potens else ""
                filename = filename.replace(f"_ifft{real_potens_tag}", "")
            # number of offset samples on the tail (as holdout or for n_eval_steps)
            self.n_tail_holdout = tail_offset
            if self.partial_holdouts.get(filename, 0) > self.n_tail_holdout:
                self.n_tail_holdout += self.partial_holdouts.get(filename, 0)

            meta_path = os.path.join(f_path, "metadata.pkl")
            with open(meta_path, "rb") as mf:
                meta = pickle.load(mf)
            self.metadata[f_id] = meta

            # read the timesteps
            if self.n_tail_holdout:
                if split == "train":
                    timesteps = meta["timesteps"][offset : -self.n_tail_holdout]
                    orig_t_index = np.arange(len(timesteps))[offset::subsample]
                else:
                    timesteps = meta["timesteps"][offset : -self.n_tail_holdout :]
                    orig_t_index = np.arange(len(timesteps))[::subsample]
                    self.steps_per_file[f_id] = len(
                        meta["timesteps"][offset::subsample]
                    )
            else:
                timesteps = meta["timesteps"][offset:]
                orig_t_index = np.arange(len(timesteps))[::subsample]
            # crop timesteps
            timesteps = timesteps[orig_t_index]
            # This only works for 1 step training (with pf and rollout aswell)
            n_samples = len(timesteps) - self.bundle_seq_length * 2 + 1
            self.file_num_samples.append(n_samples)
            self.file_num_timesteps.append(len(timesteps))
            per_file_t_indexes.append(orig_t_index)

            # TODO assume same resolution across all files
            self.resolution = self.metadata[0]["resolution"]
            self.df_shape = (2, *self.resolution)
            self.phi_resolution = (
                self.resolution[3],
                self.resolution[2],
                self.resolution[4],
            )

            # norm_stats is never None here!
            if normalization_stats is None and normalization_scope == "dataset":
                assert split == "train", "Validation must have normalization_stats."
                # normalization stats read directly from the uncompressed meta dict
                for k in self.input_fields:
                    try:
                        self.stats[k][f_id]["mean"] = meta[f"{k}_mean"]
                        self.stats[k][f_id]["std"] = meta[f"{k}_std"]
                        self.stats[k][f_id]["min"] = meta[f"{k}_min"]
                        self.stats[k][f_id]["max"] = meta[f"{k}_max"]
                    except KeyError:
                        print(f_path)
                        exit(1)
                    if k not in stats:
                        stats[k] = RunningMeanStd(
                            shape=self.stats[k][f_id]["mean"].shape
                        )
                    stats[k].update(
                        self.stats[k][f_id]["mean"],
                        self.stats[k][f_id]["std"] ** 2,
                        self.stats[k][f_id]["min"],
                        self.stats[k][f_id]["max"],
                        count=len(timesteps),
                    )

        self.cumulative_samples = np.cumsum([0] + self.file_num_samples)
        self.length = self.cumulative_samples[-1]
        self.offsets = [offset for _ in range(len(self.files))]
        # calculate offsets if we are in a partial holdout validation dataset
        if split == "val" and self.partial_holdouts:
            for file_idx, file in enumerate(self.files):
                filename = os.path.split(file)[-1]
                # optional preprocessing filename tags
                filename = filename.replace(f"_ifft_separate_zf", "")
                real_potens_tag = "_realpotens" if real_potens else ""
                filename = filename.replace(f"_ifft{real_potens_tag}", "")
                if self.n_tail_holdout:
                    self.offsets[file_idx] = (
                        self.steps_per_file[file_idx] - self.n_tail_holdout
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

        # std-based timestep filtering if enabled
        if self.timestep_std_filter is not None and self.timestep_std_filter > 0:
            self._apply_timestep_std_filter()
            # recalculate length after filtering
            self.length = len(self.flat_index_to_file_and_tstep)

        norm_dataset = normalization is not None and normalization_scope == "dataset"
        if norm_dataset and normalization_stats is None and (offset > 0 or separate_zf):
            # overwrite dataset-wide stats with new stats for data after offset only
            if separate_zf and not offset:
                stats["df"] = self._recompute_stats(key="df")
            else:
                for key in self.input_fields:
                    stats[key] = self._recompute_stats(key=key, offset=self.offsets[0])

        if normalization_scope == "dataset" and normalization_stats is None:
            for k in input_fields:
                self.stats[k]["full"]["mean"] = stats[k].mean.astype(np.float32)
                self.stats[k]["full"]["std"] = (stats[k].var ** 0.5).astype(np.float32)
                self.stats[k]["full"]["min"] = stats[k].min.astype(np.float32)
                self.stats[k]["full"]["max"] = stats[k].max.astype(np.float32)

        # conditional filtering
        if self.cond_filters:
            threshold = offset if offset > 0 else 80
            self.files = [
                f for f in self.files if self._conditioning_filter(f, threshold)
            ]

        if len(self.files) == 0:
            raise RuntimeError(f"No trajectories found! Active filters: {cond_filters}")

    def _separate_zf(self, x):
        if isinstance(x, np.ndarray):
            nky = x.shape[-1]
            zf = np.repeat(x.mean(axis=-1, keepdims=True), repeats=nky, axis=-1)
            return np.concatenate([zf, x - zf], axis=0)
        else:
            nky = x.shape[-1]
            zf = x.mean(dim=-1, keepdim=True).expand(*x.shape[:-1], nky)
            return torch.cat([zf, x - zf], dim=0)

    def _recompute_stats(self, key: str, offset: int = 0):

        def process_t_idx(t_idx, key):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]

            sample = self._load_data(file_index, t_index, use_kvikio=False)
            sample["x"] = sample["x"].numpy()
            sample["gt"] = sample["gt"].numpy()
            sample["phi"] = sample["phi"].numpy()
            sample["y_phi"] = sample["y_phi"].numpy()
            sample["gt_flux"] = sample["gt_flux"].numpy()

            if key == "df":
                x = sample["x"]
                y = sample["gt"]
                if self.decouple_mu:
                    norm_axes = (1, 3, 4, 5)
                else:
                    norm_axes = (1, 2, 3, 4, 5)
            elif key == "phi":
                x, y = sample["phi"], sample["y_phi"]
                if len(x.shape) == 3:
                    x = np.expand_dims(x, 0)
                    y = np.expand_dims(y, 0)
                norm_axes = (1, 2, 3)
            else:
                x = np.array([sample["gt_flux"]], dtype=np.float32)
                y = None
                norm_axes = (0,)

            if self.separate_zf and key == "df":
                x = self._separate_zf(x)
                y = self._separate_zf(y)

            # Compute metrics for x and y
            x_mean = np.mean(x, norm_axes, keepdims=True)
            x_var = np.var(x, norm_axes, keepdims=True)
            x_min = np.min(x, norm_axes, keepdims=True)
            x_max = np.max(x, norm_axes, keepdims=True)

            if y is not None:
                y_mean = np.mean(y, norm_axes, keepdims=True)
                y_var = np.var(y, norm_axes, keepdims=True)
                y_min = np.min(y, norm_axes, keepdims=True)
                y_max = np.max(y, norm_axes, keepdims=True)
            else:
                y_mean = y_var = y_min = y_max = None

            return x_mean, x_var, x_min, x_max, y_mean, y_var, y_min, y_max

        # NOTE: subsample by two for normalization stats!
        t_indices = (
            list(range(0, self.length, 2))
            if key in ["df", "phi"]
            else list(range(self.length))
        )

        # stats filename construction
        std_filter_tag = (
            f"_std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )
        has_mu = "_mu" if self.decouple_mu else ""
        stats_filename = f"{key}_offset{offset}{has_mu}{std_filter_tag}_stats.pkl"
        stats_path = os.path.join(self.dir, stats_filename)

        if os.path.exists(stats_path):
            stats = pickle.load(open(stats_path, "rb"))
        else:
            process_inds = partial(process_t_idx, key=key)
            stats = None
            with ThreadPoolExecutor(self.num_workers) as executor:
                # indices in parallel, collect results in list
                metrics_gen = tqdm.tqdm(
                    executor.map(process_inds, t_indices),
                    total=len(t_indices),
                    desc=f"Re-computing normalization stats for {key}",
                )

                for metrics in metrics_gen:
                    x_mean, x_var, x_min, x_max, y_mean, y_var, y_min, y_max = metrics
                    if stats is None:
                        stats = RunningMeanStd(shape=x_mean.shape)
                    stats.update(x_mean, x_var, x_min, x_max)
                    if y_mean is not None:
                        stats.update(y_mean, y_var, y_min, y_max)

            pickle.dump(stats, open(stats_path, "wb"))
            print(f"Saved recomputed stats to {stats_path}")
        return stats

    def __getitem__(
        self, index: Union[int, Tuple[int, int]], get_normalized: bool = True
    ) -> CycloneSample:
        if isinstance(index, int):
            file_index, t_index = self.flat_index_to_file_and_tstep[index]
        elif isinstance(index, Tuple):
            file_index, t_index = index

        sample = self._load_data(file_index, t_index, use_kvikio=self.load_with_kvikio)

        # standard preprocessing from parent
        x, gt = sample["x"], sample["gt"]
        if self.separate_zf:
            x = self._separate_zf(x)
            gt = self._separate_zf(gt)

        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        timestep = sample["timestep"]
        itg, dg, s_hat, q = sample["itg"], sample["dg"], sample["s_hat"], sample["q"]
        geometry = sample["geometry"]

        if self.normalization is not None and get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)
                gt = (gt - shift) / scale
            if phi is not None:
                phi, shift, scale = self.normalize(file_index, phi=phi)
                y_phi = (y_phi - shift) / scale
            if flux is not None:
                flux, *_ = self.normalize(file_index, flux=flux)

        return CycloneSample(
            df=x.to(dtype=self.dtype) if x is not None else None,
            y_df=gt.to(dtype=self.dtype) if gt is not None else None,
            phi=phi.to(dtype=self.dtype) if phi is not None else None,
            y_phi=y_phi.to(dtype=self.dtype) if y_phi is not None else None,
            y_flux=flux.to(dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            geometry=tree_map(lambda x: torch.as_tensor(x, dtype=self.dtype), geometry),
            itg=torch.tensor(itg, dtype=self.dtype),
            dg=torch.tensor(dg, dtype=self.dtype),
            s_hat=torch.tensor(s_hat, dtype=self.dtype),
            q=torch.tensor(q, dtype=self.dtype),
        )

    def _load_data(
        self, file_index: int, t_index: int, use_kvikio: bool = True
    ) -> dict:
        original_t_index = t_index + self.offsets[file_index]
        meta = self.metadata[file_index]
        data_dir = os.path.join(self.files[file_index], "data")

        x, gt, poten, y_poten, gt_flux = [], [], [], [], []

        for i in range(self.bundle_seq_length):
            t_str = str(original_t_index + i).zfill(5)
            t_str_gt = str(original_t_index + self.bundle_seq_length + i).zfill(5)

            kfile = os.path.join(data_dir, f"timestep_{t_str}.bin")
            kfile_gt = os.path.join(data_dir, f"timestep_{t_str_gt}.bin")
            phifile = os.path.join(data_dir, f"poten_{t_str}.bin")
            phifile_gt = os.path.join(data_dir, f"poten_{t_str_gt}.bin")

            if "df" in self.input_fields:
                k = read_cupy_bin(
                    file=kfile,
                    shape=self.df_shape,
                    rank=self.rank,
                    use_kvikio=use_kvikio,
                )
                k_gt = read_cupy_bin(
                    file=kfile_gt,
                    shape=self.df_shape,
                    rank=self.rank,
                    use_kvikio=use_kvikio,
                )

                if all(self.active_keys == np.array([0, 1])):
                    x.append(k)
                    gt.append(k_gt)
                else:
                    x.append(k[self.active_keys])
                    gt.append(k_gt[self.active_keys])

            if "phi" in self.input_fields:
                phi = read_cupy_bin(
                    file=phifile,
                    shape=self.phi_resolution,
                    rank=self.rank,
                    use_kvikio=use_kvikio,
                )
                phi_gt = read_cupy_bin(
                    file=phifile_gt,
                    shape=self.phi_resolution,
                    rank=self.rank,
                    use_kvikio=use_kvikio,
                )

                poten.append(phi)
                y_poten.append(phi_gt)

            flux = meta["fluxes"][original_t_index + self.bundle_seq_length + i]
            gt_flux.append(flux)

        sample = {}
        if "df" in self.input_fields:
            if self.bundle_seq_length == 1:
                x, gt = x[0], gt[0]
            else:
                x, gt = torch.stack(x, axis=1), torch.stack(gt, axis=1)
        else:
            x, gt = None, None

        if "phi" in self.input_fields:
            if self.bundle_seq_length == 1:
                poten, y_poten = poten[0], y_poten[0]
            else:
                poten, y_poten = torch.stack(poten, axis=1), torch.stack(
                    y_poten, axis=1
                )
        else:
            poten, y_poten = None, None

        sample["x"] = x
        sample["gt"] = gt
        sample["phi"] = poten
        sample["y_phi"] = y_poten
        sample["gt_flux"] = torch.tensor(gt_flux).squeeze()

        sample["timestep"] = meta["timesteps"][original_t_index]
        sample["itg"] = meta["ion_temp_grad"]
        sample["dg"] = meta["density_grad"]
        sample["s_hat"] = meta["s_hat"]
        sample["q"] = meta["q"]
        sample["geometry"] = meta["geometry"]

        return sample

    def get_fluxes(self, file_index: int):
        fluxes = self.metadata[file_index]["fluxes"]
        return torch.tensor(fluxes[1:])

    def _conditioning_filter(self, fname: str, offset: int) -> bool:
        meta_path = os.path.join(fname, "metadata.pkl")
        with open(meta_path, "rb") as mf:
            meta = pickle.load(mf)

        for cond_name, cond_range in self.cond_filters.items():
            if len(cond_name.split("_")) > 1:
                where, cond_name = cond_name.split("_")

            if cond_name in meta:
                cond = meta[cond_name]
                if not isinstance(cond_range[0], Sequence):
                    cond_range = [cond_range]

                # check filters
                if cond_name == "fluxes":
                    if where == "first":
                        cond = np.mean(cond[:offset])
                    else:
                        cond = np.mean(cond[-offset:])

                if not any(min_ <= cond <= max_ for min_, max_ in cond_range):
                    return False
            else:
                raise UserWarning(f"`{cond_name}` not found in metadata {fname}.")

        return True

    def get_timesteps(
        self,
        file_idx: torch.Tensor,
        timestep_idx: Optional[torch.Tensor] = None,
        offset: int = 0,
    ):
        if isinstance(file_idx, int):
            file_idx = torch.tensor([file_idx])
        file_idx = file_idx.cpu().long()
        if timestep_idx is None:
            timestep_idx = torch.stack(
                [
                    torch.arange(self.num_ts(file_idx[i]) - 1)
                    for i in range(file_idx.shape[0])
                ],
                dim=0,
            )
        if isinstance(timestep_idx, int):
            timestep_idx = torch.tensor([timestep_idx])
        timestep_idx = timestep_idx.cpu().long()

        B = file_idx.shape[0]
        if timestep_idx.dim() == 1:
            flat_file_idx = file_idx
            flat_timestep_idx = timestep_idx
        else:
            N = timestep_idx.shape[1]
            file_idx_expanded = file_idx.unsqueeze(1).expand(B, N)

            flat_file_idx = file_idx_expanded.flatten()
            flat_timestep_idx = timestep_idx.flatten()

        updated_index = [
            self.file_and_tstep_to_flat_index[(fidx, tidx * self.subsample)]
            for fidx, tidx in zip(flat_file_idx.tolist(), flat_timestep_idx.tolist())
        ]

        timesteps_list = []
        for idx in updated_index:
            file_index, t_index = self.flat_index_to_file_and_tstep[idx]

            timesteps_array = self.metadata[file_index]["timesteps"][offset:]
            step_value = timesteps_array[t_index]

            timesteps_list.append(torch.tensor(step_value, dtype=self.dtype))

        timesteps_tensor = torch.stack(timesteps_list)
        if timestep_idx.dim() > 1:
            timesteps_tensor = timesteps_tensor.view(B, -1)

        return timesteps_tensor

    def _apply_timestep_std_filter(self):
        k = self.timestep_std_filter
        ref_offset = self.timestep_std_offset
        valid_flat_indices = {}
        flat_idx_counter = 0

        for file_idx in range(len(self.files)):
            fluxes = self.metadata[file_idx]["fluxes"]

            # mean and std
            ref_fluxes = fluxes[ref_offset:]
            ref_mean = np.mean(ref_fluxes)
            ref_std = np.std(ref_fluxes)

            # bounds
            lower = ref_mean - k * ref_std
            upper = ref_mean + k * ref_std

            # filter timesteps
            old_indices = [
                (flat_idx, file_idx_stored, t_idx)
                for flat_idx, (
                    file_idx_stored,
                    t_idx,
                ) in self.flat_index_to_file_and_tstep.items()
                if file_idx_stored == file_idx
            ]

            for _, _, t_idx in old_indices:
                original_t_index = t_idx + self.offsets[file_idx]
                flux_idx = original_t_index + self.bundle_seq_length

                # bounds check
                if flux_idx >= len(fluxes):
                    flux_idx = len(fluxes) - 1

                flux_at_t = fluxes[flux_idx]

                if lower <= flux_at_t <= upper:
                    valid_flat_indices[flat_idx_counter] = (file_idx, t_idx)
                    flat_idx_counter += 1

        self.flat_index_to_file_and_tstep = valid_flat_indices
        self.file_and_tstep_to_flat_index = {
            (file_idx, t_idx): flat_idx
            for flat_idx, (file_idx, t_idx) in valid_flat_indices.items()
        }
