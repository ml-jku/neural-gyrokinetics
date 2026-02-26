from typing import Type, Optional, List, Tuple, Dict, Sequence, Union, Any

import re
import os
import tqdm
import pickle
import hashlib

import random
from functools import partial
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils._pytree import tree_map

from neugk.utils import RunningMeanStd, expand_as
from neugk.dataset.backend import DataBackend


@dataclass
class CycloneSample:
    df: torch.Tensor
    y_df: torch.Tensor
    phi: torch.Tensor
    y_phi: torch.Tensor
    y_flux: torch.Tensor
    timestep: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    # conditioning
    itg: torch.Tensor
    dg: torch.Tensor
    s_hat: torch.Tensor
    q: torch.Tensor
    # geometric tensors for integrals
    geometry: Optional[Dict[str, torch.Tensor]] = None

    def pin_memory(self):
        if self.df is not None:
            self.df = self.df.pin_memory()
            self.y_df = self.y_df.pin_memory()
        if self.phi is not None:
            self.phi = self.phi.pin_memory()
            self.y_phi = self.y_phi.pin_memory()
        return self


class CycloneDataset(Dataset):
    def __init__(
        self,
        backend: DataBackend,
        path: str = "/restricteddata/ukaea/gyrokinetics/preprocessed",
        split: str = "train",
        active_keys: Optional[List[str]] = None,
        fields_to_load: Optional[List[str]] = ["df"],
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
    ):
        self.fields_to_load = fields_to_load
        self.partial_holdouts = partial_holdouts if partial_holdouts is not None else {}
        assert split in ["train", "val"]
        self.split = split
        self.dtype = dtype
        active_keys = active_keys if active_keys is not None else ["re", "im"]
        assert all([a in ["re", "im"] for a in active_keys])
        if split_into_bands:
            self.active_keys = np.arange(split_into_bands * 2)
        else:
            self.active_keys = np.array([{"re": 0, "im": 1}[k] for k in active_keys])
        assert normalization_scope in ["sample", "dataset", "trajectory"]
        assert set(fields_to_load).issubset(set(normalization.keys())), "Normalization must be specified for all fields to load"
        self.normalizers = normalization
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

        self.backend = backend

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
                    match = re.match(r"^(.*?)\{([^}]+)\}(.*?)$", trajectories)
                    if not match:
                        trajectories = [trajectories]
                    prefix, ranges_str, suffix = match.groups()
                    traj_numbers = []
                    for part in ranges_str.split(","):
                        if "-" in part:
                            start, end = map(int, part.split("-"))
                            traj_numbers.extend(range(start, end + 1))
                        else:
                            traj_numbers.append(int(part))

                    trajectories = [f"{prefix}{num}{suffix}" for num in traj_numbers]

                self.files = [os.path.join(self.dir, f_name) for f_name in trajectories]

        # take all files in path
        if trajectories is None:
            self.trajectories_fnames = os.listdir(path)
            self.files = [
                os.path.join(self.dir, f_name)
                for f_name in self.trajectories_fnames
                if self.backend.is_valid(os.path.join(self.dir, f_name))
            ]
            # shuffle and split
            random.seed(random_seed)
            random.shuffle(self.files)
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

        # format paths for spatial ifft
        self.spatial_ifft = spatial_ifft
        self.files = [
            self.backend.format_path(f, spatial_ifft, split_into_bands, real_potens)
            for f in self.files
        ]

        # remove duplicates and check validity
        self.files = list(set(self.files))
        existing_files = []
        for f in self.files:
            assert self.backend.is_valid(f), f"'{f}' not valid for {self.backend}!"

        self.metadata = {}

        # apply condition filters before building indices
        if self.cond_filters:
            threshold = offset if offset > 0 else 80
            self.files = [
                f for f in self.files if self._conditioning_filter(f, threshold)
            ]

        if len(self.files) == 0:
            raise RuntimeError(f"no trajectories found! active filters: {cond_filters}")

        # load unified metadata
        self.file_num_samples = []
        self.file_num_timesteps = []
        self.steps_per_file = {}
        if normalization_stats is None:
            self.stats = {k: defaultdict(dict) for k in fields_to_load}
        per_file_t_indexes = []
        stats: Dict[str, RunningMeanStd] = {}

        for f_id, f_path in enumerate(self.files):
            filename = os.path.split(f_path)[-1]
            if spatial_ifft:
                n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
                filename = filename.replace(f"_ifft_separate_zf{n_bands_tag}", "")
                real_potens_tag = "_realpotens" if real_potens else ""
                filename = filename.replace(f"_ifft{real_potens_tag}", "")

            self.n_tail_holdout = tail_offset
            if self.partial_holdouts.get(filename, 0) > self.n_tail_holdout:
                self.n_tail_holdout += self.partial_holdouts.get(filename, 0)

            # unify metadata read
            meta = self.backend.read_metadata(f_path, self.fields_to_load)
            self.metadata[f_id] = meta

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

            timesteps = timesteps[orig_t_index]
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
                assert split == "train", "validation must have normalization_stats"
                for k in self.fields_to_load:
                    if k not in stats:
                        stats[k] = RunningMeanStd()
                    mean = meta[f"{k}_mean"]
                    std = meta[f"{k}_std"]
                    if self.normalizers[k]["agg_axes"]:
                        # aggregate along specified dimensions
                        mean, std = stats[k].aggregate_stats(mean, std, agg_axes=tuple(self.normalizers[k]["agg_axes"]))

                    self.stats[k][f_id]["mean"] = mean
                    self.stats[k][f_id]["std"] = std
                    self.stats[k][f_id]["min"] = meta[f"{k}_min"]
                    self.stats[k][f_id]["max"] = meta[f"{k}_max"]
                    
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

        if split == "val" and self.partial_holdouts:
            for file_idx, file in enumerate(self.files):
                filename = os.path.split(file)[-1]
                n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
                filename = filename.replace(f"_ifft_separate_zf{n_bands_tag}", "")
                real_potens_tag = "_realpotens" if real_potens else ""
                filename = filename.replace(f"_ifft{real_potens_tag}", "")
                if self.n_tail_holdout:
                    self.offsets[file_idx] = (
                        self.steps_per_file[file_idx] - self.n_tail_holdout
                    )

        self.flat_index_to_file_and_tstep = {}
        self.file_and_tstep_to_flat_index = {}
        for file_idx in range(len(self.files)):
            idxs_before = self.cumulative_samples[file_idx]
            for tstep_idx in range(self.file_num_samples[file_idx]):
                flat_idx = idxs_before + tstep_idx
                t_index = per_file_t_indexes[file_idx][tstep_idx]
                self.flat_index_to_file_and_tstep[flat_idx] = (file_idx, t_index)
                self.file_and_tstep_to_flat_index[(file_idx, t_index)] = flat_idx

        if self.timestep_std_filter is not None and self.timestep_std_filter > 0:
            self._apply_timestep_std_filter()
            self.length = len(self.flat_index_to_file_and_tstep)

        norm_dataset = normalization is not None and normalization_scope == "dataset"
        if norm_dataset and normalization_stats is None and (offset > 0 or separate_zf):
            if separate_zf and not offset:
                stats["df"] = self._recompute_stats(key="df")
            else:
                for key in self.fields_to_load:
                    stats[key] = self._recompute_stats(key=key, offset=self.offsets[0])

        if normalization_scope == "dataset" and normalization_stats is None:
            for k in fields_to_load:
                self.stats[k]["full"]["mean"] = stats[k].mean.astype(np.float32)
                self.stats[k]["full"]["std"] = (stats[k].var ** 0.5).astype(np.float32)
                self.stats[k]["full"]["min"] = stats[k].min.astype(np.float32)
                self.stats[k]["full"]["max"] = stats[k].max.astype(np.float32)

    def _separate_zf(self, x):
        if isinstance(x, np.ndarray):
            nky = x.shape[-1]
            zf = np.repeat(x.mean(axis=-1, keepdims=True), repeats=nky, axis=-1)
            return np.concatenate([zf, x - zf], axis=0)
        else:
            nky = x.shape[-1]
            zf = x.mean(dim=-1, keepdim=True).expand(*x.shape[:-1], nky)
            return torch.cat([zf, x - zf], dim=0)

    def _recompute_stats(
        self, key: str, offset: int = 0, prefix: str = "", suffix: str = ""
    ):

        def process_t_idx(t_idx, key):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]
            with self.backend.open(self.files[file_index]) as f:
                sample = self._load_data(f, file_index, t_index)

            # force to numpy for math
            if isinstance(sample.get("x"), torch.Tensor):
                sample["x"] = sample["x"].cpu().numpy()
            if isinstance(sample.get("phi"), torch.Tensor):
                sample["phi"] = sample["phi"].cpu().numpy()

            # handle both ae and base dataset flux naming
            flux_val = sample.get("flux") if "flux" in sample else sample.get("gt_flux")
            if isinstance(flux_val, torch.Tensor):
                flux_val = flux_val.cpu().numpy()

            if key == "df":
                x = sample["x"]
                norm_axes = tuple(self.normalizers[key]["agg_axes"]) if self.normalizers[key]["agg_axes"] else (1, 2, 3, 4, 5)
                if self.separate_zf:
                    x = self._separate_zf(x)
            elif key == "phi":
                x = sample["phi"]
                if x.ndim == 3:
                    x = np.expand_dims(x, 0)
                norm_axes = norm_axes = tuple(self.normalizers[key]["agg_axes"]) if self.normalizers[key]["agg_axes"] else (1, 2, 3)
            else:
                x = np.array([flux_val], dtype=np.float32)
                norm_axes = (0,)

            x_mean = np.mean(x, norm_axes, keepdims=True)
            x_var = np.var(x, norm_axes, keepdims=True)
            x_min = np.min(x, norm_axes, keepdims=True)
            x_max = np.max(x, norm_axes, keepdims=True)

            return x_mean, x_var, x_min, x_max

        # subsample by two for normalization stats
        t_indices = (
            list(range(0, self.length, 2))
            if key in ["df", "phi"]
            else list(range(self.length))
        )
        file_hash = hashlib.sha256("".join(sorted(self.files)).encode()).hexdigest()[:8]
        tmu = "mu" if self.decouple_mu else ""
        segments = [prefix, key, f"offset{offset}", tmu, suffix, file_hash, "stats"]
        stats_filename = "_".join(filter(None, (str(s) for s in segments))) + ".pkl"
        stats_path = os.path.join(self.dir, stats_filename)

        if os.path.exists(stats_path):
            stats = pickle.load(open(stats_path, "rb"))
        else:
            process_inds = partial(process_t_idx, key=key)
            stats = None
            with ThreadPoolExecutor(self.num_workers) as executor:
                metrics_gen = tqdm.tqdm(
                    executor.map(process_inds, t_indices),
                    total=len(t_indices),
                    desc=f"re-computing normalization stats for {key}",
                )

                for metrics in metrics_gen:
                    x_mean, x_var, x_min, x_max = metrics
                    if stats is None:
                        stats = RunningMeanStd(shape=x_mean.shape)
                    stats.update(x_mean, x_var, x_min, x_max)

            pickle.dump(stats, open(stats_path, "wb"))
            print(f"saved recomputed stats to {stats_path}")

        return stats

    def __getitem__(
        self, index: Union[int, Tuple[int, int]], get_normalized: bool = True
    ) -> CycloneSample:
        if isinstance(index, int):
            file_index, t_index = self.flat_index_to_file_and_tstep[index]
        elif isinstance(index, Tuple):
            file_index, t_index = index

        with self.backend.open(self.files[file_index]) as f:
            sample = self._load_data(f, file_index, t_index)

        x, gt = sample["x"], sample["gt"]
        if self.separate_zf:
            x = self._separate_zf(x)
            gt = self._separate_zf(gt)

        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        timestep = sample["timestep"]
        itg, dg, s_hat, q = sample["itg"], sample["dg"], sample["s_hat"], sample["q"]
        geom = sample["geometry"]

        if get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)
                gt = (gt - shift) / scale
            if phi is not None:
                phi, shift, scale = self.normalize(file_index, phi=phi)
                y_phi = (y_phi - shift) / scale
            if flux is not None:
                flux, *_ = self.normalize(file_index, flux=flux)

        return CycloneSample(
            df=torch.as_tensor(x, self.dtype) if x is not None else None,
            y_df=torch.as_tensor(gt, self.dtype) if gt is not None else None,
            phi=torch.as_tensor(phi, self.dtype) if phi is not None else None,
            y_phi=(torch.as_tensor(y_phi, self.dtype) if y_phi is not None else None),
            y_flux=(torch.as_tensor(flux, self.dtype) if flux is not None else None),
            timestep=torch.as_tensor(timestep, self.dtype),
            file_index=torch.tensor(file_index, torch.long),
            timestep_index=torch.tensor(t_index, torch.long),
            geometry=tree_map(lambda g: torch.as_tensor(g, torch.float64), geom),
            itg=torch.as_tensor(itg, self.dtype),
            dg=torch.as_tensor(dg, self.dtype),
            s_hat=torch.as_tensor(s_hat, self.dtype),
            q=torch.as_tensor(q, self.dtype),
        )

    def _load_data(self, f: Any, file_index: int, t_index: int) -> dict:
        original_t_index = t_index + self.offsets[file_index]
        meta = self.metadata[file_index]
        x, gt, poten, y_poten, gt_flux = [], [], [], [], []

        for i in range(self.bundle_seq_length):
            t_str = str(original_t_index + i).zfill(5)
            t_str_gt = str(original_t_index + self.bundle_seq_length + i).zfill(5)

            if "df" in self.fields_to_load:
                k = self.backend.read_df(
                    f, t_str, self.df_shape, self.active_keys, self.rank
                )
                k_gt = self.backend.read_df(
                    f, t_str_gt, self.df_shape, self.active_keys, self.rank
                )
                x.append(k)
                gt.append(k_gt)

            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(f, t_str, self.phi_resolution, self.rank)
                phi_gt = self.backend.read_phi(
                    f, t_str_gt, self.phi_resolution, self.rank
                )
                poten.append(phi)
                y_poten.append(phi_gt)

            flux = meta["fluxes"][original_t_index + self.bundle_seq_length + i]
            gt_flux.append(flux)

        sample = {}
        # stack arrays/tensors
        if "df" in self.fields_to_load:
            if self.bundle_seq_length == 1:
                x, gt = x[0], gt[0]
            else:
                x = (
                    torch.stack(x, axis=1)
                    if isinstance(x[0], torch.Tensor)
                    else np.stack(x, axis=1)
                )
                gt = (
                    torch.stack(gt, axis=1)
                    if isinstance(gt[0], torch.Tensor)
                    else np.stack(gt, axis=1)
                )
        else:
            x, gt = None, None

        if "phi" in self.fields_to_load:
            if self.bundle_seq_length == 1:
                poten, y_poten = poten[0], y_poten[0]
            else:
                poten = (
                    torch.stack(poten, axis=1)
                    if isinstance(poten[0], torch.Tensor)
                    else np.stack(poten, axis=1)
                )
                y_poten = (
                    torch.stack(y_poten, axis=1)
                    if isinstance(y_poten[0], torch.Tensor)
                    else np.stack(y_poten, axis=1)
                )
        else:
            poten, y_poten = None, None

        sample["x"] = x
        sample["gt"] = gt
        sample["phi"] = poten
        sample["y_phi"] = y_poten
        sample["gt_flux"] = (
            torch.tensor(gt_flux).squeeze()
            if isinstance(gt_flux[0], torch.Tensor)
            else np.array(gt_flux).squeeze()
        )

        sample["timestep"] = meta["timesteps"][original_t_index]
        sample["itg"] = meta["ion_temp_grad"].squeeze()
        sample["dg"] = meta["density_grad"].squeeze()
        sample["s_hat"] = meta["s_hat"].squeeze()
        sample["q"] = meta["q"].squeeze()
        sample["geometry"] = meta["geometry"]
        return sample

    def normalize(
        self,
        file_index: int,
        df: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        flux: Optional[torch.Tensor] = None,
        return_stats: bool = True,
    ):
        if df is not None:
            field = "df"
            x = df
        elif phi is not None:
            field = "phi"
            x = phi
        elif flux is not None:
            field = "flux"
            x = flux
        else:
            raise ValueError

        scale, shift = self._get_scale_shift(file_index, field, x)
        if return_stats:
            return (x - shift) / scale, shift, scale
        else:
            return (x - shift) / scale

    def denormalize(
        self,
        file_index: int,
        df: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        flux: Optional[torch.Tensor] = None,
    ):
        if df is not None:
            field = "df"
            x = df
        elif phi is not None:
            field = "phi"
            x = phi
        elif flux is not None:
            field = "flux"
            x = flux
        else:
            raise ValueError

        scale, shift = self._get_scale_shift(file_index, field, x)
        return x * scale + shift

    def _conditioning_filter(self, fname: str, offset: int) -> bool:
        meta = self.backend.read_metadata(fname, self.fields_to_load)
        for cond_name, cond_range in self.cond_filters.items():
            if len(cond_name.split("_")) > 1:
                where, cond_name = cond_name.split("_")
            if cond_name in meta:
                cond = meta[cond_name]
                if not isinstance(cond_range[0], Sequence):
                    cond_range = [cond_range]
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

    def _apply_timestep_std_filter(self):
        k = self.timestep_std_filter
        ref_offset = self.timestep_std_offset
        valid_flat_indices = {}
        flat_idx_counter = 0

        for file_idx in range(len(self.files)):
            fluxes = self.metadata[file_idx]["fluxes"]
            ref_fluxes = fluxes[ref_offset:]
            ref_mean = np.mean(ref_fluxes)
            ref_std = np.std(ref_fluxes)

            lower = ref_mean - k * ref_std
            upper = ref_mean + k * ref_std

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

    def _get_scale_shift(self, file_index: int, field: str, x) -> Tuple:
        shift, scale = np.array(0.0), np.array(1.0)
        if self.normalization_scope == "sample":
            pass
        else:
            key = "full" if self.normalization_scope == "dataset" else file_index
            if self.normalizers[field]["type"] == "zscore":
                shift = expand_as(self.stats[field][key][f"mean"], x)
                scale = expand_as(self.stats[field][key][f"std"], x)
            if self.normalizers[field]["type"] == "minmax":
                x_min = expand_as(self.stats[field][key][f"min"], x)
                x_max = expand_as(self.stats[field][key][f"max"], x)
                scale = (x_max - x_min) / self.minmax_beta1
                shift = x_min + scale * self.minmax_beta2
        if isinstance(x, torch.Tensor):
            scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
            shift = torch.as_tensor(shift, dtype=x.dtype, device=x.device)
        return scale, shift

    def get_at_time(
        self,
        file_idx: torch.Tensor,
        timestep_idx: torch.Tensor,
        get_normalized: bool = True,
        num_workers: int = 1,
    ):
        def _fetch(idx):
            return self.__getitem__(idx, get_normalized)

        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                samples = list(
                    executor.map(
                        _fetch, list(zip(file_idx.tolist(), timestep_idx.tolist()))
                    )
                )
            sample = self.collate(samples)
        else:
            sample = self.collate(
                [
                    self.__getitem__((f_idx, t_idx), get_normalized)
                    for f_idx, t_idx in zip(file_idx.tolist(), timestep_idx.tolist())
                ]
            )
        return sample

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

    def get_fluxes(self, file_index: int):
        fluxes = self.metadata[file_index]["fluxes"]
        return torch.tensor(fluxes[1:])

    def get_avg_flux(self, file_index: Union[int, Sequence[int]]):
        if not isinstance(file_index, Sequence):
            file_index = [file_index]
        avg_fluxes = []
        for f in file_index:
            fluxes = self.get_fluxes(f)
            fluxes = fluxes[-80:]
            avg_fluxes.append(float(fluxes.mean()))
        if len(avg_fluxes) == 1:
            return avg_fluxes[0]
        else:
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
        return torch.stack(flux_seq, 0)

    def __len__(self):
        return self.length

    def num_ts(self, file_idx: int):
        return self.file_num_timesteps[file_idx]

    def collate(self, batch: Sequence[CycloneSample]):

        def stack_batch(_b: Sequence[CycloneSample], key: str):
            if hasattr(_b[0], key) is not None:
                return torch.stack([getattr(sample, key) for sample in _b])
            return None

        return CycloneSample(
            df=stack_batch(batch, "df"),
            y_df=stack_batch(batch, "y_df"),
            phi=stack_batch(batch, "phi"),
            y_phi=stack_batch(batch, "y_phi"),
            y_flux=stack_batch(batch, "y_flux"),
            timestep=stack_batch(batch, "timestep"),
            file_index=stack_batch(batch, "file_index"),
            timestep_index=stack_batch(batch, "timestep_index"),
            itg=stack_batch(batch, "itg"),
            dg=stack_batch(batch, "dg"),
            s_hat=stack_batch(batch, "s_hat"),
            q=stack_batch(batch, "q"),
            geometry=tree_map(
                lambda *x: torch.stack([torch.as_tensor(v) for v in x]),
                *[s.geometry for s in batch],
            ),
        )


# TODO(gg did not test)
class LinearCycloneDataset(CycloneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, linear_sims=True, **kwargs)
        temp_stats = self._recompute_stats_linear()

        self.stats["df"]["full"]["mean"] = temp_stats.mean.astype(np.float32)
        self.stats["df"]["full"]["std"] = (temp_stats.var ** (1 / 2)).astype(np.float32)
        self.stats["df"]["full"]["min"] = temp_stats.min.astype(np.float32)
        self.stats["df"]["full"]["max"] = temp_stats.max.astype(np.float32)

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneSample:
        file_index = index

        with self.backend.open(self.files[file_index]) as f:
            t_index = len(self.metadata[file_index]["timesteps"]) - 1
            sample = self._load_data(f, file_index, t_index)

        x = sample["x"]
        if self.separate_zf:
            x = self._separate_zf(x)
        gt = None

        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        timestep = sample["timestep"]
        itg, dg, s_hat, q = sample["itg"], sample["dg"], sample["s_hat"], sample["q"]
        geometry = sample["geometry"]
        fluxavg = sample["fluxavg"]
        if get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)
            if phi is not None:
                phi, shift, scale = self.normalize(file_index, phi=phi)
                y_phi = (y_phi - shift) / scale
            if flux is not None:
                flux, shift, scale = self.normalize(file_index, flux=flux)
            # note: fluxavg normalization falls back to pure assignment if unsupported

        return CycloneSample(
            df=torch.as_tensor(x, dtype=self.dtype) if x is not None else None,
            y_df=torch.as_tensor(gt, dtype=self.dtype) if gt is not None else None,
            phi=torch.as_tensor(phi, dtype=self.dtype) if phi is not None else None,
            y_phi=(
                torch.as_tensor(y_phi, dtype=self.dtype) if y_phi is not None else None
            ),
            y_flux=torch.as_tensor(flux, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            itg=torch.as_tensor(itg, dtype=self.dtype),
            dg=torch.as_tensor(dg, dtype=self.dtype),
            s_hat=torch.as_tensor(s_hat, dtype=self.dtype),
            q=torch.as_tensor(q, dtype=self.dtype),
            geometry=tree_map(
                lambda geom: torch.as_tensor(geom, dtype=self.dtype), geometry
            ),
            y_fluxavg=(
                torch.as_tensor(fluxavg, dtype=self.dtype)
                if fluxavg is not None
                else None
            ),
            position=None,
        )

    def _load_data(self, f, file_index, t_index) -> dict:
        original_t_index = t_index
        meta = self.metadata[file_index]

        x, poten, gt_flux = [], [], []
        for i in range(self.bundle_seq_length):
            t_str = str(original_t_index + i).zfill(5)

            if "df" in self.fields_to_load:
                k = self.backend.read_df(
                    f, t_str, self.df_shape, self.active_keys, self.rank
                )
                x.append(k)

            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(f, t_str, self.phi_resolution, self.rank)
                poten.append(phi)

            if "flux" in self.fields_to_load:
                flux = meta["fluxes"][original_t_index + i]
                gt_flux.append(flux)

        sample = {}
        if "df" in self.fields_to_load:
            if self.bundle_seq_length == 1:
                x = x[0]
            else:
                x = (
                    torch.stack(x, axis=1)
                    if isinstance(x[0], torch.Tensor)
                    else np.stack(x, axis=1)
                )
        else:
            x = None

        if "phi" in self.fields_to_load:
            if self.bundle_seq_length == 1:
                poten = poten[0]
            else:
                poten = (
                    torch.stack(poten, axis=1)
                    if isinstance(poten[0], torch.Tensor)
                    else np.stack(poten, axis=1)
                )
        else:
            poten = None

        if "flux" in self.fields_to_load:
            if self.bundle_seq_length == 1:
                gt_flux = np.array(gt_flux).squeeze()
            else:
                gt_flux = np.stack(gt_flux, axis=1)
        else:
            gt_flux = None

        sample["x"] = x
        sample["gt"] = None
        sample["phi"] = poten
        sample["y_phi"] = None
        sample["gt_flux"] = (
            torch.tensor(gt_flux).squeeze() if gt_flux is not None else None
        )

        sample["timestep"] = meta["timesteps"][original_t_index]
        sample["itg"] = meta["ion_temp_grad"].squeeze()
        sample["dg"] = meta["density_grad"].squeeze()
        sample["s_hat"] = meta["s_hat"].squeeze()
        sample["q"] = meta["q"].squeeze()
        sample["geometry"] = meta["geometry"]
        sample["fluxavg"] = (
            self.get_avg_flux(file_index).squeeze()
            if "fluxavg" in self.fields_to_load
            else None
        )
        sample["position"] = None
        return sample

    def get_timesteps(
        self,
        file_idx: torch.Tensor,
        timestep_idx: Optional[torch.Tensor] = None,
        offset: int = 0,
    ):
        if isinstance(file_idx, int):
            file_idx = torch.tensor([file_idx])
        file_idx = file_idx.cpu().long()

        timesteps_tensor = torch.zeros_like(file_idx)
        for i, file_index in enumerate(file_idx):
            # directly use cached metadata to avoid opening files
            t_index = len(self.metadata[file_index.item()]["timesteps"]) - 1
            timesteps_tensor[i] = t_index

        return timesteps_tensor

    def _recompute_stats_linear(self):
        t_indices = list(range(0, len(self.files)))
        stats_path = os.path.join(
            self.dir, f"df_linear_{len(self.files)}sims_stats.pkl"
        )

        if os.path.exists(stats_path):
            stats = pickle.load(open(stats_path, "rb"))
        else:
            stats = None
            for index in tqdm.tqdm(
                t_indices, desc="re-computing normalization stats for df"
            ):
                file_index = index

                with self.backend.open(self.files[file_index]) as f:
                    t_index = len(self.metadata[file_index]["timesteps"]) - 1
                    sample = self._load_data(f, file_index, t_index)

                x = sample["x"]
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()

                norm_axes = (1, 2, 3, 4, 5)
                if self.separate_zf:
                    x = self._separate_zf(x)

                x_mean = np.mean(x, norm_axes, keepdims=True)
                x_var = np.var(x, norm_axes, keepdims=True)
                x_min = np.min(x, norm_axes, keepdims=True)
                x_max = np.max(x, norm_axes, keepdims=True)

                if stats is None:
                    stats = RunningMeanStd(shape=x_mean.shape)
                stats.update(x_mean, x_var, x_min, x_max)

            pickle.dump(stats, open(stats_path, "wb"))

        return stats

    def get_at_time(
        self,
        file_idx: torch.Tensor,
        timestep_idx: torch.Tensor,
        get_normalized: bool = True,
    ):
        sample = self.collate(
            [self.__getitem__(idx, get_normalized) for idx in file_idx]
        )
        return sample

    def __len__(self):
        return len(self.files)


# TODO(gg did not test)
class CoordinateCycloneDataset(CycloneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            len(self.fields_to_load) == 2 and "position" in self.fields_to_load
        ), "cannot load multiple fields for coordinates"
        assert "flux" not in self.fields_to_load, "loading flux coordinates not supported"

    def _load_data(self, f, file_index, t_index, n_subsamples=65536) -> dict:
        original_t_index = t_index + self.offsets[file_index]
        meta = self.metadata[file_index]
        assert (
            self.bundle_seq_length == 1
        ), "bundling of >1 steps not supported for coordinates"

        t_str = str(original_t_index).zfill(5)
        t_str_gt = str(original_t_index + 1).zfill(5)

        if "df" in self.fields_to_load:
            k = self.backend.read_df(
                f, t_str, self.df_shape, self.active_keys, self.rank
            )
            k_gt = self.backend.read_df(
                f, t_str_gt, self.df_shape, self.active_keys, self.rank
            )

            # cast to numpy for coordinate extraction
            if isinstance(k, torch.Tensor):
                k = k.cpu().numpy()
            if isinstance(k_gt, torch.Tensor):
                k_gt = k_gt.cpu().numpy()

            position = np.indices(k.shape[1:]).reshape(5, -1).T
            x = np.concatenate([k[0].ravel()[None], k[1].ravel()[None]])
            y = np.concatenate([k_gt[0].ravel()[None], k_gt[1].ravel()[None]])
        else:
            x = None

        if "phi" in self.fields_to_load:
            phi = self.backend.read_phi(f, t_str_gt, self.phi_resolution, self.rank)
            if isinstance(phi, torch.Tensor):
                phi = phi.cpu().numpy()

            poten = phi.ravel()
            position = np.indices(phi.shape).reshape(5, -1).T
        else:
            phi = None

        # subsample positions
        rand_inds = np.random.choice(len(position), size=n_subsamples)
        position = position[rand_inds]

        sample = {}
        sample["x"] = x[:, rand_inds] if x is not None else None
        sample["gt"] = y[:, rand_inds] if y is not None else None
        sample["phi"] = poten[:, rand_inds] if phi is not None else None
        sample["y_phi"] = None
        sample["gt_flux"] = None
        sample["position"] = position

        sample["timestep"] = meta["timesteps"][original_t_index]
        sample["itg"] = meta["ion_temp_grad"].squeeze()
        sample["dg"] = meta["density_grad"].squeeze()
        sample["s_hat"] = meta["s_hat"].squeeze()
        sample["q"] = meta["q"].squeeze()
        sample["geometry"] = meta["geometry"]
        sample["fluxavg"] = (
            self.get_avg_flux(file_index).squeeze()
            if "fluxavg" in self.fields_to_load
            else None
        )

        return sample

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneSample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        with self.backend.open(self.files[file_index]) as f:
            sample = self._load_data(f, file_index, t_index)

        x, gt = sample["x"], sample["gt"]
        if self.separate_zf:
            x = self._separate_zf(x)
            gt = self._separate_zf(gt)

        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        timestep = sample["timestep"]
        itg, dg, s_hat, q = sample["itg"], sample["dg"], sample["s_hat"], sample["q"]
        geometry = sample["geometry"]
        fluxavg = sample["fluxavg"]
        position = sample["position"]

        if get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)
                gt = (gt - shift) / scale
            if phi is not None:
                phi, shift, scale = self.normalize(file_index, phi=phi)
                y_phi = (y_phi - shift) / scale
            if flux is not None:
                flux, *_ = self.normalize(file_index, flux=flux)

        return CycloneSample(
            df=torch.as_tensor(x, dtype=self.dtype) if x is not None else None,
            y_df=torch.as_tensor(gt, dtype=self.dtype) if gt is not None else None,
            phi=torch.as_tensor(phi, dtype=self.dtype) if phi is not None else None,
            y_phi=(
                torch.as_tensor(y_phi, dtype=self.dtype) if y_phi is not None else None
            ),
            y_flux=(
                torch.as_tensor(flux, dtype=self.dtype) if flux is not None else None
            ),
            y_fluxavg=(
                torch.as_tensor(fluxavg, dtype=self.dtype)
                if fluxavg is not None
                else None
            ),
            position=torch.as_tensor(position, dtype=self.dtype),
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            geometry=tree_map(
                lambda geom: torch.as_tensor(geom, dtype=self.dtype), geometry
            ),
            itg=torch.as_tensor(itg, dtype=self.dtype),
            dg=torch.as_tensor(dg, dtype=self.dtype),
            s_hat=torch.as_tensor(s_hat, dtype=self.dtype),
            q=torch.as_tensor(q, dtype=self.dtype),
        )
