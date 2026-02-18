from typing import Type, Optional, List, Tuple, Dict, Sequence, Union
import re
import os
import h5py
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils._pytree import tree_map
import random
import pickle
from functools import partial
import hashlib

from neugk.utils import RunningMeanStd


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
        tail_offset: int = 0,
        separate_zf: bool = False,
        decouple_mu: bool = False,
        num_workers: int = 4,
        real_potens: bool = False,
        timestep_std_filter: float = None,
        timestep_std_offset: int = 80,
    ):
        self.input_fields = input_fields
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

                self.files = [os.path.join(self.dir, f_name) for f_name in trajectories]

        # take all files in path
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

        # if set, load preprocessed files with ifft
        self.spatial_ifft = spatial_ifft
        if spatial_ifft:
            if split_into_bands:
                n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
                self.files = [
                    (
                        f
                        if f"_ifft_separate_zf{n_bands_tag}.h5" in f
                        else re.sub(r"\.h5$", f"_ifft_separate_zf{n_bands_tag}.h5", f)
                    )
                    for f in self.files
                ]
            else:
                real_potens_tag = "_realpotens" if real_potens else ""
                self.files = [
                    (
                        f
                        if f"_ifft{real_potens_tag}.h5" in f
                        else re.sub(r"\.h5$", f"_ifft{real_potens_tag}.h5", f)
                    )
                    for f in self.files
                ]

        # remove duplicates
        self.files = list(set(self.files))
        for f in self.files:
            assert os.path.isfile(f), f"Trajectory  '{f}' does not exist!"

        # we use offsets for flux filtering
        if self.cond_filters:
            threshold = offset if offset > 0 else 80
            self.files = [
                f for f in self.files if self._conditioning_filter(f, threshold)
            ]

        if len(self.files) == 0:
            raise RuntimeError(f"No trajectories found! Active filters: {cond_filters}")

        # get metadata (samples per file and normalization stats)
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
                n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
                filename = filename.replace(f"_ifft_separate_zf{n_bands_tag}", "")
                real_potens_tag = "_realpotens" if real_potens else ""
                filename = filename.replace(f"_ifft{real_potens_tag}", "")
            # number of offset samples on the tail (as holdout or for n_eval_steps)
            self.n_tail_holdout = tail_offset
            if self.partial_holdouts.get(filename, 0) > self.n_tail_holdout:
                self.n_tail_holdout += self.partial_holdouts.get(filename, 0)
            with h5py.File(f_path, "r") as f:
                # read the timesteps
                if self.n_tail_holdout:
                    if split == "train":
                        timesteps = f["metadata/timesteps"][
                            offset : -self.n_tail_holdout
                        ]
                        orig_t_index = np.arange(len(timesteps))[offset::subsample]
                    else:
                        timesteps = f["metadata/timesteps"][
                            offset : -self.n_tail_holdout :
                        ]
                        orig_t_index = np.arange(len(timesteps))[::subsample]
                        self.steps_per_file[f_id] = len(
                            f["metadata/timesteps"][:][offset::subsample]
                        )
                else:
                    timesteps = f["metadata/timesteps"][:][offset:]
                    orig_t_index = np.arange(len(timesteps))[::subsample]
                # crop timesteps
                timesteps = timesteps[orig_t_index]
                # This only works for 1 step training (with pf and rollout aswell)
                n_samples = len(timesteps) - self.bundle_seq_length * 2 + 1
                self.file_num_samples.append(n_samples)
                self.file_num_timesteps.append(len(timesteps))
                per_file_t_indexes.append(orig_t_index)
                # TODO: norm_stats is never None here!
                if normalization_stats is None and normalization_scope == "dataset":
                    assert split == "train", "Validation must have normalization_stats."
                    # normalization stats
                    for k in self.input_fields:
                        try:
                            self.stats[k][f_id]["mean"] = f[f"metadata/{k}_mean"][:]
                            self.stats[k][f_id]["std"] = f[f"metadata/{k}_std"][:]
                            self.stats[k][f_id]["min"] = f[f"metadata/{k}_min"][:]
                            self.stats[k][f_id]["max"] = f[f"metadata/{k}_max"][:]
                        except:
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
        # TODO is this part still in use?
        if split == "val" and self.partial_holdouts:
            for file_idx, file in enumerate(self.files):
                filename = os.path.split(file)[-1]
                # optional preprocessing filename tags
                n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
                filename = filename.replace(f"_ifft_separate_zf{n_bands_tag}", "")
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
                    # TODO: assumes offsets are equal across all sims
                    stats[key] = self._recompute_stats(key=key, offset=self.offsets[0])

        if normalization_scope == "dataset" and normalization_stats is None:
            for k in input_fields:
                self.stats[k]["full"]["mean"] = stats[k].mean.astype(np.float32)
                self.stats[k]["full"]["std"] = (stats[k].var ** 0.5).astype(np.float32)
                self.stats[k]["full"]["min"] = stats[k].min.astype(np.float32)
                self.stats[k]["full"]["max"] = stats[k].max.astype(np.float32)

        # TODO assume same resolution across all files
        with h5py.File(self.files[0], "r") as f:
            self.resolution = f["metadata/resolution"][:]
            self.phi_resolution = (
                self.resolution[3],
                self.resolution[2],
                self.resolution[4],
            )

    def _separate_zf(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        nky = x.shape[-1]
        zf = np.repeat(x.mean(axis=-1, keepdims=True), repeats=nky, axis=-1)
        x = np.concatenate([zf, x - zf], axis=0)
        return x

    def _recompute_stats(self, key: str, offset: int = 0):

        def process_t_idx(t_idx, key):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]
            with h5py.File(self.files[file_index], "r") as f:
                sample = self._load_data(f, file_index, t_index)

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
        if isinstance(index, int):
            # lookup file index and time index from flat index
            file_index, t_index = self.flat_index_to_file_and_tstep[index]
        elif isinstance(index, Tuple):
            # direct access
            file_index, t_index = index
            assert file_index < len(self.files)
            assert t_index < self.num_ts(file_index) + self.n_tail_holdout

        with h5py.File(self.files[file_index], "r") as f:
            sample = self._load_data(f, file_index, t_index)
        # k-fields
        x, gt = sample["x"], sample["gt"]
        if self.separate_zf:
            x = self._separate_zf(x)
            gt = self._separate_zf(gt)
        # accessory fields
        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        # conditioning fields
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

        # x = self._shape_correction(x)
        # gt = self._shape_correction(gt)

        return CycloneSample(
            df=torch.tensor(x, dtype=self.dtype) if x is not None else None,
            y_df=torch.tensor(gt, dtype=self.dtype) if gt is not None else None,
            phi=(torch.tensor(phi, dtype=self.dtype) if phi is not None else None),
            y_phi=(
                torch.tensor(y_phi, dtype=self.dtype) if y_phi is not None else None
            ),
            y_flux=torch.tensor(flux, dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            geometry=tree_map(lambda x: torch.from_numpy(x), geometry),
            # conditioning
            itg=torch.tensor(itg, dtype=self.dtype),
            dg=torch.tensor(dg, dtype=self.dtype),
            s_hat=torch.tensor(s_hat, dtype=self.dtype),
            q=torch.tensor(q, dtype=self.dtype),
        )

    def _load_data(
        self, data, file_index, t_index
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        original_t_index = t_index + self.offsets[file_index]
        x = []
        gt = []
        poten = []
        y_poten = []
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
                if all(self.active_keys == np.array([0, 1])):
                    x.append(k)
                    gt.append(k_gt)
                else:
                    x.append(k[self.active_keys])
                    gt.append(k_gt[self.active_keys])
            if "phi" in self.input_fields:
                phi = data[f"data/{phi_name}"][:]
                phi_gt = data[f"data/{phi_name_gt}"][:]
                poten.append(phi)
                y_poten.append(phi_gt)

            # target flux
            flux = data["metadata/fluxes"][
                original_t_index + self.bundle_seq_length + i
            ]

            gt_flux.append(flux)

        sample = {}
        if "df" in self.input_fields:
            # stack to shape (c, t, v1, v2, s, x, y)
            if self.bundle_seq_length == 1:
                # sqeeze out time if we only have 1 timestep
                x, gt = x[0], gt[0]
            else:
                x = np.stack(x, axis=1)
                gt = np.stack(gt, axis=1)
        else:
            x, gt = None, None

        if "phi" in self.input_fields:
            # stack to shape (c, t, x, s, y)
            if self.bundle_seq_length == 1:
                poten = poten[0]
                y_poten = y_poten[0]
            else:
                poten = np.stack(poten, axis=1)
                y_poten = np.stack(y_poten, axis=1)

        else:
            poten, y_poten = None, None

        sample["x"] = x
        sample["gt"] = gt
        sample["phi"] = poten
        sample["y_phi"] = y_poten
        sample["gt_flux"] = np.array(gt_flux).squeeze()
        # load conditioning
        sample["timestep"] = data["metadata/timesteps"][original_t_index]
        sample["itg"] = data["metadata/ion_temp_grad"][:].squeeze()
        sample["dg"] = data["metadata/density_grad"][:].squeeze()
        sample["s_hat"] = data["metadata/s_hat"][:].squeeze()
        sample["q"] = data["metadata/q"][:].squeeze()
        # load geometry
        sample["geometry"] = {k: np.array(v[()]) for k, v in data["geometry"].items()}
        return sample

    def normalize(
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
        return (x - shift) / scale, shift, scale

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
        with h5py.File(fname, "r") as f:
            for cond_name, cond_range in self.cond_filters.items():
                if len(cond_name.split("_")) > 1:
                    where, cond_name = cond_name.split("_")
                if f"metadata/{cond_name}" in f:
                    cond = f[f"metadata/{cond_name}"][:]
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

    def _apply_timestep_std_filter(self):
        """Filter timesteps based on flux being within k*std of reference mean
        timesteps where flux is outside [mean - k*std, mean + k*std] are removed
        """
        k = self.timestep_std_filter
        ref_offset = self.timestep_std_offset
        valid_flat_indices = {}
        flat_idx_counter = 0

        for file_idx, f_path in enumerate(self.files):
            with h5py.File(f_path, "r") as f:
                fluxes = f["metadata/fluxes"][:]
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

            for old_flat_idx, file_idx_stored, t_idx in old_indices:
                # t_idx is relative to the offset, so we need to add offset to get the original index
                # target flux is at original_t_index + bundle_seq_length
                original_t_index = t_idx + self.offsets[file_idx]
                flux_idx = original_t_index + self.bundle_seq_length

                # bounds check
                if flux_idx >= len(fluxes):
                    flux_idx = len(fluxes) - 1

                flux_at_t = fluxes[flux_idx]

                if lower <= flux_at_t <= upper:
                    valid_flat_indices[flat_idx_counter] = (file_idx, t_idx)
                    flat_idx_counter += 1

        # rebuild the mappings
        self.flat_index_to_file_and_tstep = valid_flat_indices
        self.file_and_tstep_to_flat_index = {
            (file_idx, t_idx): flat_idx
            for flat_idx, (file_idx, t_idx) in valid_flat_indices.items()
        }

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
                shift = self.stats[field][key]["mean"]
                scale = self.stats[field][key]["std"]
            if self.normalization == "minmax":
                x_min = self.stats[field][key]["min"]
                x_max = self.stats[field][key]["max"]
                scale = (x_max - x_min) / self.minmax_beta1
                shift = x_min + scale * self.minmax_beta2
            # scale, shift = expand_as(scale, x), expand_as(shift, x)
        if isinstance(x, torch.Tensor):
            scale = torch.from_numpy(scale).to(x.device)
            shift = torch.from_numpy(shift).to(x.device)
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
                # direct indexing to access hidden tail for evaluation
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
        if isinstance(timestep_idx, int):
            timestep_idx = torch.tensor([timestep_idx])
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
                timesteps_array = f["metadata/timesteps"][offset:]
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

    def get_avg_flux(self, file_index: Union[int, Sequence[int]]):
        if not isinstance(file_index, Sequence):
            file_index = [file_index]
        avg_fluxes = []
        for f in file_index:
            fluxes = self.get_fluxes(f)
            fluxes = fluxes[-80:]  # TODO naive crop linear phase
            avg_fluxes.append(float(fluxes.mean()))
        if len(avg_fluxes) == 1:
            return avg_fluxes[0]  # return float
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
            file_index=torch.stack([sample.file_index for sample in batch]),
            timestep_index=torch.stack([sample.timestep_index for sample in batch]),
            geometry=tree_map(lambda *x: torch.stack(x), *[s.geometry for s in batch]),
            itg=torch.stack([sample.itg for sample in batch]),
            dg=torch.stack([sample.dg for sample in batch]),
            s_hat=torch.stack([sample.s_hat for sample in batch]),
            q=torch.stack([sample.q for sample in batch]),
        )


class LinearCycloneDataset(CycloneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, linear_sims=True, **kwargs)
        temp_stats = self._recompute_stats_linear()
        import pdb

        pdb.set_trace()
        self.stats["df"]["full"]["mean"] = temp_stats.mean.astype(np.float32)
        self.stats["df"]["full"]["std"] = (temp_stats.var ** (1 / 2)).astype(np.float32)
        self.stats["df"]["full"]["min"] = temp_stats.min.astype(np.float32)
        self.stats["df"]["full"]["max"] = temp_stats.max.astype(np.float32)

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
        file_index = index

        with h5py.File(self.files[file_index], "r") as f:
            df_files = [key for key in f["data"].keys() if key.startswith("timestep")]
            t_index = len(df_files) - 1
            sample = self._load_data(f, file_index, t_index)
        # k-fields
        x = sample["x"]
        if self.separate_zf:
            x = self._separate_zf(x)
        gt = None
        # potential field and flux
        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        # conditioning fields
        timestep = sample["timestep"]
        itg, dg, s_hat, q = sample["itg"], sample["dg"], sample["s_hat"], sample["q"]
        # geometric tensor
        geometry = sample["geometry"]
        # accessory
        fluxavg = sample["fluxavg"]
        if self.normalization is not None and get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)

            if phi is not None:
                phi, shift, scale = self.normalize(file_index, phi=phi)
                y_phi = (y_phi - shift) / scale

            if flux is not None:
                flux, shift, scale = self.normalize(file_index, flux=flux)

            if fluxavg is not None:
                fluxavg, *_ = self.normalize(file_index, fluxavg=fluxavg)

        return CycloneSample(
            df=torch.tensor(x, dtype=self.dtype) if x is not None else None,
            y_df=torch.tensor(gt, dtype=self.dtype) if gt is not None else None,
            phi=(torch.tensor(phi, dtype=self.dtype) if phi is not None else None),
            y_phi=(
                torch.tensor(y_phi, dtype=self.dtype) if y_phi is not None else None
            ),
            y_flux=torch.tensor(flux, dtype=self.dtype),
            # index info
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            # conditioning
            timestep=torch.tensor(timestep, dtype=self.dtype),
            itg=torch.tensor(itg, dtype=self.dtype),
            dg=torch.tensor(dg, dtype=self.dtype),
            s_hat=torch.tensor(s_hat, dtype=self.dtype),
            q=torch.tensor(q, dtype=self.dtype),
            # geometric tensors
            geometry=tree_map(lambda x: torch.from_numpy(x), geometry),
            # accessory
            y_fluxavg=(
                torch.tensor(fluxavg, dtype=self.dtype) if fluxavg is not None else None
            ),
            position=None,
        )

    def _load_data(
        self, data, file_index, t_index
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        original_t_index = t_index

        x = []
        poten = []
        gt_flux = []
        for i in range(self.bundle_seq_length):
            # read the input
            k_name = "timestep_" + str(original_t_index + i).zfill(5)
            phi_name = "poten_" + str(original_t_index + i).zfill(5)

            if "df" in self.input_fields:
                k = data[f"data/{k_name}"][:]
                # select only active re/im parts
                if all(self.active_keys == np.array([0, 1])):
                    x.append(k)
                else:
                    x.append(k[self.active_keys])

            if "phi" in self.input_fields:
                phi = data[f"data/{phi_name}"][:]
                poten.append(phi)

            if "flux" in self.input_fields:
                # target flux
                flux = data["metadata/fluxes"][:].squeeze().item()
                gt_flux.append(flux)

        sample = {}
        if "df" in self.input_fields:
            # stack to shape (c, t, v1, v2, s, x, y)
            if self.bundle_seq_length == 1:
                # sqeeze out time if we only have 1 timestep
                x = x[0]
            else:
                x = np.stack(x, axis=1)
        else:
            x = None

        if "phi" in self.input_fields:
            # stack to shape (c, t, x, s, y)
            if self.bundle_seq_length == 1:
                poten = poten[0]
            else:
                poten = np.stack(poten, axis=1)
        else:
            poten = None

        if "flux" in self.input_fields:
            # stack to shape (c, t, x, s, y)
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
        sample["gt_flux"] = np.array(gt_flux).squeeze()
        # load conditioning
        sample["timestep"] = data["metadata/timesteps"][original_t_index]
        sample["itg"] = data["metadata/ion_temp_grad"][:].squeeze()
        sample["dg"] = data["metadata/density_grad"][:].squeeze()
        sample["s_hat"] = data["metadata/s_hat"][:].squeeze()
        sample["q"] = data["metadata/q"][:].squeeze()
        # load geometry
        sample["geometry"] = {k: np.array(v[()]) for k, v in data["geometry"].items()}
        sample["fluxavg"] = (
            self.get_avg_flux(file_index).squeeze()
            if "fluxavg" in self.input_fields
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
        # file_idx: (B,)
        if isinstance(file_idx, int):
            file_idx = torch.tensor([file_idx])
        file_idx = file_idx.cpu().long()
        # all timesteps in file

        timesteps_tensor = torch.zeros_like(file_idx)
        for i, file_index in enumerate(file_idx):
            with h5py.File(self.files[file_index], "r") as f:
                df_files = [
                    key for key in f["data"].keys() if key.startswith("timestep")
                ]
                t_index = len(df_files) - 1
                timesteps_tensor[i] = t_index

        return timesteps_tensor

    def _recompute_stats_linear(self):
        t_indices = list(range(0, len(self.files)))

        if os.path.exists(
            os.path.join(self.dir, f"df_linear_{len(self.files)}sims_stats.pkl")
        ):
            stats = pickle.load(
                open(
                    os.path.join(
                        self.dir, f"df_linear_{len(self.files)}sims_stats.pkl"
                    ),
                    "rb",
                )
            )
        else:
            stats = None
            for index in tqdm.tqdm(
                t_indices, desc="Re-computing normalization stats for df"
            ):
                file_index = index

                with h5py.File(self.files[file_index], "r") as f:
                    df_files = [
                        key for key in f["data"].keys() if key.startswith("timestep")
                    ]
                    t_index = len(df_files) - 1
                    sample = self._load_data(f, file_index, t_index)

                x = sample["x"]
                norm_axes = (1, 2, 3, 4, 5)
                if self.separate_zf:
                    x = self._separate_zf(x)

                # Compute metrics for x and y
                x_mean = np.mean(x, norm_axes, keepdims=True)
                x_var = np.var(x, norm_axes, keepdims=True)
                x_min = np.min(x, norm_axes, keepdims=True)
                x_max = np.max(x, norm_axes, keepdims=True)

                if stats is None:
                    stats = RunningMeanStd(shape=x_mean.shape)
                stats.update(x_mean, x_var, x_min, x_max)

            pickle.dump(
                stats,
                open(
                    os.path.join(
                        self.dir, f"df_linear_{len(self.files)}sims_stats.pkl"
                    ),
                    "wb",
                ),
            )

        return stats

    def get_at_time(
        self,
        file_idx: torch.Tensor,
        timestep_idx: torch.Tensor,
        get_normalized: bool = True,
    ):
        # Compute the flat indices from the file indices and time indices
        sample = self.collate(
            [self.__getitem__(idx, get_normalized) for idx in file_idx]
        )
        return sample

    def __len__(self):
        # length for samples is number of files
        return len(self.files)


class CoordinateCycloneDataset(CycloneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            len(self.input_fields) == 2 and "position" in self.input_fields
        ), "Cannot load multiple fields for coordinates"
        assert "flux" not in self.input_fields, "Loading flux coordinates not supported"

    def _load_data(
        self,
        data,
        file_index,
        t_index,
        n_subsamples=65536,
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        original_t_index = t_index + self.offsets[file_index]
        assert (
            self.bundle_seq_length == 1
        ), "Bundling of >1 steps not supported for coordinates"
        # read the input
        k_name = "timestep_" + str(original_t_index).zfill(5)
        k_name_gt = "timestep_" + str(original_t_index + 1).zfill(5)
        phi_name = "poten_" + str(original_t_index + 1).zfill(5)

        if "df" in self.input_fields:
            k = data[f"data/{k_name}"][:]
            k_gt = data[f"data/{k_name_gt}"][:]
            # select only active re/im parts
            position = np.indices(k.shape[1:]).reshape(5, -1).T
            x = np.concatenate([k[0].ravel()[None], k[1].ravel()[None]])
            y = np.concatenate([k_gt[0].ravel()[None], k_gt[1].ravel()[None]])
        else:
            x = None

        if "phi" in self.input_fields:
            phi = data[f"data/{phi_name}"][:]
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
        sample["gt_flux"] = None  # TODO: support flux coordinates
        sample["position"] = position
        # load conditioning
        sample["timestep"] = data["metadata/timesteps"][original_t_index]
        sample["itg"] = data["metadata/ion_temp_grad"][:].squeeze()
        sample["dg"] = data["metadata/density_grad"][:].squeeze()
        sample["s_hat"] = data["metadata/s_hat"][:].squeeze()
        sample["q"] = data["metadata/q"][:].squeeze()
        # load geometry
        sample["geometry"] = {k: np.array(v[()]) for k, v in data["geometry"].items()}
        sample["fluxavg"] = (
            self.get_avg_flux(file_index).squeeze()
            if "fluxavg" in self.input_fields
            else None
        )
        return sample

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
        phi, y_phi, flux = sample["phi"], sample["y_phi"], sample["gt_flux"]
        # conditioning fields
        timestep = sample["timestep"]
        itg, dg, s_hat, q = sample["itg"], sample["dg"], sample["s_hat"], sample["q"]
        geometry = sample["geometry"]
        fluxavg = sample["fluxavg"]
        position = sample["position"]

        if self.normalization is not None and get_normalized:
            if x is not None:
                x, shift, scale = self.normalize(file_index, df=x)
                gt = (gt - shift) / scale

            if phi is not None:
                phi, shift, scale = self.normalize(file_index, phi=phi)
                y_phi = (y_phi - shift) / scale

            if flux is not None:
                flux, *_ = self.normalize(file_index, flux=flux)

            if fluxavg is not None:
                fluxavg, *_ = self.normalize(file_index, fluxavg=fluxavg)

        return CycloneSample(
            df=torch.tensor(x, dtype=self.dtype) if x is not None else None,
            y_df=torch.tensor(gt, dtype=self.dtype) if gt is not None else None,
            phi=(torch.tensor(phi, dtype=self.dtype) if phi is not None else None),
            y_phi=(
                torch.tensor(y_phi, dtype=self.dtype) if y_phi is not None else None
            ),
            y_flux=torch.tensor(flux, dtype=self.dtype) if flux is not None else None,
            y_fluxavg=(
                torch.tensor(fluxavg, dtype=self.dtype) if fluxavg is not None else None
            ),
            position=torch.tensor(position, dtype=self.dtype),
            timestep=torch.tensor(timestep, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            geometry=tree_map(lambda x: torch.from_numpy(x), geometry),
            # conditioning
            itg=torch.tensor(itg, dtype=self.dtype),
            dg=torch.tensor(dg, dtype=self.dtype),
            s_hat=torch.tensor(s_hat, dtype=self.dtype),
            q=torch.tensor(q, dtype=self.dtype),
        )
