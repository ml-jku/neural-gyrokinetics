from typing import Iterator, List, Type, Dict, Optional, Callable, Annotated, Tuple
import os
import random
import re

import h5py
import einops
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor


class MHDDataset(Dataset):
    def __init__(
        self,
        path: str,
        active_keys: List[str],
        split: str = "train",
        input_seq_length: int = 1,
        target_seq_length: int = 1,
        dtype: Type = None,
        random_seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        num_workers: int = 1,
        use_tqdm: bool = False,
    ):
        super().__init__()

        assert split in ["train", "val", "test"]
        self.dtype = torch.float32 if dtype is None else dtype

        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length
        self.extra_steps = self.input_seq_length + self.target_seq_length - 1

        self.active_keys = active_keys
        self.num_workers = num_workers
        self.bounds = None
        self.traj_length = None
        self.use_tqdm = use_tqdm

        # Walk through files and construct the file list
        self.files = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".h5"):
                    full_path = os.path.join(root, file)
                    self.files.append(full_path)

        # Shuffle and split files into training and validation sets
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

        self.data_chunks = [
            {
                "traj_tag": re.sub(r"\D", "", file.split("/")[-1].replace("h5", "")),
                "file_path": file,
            }
            for file in self.files
        ]

        self.traj_length, self.available_keys = self._get_traj_data()
        assert all(
            [k in self.available_keys for k in self.active_keys]
        ), "Variable key mismatch."
        self.n_samples_per_file = self.traj_length - self.extra_steps
        assert self.n_samples_per_file >= 1, (
            "Extra sample window is larger than the trajectory. "
            f"Extra input: {self.input_seq_length}, "
            f"extra target: {self.target_seq_length}, "
            f"trajectory length: {self.traj_length}."
        )

        self.current_chunk = self._load_h5_chunk(
            self.data_chunks[0]["traj_tag"], self.data_chunks[0]["file_path"]
        )

    def _get_window(self, sample, t):
        x, y = {}, {}
        for k in self.active_keys:
            x[k] = torch.tensor(
                np.array(
                    [sample[k][t + dt, ...] for dt in range(self.input_seq_length)]
                )
            )
            y[k] = torch.tensor(
                np.array(
                    [
                        sample[k][t + self.input_seq_length + dt, ...]
                        for dt in range(self.target_seq_length)
                    ]
                )
            )
            x[k] = x[k].type(self.dtype)
            y[k] = y[k].type(self.dtype)

        return {"x": x, "y": y}

    def _get_grid(self, chunk):
        grid_x = torch.from_numpy(np.array(chunk["R_mesh(nR,nZ)"])).type(self.dtype)
        grid_y = torch.from_numpy(np.array(chunk["Z_mesh(nR,nZ)"])).type(self.dtype)
        return torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)])

    def __getitem__(self, index):
        # Calculate file index and remainder for time index in file
        file_index = index // self.n_samples_per_file
        t_index = index % self.n_samples_per_file

        # Lazy load the sample
        chunk = self.data_chunks[file_index]

        if self.current_chunk["tag"] != chunk["traj_tag"]:
            self.current_chunk = self._load_h5_chunk(
                chunk["traj_tag"], chunk["file_path"]
            )

        file = self.current_chunk["file"]

        grid = self._get_grid(file)
        sample = self._get_window(file, t_index)
        sample["traj_tag"] = chunk["traj_tag"]
        sample["ts"] = t_index
        sample["grid"] = grid

        if self.bounds is not None:
            sample = self._normalize_sample(sample)

        return sample

    def __len__(self):
        return len(self.data_chunks) * self.n_samples_per_file

    def get_traj(self, tag):
        assert tag in self.trajectory_tags(), f"Trajectory with tag {tag} is missing."

        # Find file path corresponding to the given tag
        for chunk in self.data_chunks:
            if chunk["traj_tag"] == tag:
                file_path = chunk["file_path"]

        traj = {}
        grid = None

        with h5py.File(file_path, "r") as file:
            # Lazy load all relevant timesteps in the file
            for k in self.active_keys:
                var = np.array(file[k], dtype=np.float32)
                # Normalize
                if self.bounds is not None:
                    x_min = self.bounds[k][0]
                    x_max = self.bounds[k][1]
                    traj[k] = 2 * (var - x_min) / (x_max - x_min) - 1
                else:
                    traj[k] = var

            # Obtain the grid only once
            if grid is None:
                grid = self._get_grid(file)

        return traj, grid

    def trajectory_tags(self):
        return set([traj["traj_tag"] for traj in self.data_chunks])

    def get_stats(self):
        stats = {k: [] for k in self.active_keys}
        for s in self.data:
            for k, v in s["x"].items():
                stats[k].append(v)

        for k, v in stats.items():
            v = np.array(v)
            stats[k] = {"mean": v.mean(), "std": v.std()}

        return stats

    def _load_h5_chunk(self, tag, path):
        with h5py.File(path, "r") as file:
            data = {
                k: file[k][:]
                for k in self.active_keys + ["R_mesh(nR,nZ)", "Z_mesh(nR,nZ)"]
            }
        return {"tag": tag, "file": data}

    def _get_traj_data(self):
        # Get the number of timesteps in the trajectory
        chunk = self.data_chunks[0]
        with h5py.File(chunk["file_path"], "r") as file:
            traj_length = file["n_times"][0]
            available_keys = file["variables_list"][()].decode("utf-8").split(",")

        return traj_length, available_keys

    def get_bounds(self):
        # Get bounds for all keys across all trajectories in parallel
        bounds = {k: [np.inf, -np.inf] for k in self.active_keys}

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._get_local_bounds_for_file,
                    data_chunk["file_path"],
                    self.active_keys,
                )
                for data_chunk in self.data_chunks
            ]

            if self.use_tqdm:
                futures = tqdm(futures, "Calculating boundaries for normalization...")
            else:
                print("Calculating boundaries for normalization...")

            for future in futures:
                local_bounds = future.result()
                for k in self.active_keys:
                    bounds[k][0] = min(bounds[k][0], local_bounds[k][0])
                    bounds[k][1] = max(bounds[k][1], local_bounds[k][1])

        return bounds

    def _get_local_bounds_for_file(self, file_path, active_keys):
        # Open the file and calculate bounds for the active keys
        with h5py.File(file_path, "r") as file:
            local_bounds = {
                k: [np.array(file[k]).min(), np.array(file[k]).max()]
                for k in active_keys
            }
        return local_bounds

    def _normalize_sample(self, sample):
        # Normalize the whole data to [-1, 1], according to Neural-Parareal
        for k in self.active_keys:
            x_min = self.bounds[k][0]
            x_max = self.bounds[k][1]
            sample["x"][k] = 2 * (sample["x"][k] - x_min) / (x_max - x_min) - 1
            sample["y"][k] = 2 * (sample["y"][k] - x_min) / (x_max - x_min) - 1
        return sample

    def normalize(self, bounds):
        self.bounds = bounds

    def trajectory_sampler(self):
        trajectory_ids = np.arange(len(self.data_chunks))
        return TrajectoryPriorityRandomSampler(
            self, trajectory_ids, self.n_samples_per_file
        )

    def collate(self,
                sample: Dict,
                augmentations: Optional[List[Callable]] = None,
                device: torch.DeviceObjType = "cuda",
                ) -> Annotated[Tuple[torch.Tensor], 4]:
        x, grid = sample["x"], sample["grid"]

        x = torch.cat([x[k].unsqueeze(1) for k in x], dim=1)
        if x.ndim == 5:
            x = einops.rearrange(x, "b c t x y -> b (c t) x y")

        grid = grid.permute(0, 2, 3, 1)  # (bs, x, y, dim)
        # expand grid to image dimension
        grid[..., 0] = (grid[..., 0] - grid[..., 0].min()) / (
            grid[..., 0].max() - grid[..., 0].min()
        )
        grid[..., 1] = (grid[..., 1] - grid[..., 1].min()) / (
            grid[..., 1].max() - grid[..., 1].min()
        )
        # N = grid.shape[1]
        # grid = grid * N

        y = sample.get("y", None)
        if y is not None:
            y = torch.cat([y[k].unsqueeze(1) for k in y], dim=1)
            y = y.permute(2, 0, 1, 3, 4)  # (t, b, c, x, y)

        ts = sample.get("ts", None)
        if ts is not None:
            if not isinstance(ts, torch.Tensor):
                ts = torch.tensor([ts])
            ts = ts.to(device)

        if augmentations is not None:
            for aug_fn in augmentations:
                x = aug_fn(x)
            x = x.to(device)

        return x.to(device), grid.to(device), y, ts


class TrajectoryPriorityRandomSampler(Sampler[int]):
    def __init__(
        self,
        data: Dataset,
        trajectory_ids: List[int],
        n_samples_per_traj: int,
        shuffle_trajs: bool = False,
    ):
        self.data = data
        self.n_samples_per_traj = n_samples_per_traj
        self._current_traj_tag = None
        self.trajectory_ids = trajectory_ids
        self.ts_ids = np.arange(self.n_samples_per_traj)
        if shuffle_trajs:
            random.shuffle(self.trajectory_ids)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[int]:
        # sample random traj ID
        for chunk_id in self.trajectory_ids:
            # sample random dt
            random.shuffle(self.ts_ids)
            for t_id in self.ts_ids:
                idx = chunk_id * self.n_samples_per_traj + t_id
                yield idx

def denormalize(trajectory, bounds):
    # Denormalize the trajectory from [-1, 1] back to original scale
    for k, v in trajectory.items():
        x_min = bounds[k][0]
        x_max = bounds[k][1]
        v = 0.5 * (v + 1) * (x_max - x_min) + x_min
    return trajectory
