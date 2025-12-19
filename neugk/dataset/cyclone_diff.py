from typing import Optional, Tuple, Dict, Sequence
import h5py
import os
import pickle
import tqdm
import hashlib
from dataclasses import dataclass
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from torch.utils._pytree import tree_map

from neugk.dataset import CycloneDataset
from neugk.utils import RunningMeanStd


@dataclass
class CycloneAESample:
    df: torch.Tensor
    phi: torch.Tensor
    flux: torch.Tensor
    avg_flux: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    # conditioning
    timestep: torch.Tensor
    conditioning: torch.Tensor
    # geometric tensors for integrals
    geometry: Optional[Dict[str, torch.Tensor]] = None

    def pin_memory(self):
        if self.df is not None:
            self.df = self.df.pin_memory()
        if self.phi is not None:
            self.phi = self.phi.pin_memory()
        return self


class CycloneAEDataset(CycloneDataset):
    def __init__(self, *args, stage: str, conditions: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)

        self.stage = stage
        self.conditions = conditions

    def _recompute_stats(self, key: str, offset: int = 0):
        if key in ["df", "phi"]:
            t_indices = list(range(0, self.length, 2))
        else:
            t_indices = list(range(0, self.length))

        def process_t_idx(t_idx, key):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]
            with h5py.File(self.files[file_index], "r") as f:
                sample = self._load_data(f, file_index, t_index)

            if key == "df":
                x = sample["x"]
                norm_axes = (1, 2, 3, 4, 5)
            elif key == "phi":
                x = sample["phi"]
                if len(x.shape) == 3:
                    x = np.expand_dims(x, 0)
                norm_axes = (1, 2, 3)
            else:
                x = np.array([sample["flux"]], dtype=np.float32)
                norm_axes = (0,)

            if self.separate_zf and key == "df":
                x = self._separate_zf(x)

            # Compute metrics for x and y
            x_mean = np.mean(x, norm_axes, keepdims=True)
            x_var = np.var(x, norm_axes, keepdims=True)
            x_min = np.min(x, norm_axes, keepdims=True)
            x_max = np.max(x, norm_axes, keepdims=True)
            return (x_mean, x_var, x_min, x_max)

        traj_hash = hashlib.sha256("".join(sorted(self.files)).encode()).hexdigest()[:8]
        stats_dump_pkl = os.path.join(
            self.dir, f"diff_{key}_offset{offset}_{traj_hash}_stats.pkl"
        )
        if os.path.exists(stats_dump_pkl):
            stats = pickle.load(open(stats_dump_pkl, "rb"))
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
                    x_mean, x_var, x_min, x_max = metrics
                    if stats is None:
                        stats = RunningMeanStd(shape=x_mean.shape)
                    stats.update(x_mean, x_var, x_min, x_max)

                pickle.dump(stats, open(stats_dump_pkl, "wb"))

        return stats

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneAESample:
        """
        Args:
            index (int): Flat index with dataset ordering.

        Returns:
            CycloneAESample: dataclass
                - df (torch.Tensor): target density, shape `(c, bundle, v1, v2, s, x, y)`
                - phi (torch.Tensor): target potential, shape `(c, bundle, x, s, y)`
                - flux (torch.Tensor): target flux, shape `()`
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
        x = sample["x"]
        if self.separate_zf:
            x = self._separate_zf(x)
        # accessory fields
        phi, flux = sample["phi"], sample["flux"]
        avg_flux = self.get_avg_flux(file_index)
        # conditioning fields
        timestep = sample["timestep"]
        conditioning = torch.stack(
            [torch.tensor(sample[k], dtype=self.dtype) for k in self.conditions], dim=-1
        )
        geometry = sample["geometry"]
        if self.normalization is not None and get_normalized:
            if x is not None:
                x = self.normalize(file_index, df=x)

            if phi is not None:
                phi = self.normalize(file_index, phi=phi)

            if flux is not None:
                flux = self.normalize(file_index, flux=flux)

        return CycloneAESample(
            df=torch.tensor(x, dtype=self.dtype) if x is not None else None,
            phi=(torch.tensor(phi, dtype=self.dtype) if phi is not None else None),
            flux=torch.tensor(flux, dtype=self.dtype),
            avg_flux=torch.tensor(avg_flux, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            geometry=tree_map(lambda x: torch.from_numpy(x), geometry),
            # conditioning
            timestep=torch.tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
        )

    def _load_data(
        self, data, file_index, t_index
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        orig_t_index = t_index + self.offsets[file_index]
        xs, potens, fluxes = [], [], []
        for i in range(self.bundle_seq_length):
            # read the input
            k_name = "timestep_" + str(orig_t_index + i).zfill(5)
            phi_name = "poten_" + str(orig_t_index + i).zfill(5)
            if "df" in self.input_fields:
                k = data[f"data/{k_name}"][:]
                # select only active re/im parts
                if all(self.active_keys == np.array([0, 1])):
                    xs.append(k)
                else:
                    xs.append(k[self.active_keys])
            if "phi" in self.input_fields:
                phi = data[f"data/{phi_name}"][:]
                potens.append(phi)

            # target flux
            flux = data["metadata/fluxes"][orig_t_index + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.input_fields:
            # stack to shape (c, t, v1, v2, s, x, y)
            xs = xs[0] if self.bundle_seq_length == 1 else np.stack(xs, axis=1)
        else:
            xs = None

        if "phi" in self.input_fields:
            potens = (
                potens[0] if self.bundle_seq_length == 1 else np.stack(potens, axis=1)
            )

        else:
            potens = None

        sample["x"] = xs
        sample["phi"] = potens
        sample["flux"] = np.array(fluxes).squeeze()
        # load conditioning
        sample["timestep"] = data["metadata/timesteps"][orig_t_index]
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
            # field = "flux"
            # x = flux
            return flux
        else:
            raise ValueError

        scale, shift = self._get_scale_shift(file_index, field, x)
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
            # field = "flux"
            # x = flux
            return flux
        else:
            raise ValueError

        scale, shift = self._get_scale_shift(file_index, field, x)
        return x * scale + shift

    def collate(self, batch: Sequence[CycloneAESample]):
        # batch is a list of CycloneSamples
        return CycloneAESample(
            df=(
                torch.stack([sample.df for sample in batch])
                if batch[0].df is not None
                else None
            ),
            phi=(
                torch.stack([sample.phi for sample in batch])
                if batch[0].phi is not None
                else None
            ),
            flux=torch.stack([sample.flux for sample in batch]),
            avg_flux=torch.stack([sample.avg_flux for sample in batch]),
            file_index=torch.stack([sample.file_index for sample in batch]),
            timestep_index=torch.stack([sample.timestep_index for sample in batch]),
            timestep=torch.stack([sample.timestep for sample in batch]),
            conditioning=torch.stack([sample.conditioning for sample in batch]),
            geometry=tree_map(lambda *x: torch.stack(x), *[s.geometry for s in batch]),
        )
