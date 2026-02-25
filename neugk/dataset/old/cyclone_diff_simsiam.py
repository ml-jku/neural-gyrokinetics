from typing import Optional, Tuple, Dict, Sequence
import h5py
import os
import pickle
import tqdm
import hashlib
from dataclasses import dataclass
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from einops import rearrange

import torch
import numpy as np
from torch.utils._pytree import tree_map

from neugk.dataset.cyclone_diff import CycloneAESample, CycloneAEDataset
from neugk.dataset import CycloneDataset
from neugk.utils import RunningMeanStd


@dataclass
class CycloneSimSiamSample(CycloneAESample):
    df_aug: torch.Tensor = None
    timestep_index_aug: torch.Tensor = None


class CycloneSimSiamDataset(CycloneAEDataset):
    def __getitem__(
        self, index: int, get_normalized: bool = True
    ) -> CycloneSimSiamSample:
        # lookup file index and time index from flat index
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        with h5py.File(self.files[file_index], "r") as f:
            sample = self._load_data(f, file_index, t_index)
        # k-fields
        x, x_aug = sample["x"], sample["x_aug"]
        t_index_aug = sample["t_index_aug"]
        if x is not None and self.separate_zf:
            x = self._separate_zf(x)
            x_aug = self._separate_zf(x_aug)
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
                x_aug = self.normalize(file_index, df=x_aug)

            if phi is not None:
                phi = self.normalize(file_index, phi=phi)

            if flux is not None:
                flux = self.normalize(file_index, flux=flux)

        if phi is not None and phi.ndim == 3:
            phi = phi[None]

        return CycloneSimSiamSample(
            df=torch.tensor(x, dtype=self.dtype) if x is not None else None,
            df_aug=torch.tensor(x_aug, dtype=self.dtype) if x is not None else None,
            phi=(torch.tensor(phi, dtype=self.dtype) if phi is not None else None),
            flux=torch.tensor(flux, dtype=self.dtype),
            avg_flux=torch.tensor(avg_flux, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            timestep_index_aug=torch.tensor(t_index_aug, dtype=torch.long),
            geometry=tree_map(lambda x: torch.from_numpy(x), geometry),
            # conditioning
            timestep=torch.tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
        )

    def _load_data(
        self, data, file_index, t_index
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        offset, n_ts = self.offsets[file_index], self.file_num_timesteps[file_index]
        orig_t_index = t_index + offset
        xs, xs2, phis, fluxes = [], [], [], []
        t_aug = []
        for i in range(self.bundle_seq_length):
            # read the input
            k_name = "timestep_" + str(orig_t_index + i).zfill(5)
            while (rnd := np.random.randint(offset, n_ts + offset)) == orig_t_index + i:
                pass
            k_name2 = "timestep_" + str(rnd).zfill(5)
            t_aug.append(rnd)
            phi_name = "poten_" + str(orig_t_index + i).zfill(5)
            if "df" in self.input_fields:
                k = data[f"data/{k_name}"][:]
                k2 = data[f"data/{k_name2}"][:]
                # select only active re/im parts
                if all(self.active_keys == np.array([0, 1])):
                    xs.append(k)
                    xs2.append(k2)
                else:
                    xs.append(k[self.active_keys])
                    xs2.append(k2[self.active_keys])
            if "phi" in self.input_fields:
                phis.append(data[f"data/{phi_name}"][:])

            # target flux
            flux = data["metadata/fluxes"][orig_t_index + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.input_fields:
            # stack to shape (c, t, v1, v2, s, x, y)
            xs = xs[0] if self.bundle_seq_length == 1 else np.stack(xs, axis=1)
            xs2 = xs2[0] if self.bundle_seq_length == 1 else np.stack(xs2, axis=1)
        else:
            xs = xs2 = None

        if "phi" in self.input_fields:
            phis = phis[0] if self.bundle_seq_length == 1 else np.stack(phis, axis=1)
        else:
            phis = None

        t_aug = t_aug[0] if self.bundle_seq_length == 1 else np.stack(t_aug, axis=1)

        sample["x"] = xs
        sample["x_aug"] = xs2
        sample["phi"] = phis
        sample["flux"] = np.array(fluxes).squeeze()
        # load conditioning
        sample["timestep"] = data["metadata/timesteps"][orig_t_index]
        sample["t_index_aug"] = t_aug
        sample["itg"] = data["metadata/ion_temp_grad"][:].squeeze()
        sample["dg"] = data["metadata/density_grad"][:].squeeze()
        sample["s_hat"] = data["metadata/s_hat"][:].squeeze()
        sample["q"] = data["metadata/q"][:].squeeze()
        # load geometry
        sample["geometry"] = {k: np.array(v[()]) for k, v in data["geometry"].items()}
        return sample

    def collate(self, batch: Sequence[CycloneSimSiamSample]):
        # batch is a list of CycloneSamples
        return CycloneSimSiamSample(
            df=(
                torch.stack([s.df for s in batch]) if batch[0].df is not None else None
            ),
            df_aug=(
                torch.stack([s.df_aug for s in batch])
                if batch[0].df_aug is not None
                else None
            ),
            phi=(
                torch.stack([s.phi for s in batch])
                if batch[0].phi is not None
                else None
            ),
            flux=torch.stack([s.flux for s in batch]),
            avg_flux=torch.stack([s.avg_flux for s in batch]),
            file_index=torch.stack([s.file_index for s in batch]),
            timestep_index=torch.stack([s.timestep_index for s in batch]),
            timestep_index_aug=torch.stack([s.timestep_index_aug for s in batch]),
            timestep=torch.stack([s.timestep for s in batch]),
            conditioning=torch.stack([s.conditioning for s in batch]),
            geometry=tree_map(lambda *x: torch.stack(x), *[s.geometry for s in batch]),
        )
