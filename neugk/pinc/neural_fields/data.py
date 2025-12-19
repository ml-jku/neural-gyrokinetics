from typing import Tuple, Sequence, Optional, Union


import h5py
import numpy as np
import torch
from einops import rearrange
from math import exp

from torch.utils.data import Dataset

from neugk.integrals import FluxIntegral
from neugk.pinc.neural_fields.nf_utils import df_fft, df_ifft


class CycloneNFDataset(Dataset):
    def __init__(
        self,
        trajectory: str,
        timesteps: Union[int, Sequence[int]],
        path: str = "/restricteddata/ukaea/gyrokinetics/preprocessed",
        realpotens: bool = False,
        normalize: Optional[str] = None,
        normalize_coords: bool = False,
        beta1: float = 1.0,
        beta2: float = 0.0,
        spatial_fft: bool = False,
        separate_ky_modes: Optional[Sequence[int]] = None,
        flux_fields: bool = False,
        flux_fields_train: bool = False,
    ):
        super().__init__()

        self.raw_path = f"{path}/{trajectory.replace('.h5', '')}"

        trajectory = trajectory.replace(".h5", "")
        h5_file = f"{path}/{trajectory}_ifft{'_realpotens' if realpotens else ''}.h5"

        self.normalize = normalize
        self.flux_fields = flux_fields
        self.spatial_fft = spatial_fft
        self.flux_fields_train = flux_fields and flux_fields_train
        self.realpotens = realpotens
        self.beta1 = beta1
        self.beta2 = beta2

        if isinstance(timesteps, int):
            timesteps = [timesteps]
        self.timesteps = timesteps

        # load all samples
        self.df, self.phi, self.flux, self.geom = self._load_gkw_data(
            h5_file, timesteps, separate_ky_modes
        )

        grid = torch.meshgrid(
            [torch.arange(d) for d in self.df.shape[1:]], indexing="ij"
        )
        self.grid = torch.stack(grid, dim=-1)
        self.indices = self.grid.flatten(0, -2)
        # normalize coords
        if normalize_coords:
            norm_ndim = torch.tensor(self.grid.shape[:-1])[
                (None,) * (self.grid.ndim - 1)
            ]
        else:
            norm_ndim = torch.tensor(1.0)
        self.grid = self.grid / norm_ndim
        self.norm_ndim = norm_ndim
        self.f_df = rearrange(self.df, "c ... -> c (...)")
        self.f_grid = rearrange(self.grid, "... d -> (...) d")
        if self.flux_fields_train:
            self.f_flux = rearrange(self.flux, "c ... -> c (...)")

        self._get_norm_stats()

        sgrid = np.loadtxt(f"/restricteddata/ukaea/gyrokinetics/raw/{trajectory}/sgrid")
        self.ds = sgrid[1] - sgrid[0]

    def _load_gkw_data(
        self,
        h5_file: str,
        timesteps: Sequence[int],
        separate_ky_modes: Optional[Sequence[int]] = None,
    ):
        dfs, phis, fluxes = [], [], []

        with h5py.File(h5_file, "r") as data:
            geom = {k: np.array(v[()]) for k, v in data["geometry"].items()}
            for t in timesteps:
                dfs.append(data[f"data/timestep_{str(t).zfill(5)}"][:])
                phis.append(data[f"data/poten_{str(t).zfill(5)}"][:])
                fluxes.append(data["metadata/fluxes"][t])

        dfs = torch.from_numpy(np.stack(dfs, 0)).squeeze(0)
        phis = torch.from_numpy(np.stack(phis, 0)).squeeze(0)
        fluxes = torch.from_numpy(np.stack(fluxes, 0)).squeeze(0)
        geom = {k: torch.from_numpy(g).squeeze(0) for k, g in geom.items()}
        if len(timesteps) > 1:
            dfs = rearrange(dfs, "t c ... -> c t ...")

        if self.flux_fields or self.realpotens:
            geom_ = {k: g[None] for k, g in geom.items()}
            dfs_ = dfs.clone()
            if len(timesteps) == 1:
                dfs_ = dfs_[:, None]
            phis_int, fluxes_int = [], []
            for t in range(len(timesteps)):
                assert dfs_[:, t].shape[0] == 2
                integrator = FluxIntegral(flux_fields=self.flux_fields)
                phi_t, (_, fluxes_t, _) = integrator(geom_, df=dfs_[None, :, t])
                fluxes_int.append(fluxes_t.squeeze(0))
                phis_int.append(phi_t.squeeze(0))
            if self.flux_fields:
                fluxes = torch.stack(fluxes_int, 0).squeeze(0)
            if self.realpotens:
                # replace potentials (realpotens incompatible with losses)
                phis = torch.stack(phis_int, 0).squeeze(0)

        if len(timesteps) > 1:
            phis = rearrange(phis, "t c ... -> c t ...")
            if self.flux_fields:
                fluxes = rearrange(fluxes, "t c ... -> c t ...")

        if self.spatial_fft or separate_ky_modes is not None:
            dfs = self._split_into_bands(dfs, self.spatial_fft, separate_ky_modes)

        return dfs, phis, fluxes, geom

    def _split_into_bands(
        self,
        df: np.ndarray,
        spatial_fft: bool,
        separate_ky_modes: Optional[Sequence[int]] = None,
    ):
        df = df_fft(df)
        if separate_ky_modes is not None and len(separate_ky_modes) > 0:
            # split modes
            n_ky_modes = df.shape[-1]
            filters = []
            for mode in separate_ky_modes:
                band = df.clone()
                mask = torch.zeros_like(band)
                mask[..., mode] = 1.0
                band = band * mask
                filters.append(band)
            # every other frequency
            residual = df.clone()
            flat_selected = []
            for group in separate_ky_modes:
                flat_selected.extend([group] if isinstance(group, int) else group)
            # modes left out
            residual_modes = set(list(range(n_ky_modes)))
            residual_modes = list(residual_modes.difference(set(flat_selected)))
            if len(residual_modes) > 0:
                residual = df.clone()
                mask = torch.zeros_like(residual)
                mask[..., residual_modes] = 1.0
                residual = residual * mask
                filters.append(residual)
            if not spatial_fft:
                filters = [df_ifft(f) for f in filters]
            # concat modes on channels
            df = torch.cat(filters, 0)
        return df

    def _get_norm_stats(self):
        self.scale = {
            "df": torch.ones((self.f_df.shape[0], 1)),
            "phi": torch.ones((2, *[1] * (self.phi.ndim - 1))),
        }
        self.shift = {
            "df": torch.zeros((self.f_df.shape[0], 1)),
            "phi": torch.zeros((2, *[1] * (self.phi.ndim - 1))),
        }
        if self.flux_fields:
            self.scale["flux"] = torch.ones((2, *[1] * (self.flux.ndim - 1)))
            self.shift["flux"] = torch.zeros((2, *[1] * (self.flux.ndim - 1)))
        if self.normalize is not None:
            # distribution stats
            if self.normalize == "minmax":
                df_min, df_max = self.f_df.min(1).values, self.f_df.max(1).values
                self.scale["df"] = (df_max - df_min) / self.beta1
                self.shift["df"] = df_min + self.scale["df"] * self.beta2
            if self.normalize == "zscore":
                df_mean, df_std = self.f_df.mean(1), self.f_df.std(1)
                self.scale["df"] = df_std / self.beta1
                self.shift["df"] = df_mean + self.scale["df"] * self.beta2
            self.scale["df"] = self.scale["df"][:, None]
            self.shift["df"] = self.shift["df"][:, None]
            # potential stats
            if self.normalize == "minmax":
                phi_min = self.phi.flatten(1, -1).min(1).values
                phi_max = self.phi.flatten(1, -1).max(1).values
                self.scale["phi"] = (phi_max - phi_min) / self.beta1
                self.shift["phi"] = phi_min + self.scale["phi"] * self.beta2
            if self.normalize == "zscore":
                phi_mean = self.phi.flatten(1, -1).mean(1)
                phi_std = self.phi.flatten(1, -1).std(1)
                self.scale["phi"] = phi_std / self.beta1
                self.shift["phi"] = phi_mean + self.scale["phi"] * self.beta2
            self.scale["phi"] = self.scale["phi"][:, *[None] * (self.phi.ndim - 1)]
            self.shift["phi"] = self.shift["phi"][:, *[None] * (self.phi.ndim - 1)]
            if self.flux_fields:
                # flux field stats
                if self.normalize == "minmax":
                    flux_min = self.flux.flatten(1, -1).min(1).values
                    flux_max = self.flux.flatten(1, -1).max(1).values
                    self.scale["flux"] = (flux_max - flux_min) / self.beta1
                    self.shift["flux"] = flux_min + self.scale["flux"] * self.beta2
                if self.normalize == "zscore":
                    flux_mean = self.flux.flatten(1, -1).mean(1)
                    flux_std = self.flux.flatten(1, -1).std(1)
                    self.scale["flux"] = flux_std / self.beta1
                    self.shift["flux"] = flux_mean + self.scale["flux"] * self.beta2
                fndim = self.flux.ndim - 1
                self.scale["flux"] = self.scale["flux"][:, *[None] * fndim]
                self.shift["flux"] = self.shift["flux"][:, *[None] * fndim]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, int):
            index = [index]
        if self.flux_fields_train:
            flux, coords = self.f_flux[:, index], self.f_grid[index, :]
            shift = self.shift["flux"].flatten(1, -1)
            scale = self.scale["flux"].flatten(1, -1)
            flux = (flux - shift) / scale
            return flux.T, coords
        else:
            df, coords = self.f_df[:, index], self.f_grid[index, :]
            df = (df - self.shift["df"]) / self.scale["df"]
            return df.T, coords

    def to(self, device: torch.device):
        self.f_df = self.f_df.to(device)
        self.phi = self.phi.to(device)
        self.f_grid = self.f_grid.to(device)
        self.geom = {k: v.to(device) for k, v in self.geom.items()}
        for k in self.scale:
            self.scale[k] = self.scale[k].to(device)
            self.shift[k] = self.shift[k].to(device)
        self.flux = self.flux.to(device)
        if self.flux_fields_train:
            self.f_flux = self.f_flux.to(device)
        self.norm_ndim = self.norm_ndim.to(device)
        return self

    def shuffle(self):
        perm = torch.randperm(self.f_grid.shape[0])
        self.f_df = self.f_df[:, perm]
        self.f_grid = self.f_grid[perm, :]

    def cpu(self):
        return self.to("cpu")

    @property
    def device(self):
        return self.f_df.device

    @property
    def ndim(self) -> int:
        return self.df.ndim - 1

    @property
    def nchannels(self) -> int:
        return self.df.shape[0]

    @property
    def full_df(self) -> torch.Tensor:
        if self.nchannels == 2:
            return self.df
        else:
            return sum(self.df.chunk(self.nchannels // 2))


class CycloneNFDataLoader:
    def __init__(
        self,
        dataset: CycloneNFDataset,
        batch_size: int,
        preload: bool = False,
        shuffle: bool = False,
        pin_memory: bool = False,
        subsample: float = 1.0,
        prefetch_factor: int = 2,
    ):
        self.dataset = dataset
        self._batch_size = batch_size
        self.preload = preload
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.subsample = subsample
        self.prefetch_factor = prefetch_factor
        self.device = "cpu"

    def __len__(self):
        return int(
            (len(self.dataset) + self.batch_size - 1)
            // self.batch_size
            * self.subsample
        )

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()

        indices = int(len(self.dataset) * self.subsample)
        for start_idx in range(0, indices, self.batch_size):
            dfs, coords = self.dataset[start_idx : start_idx + self.batch_size]

            if self.pin_memory and dfs.device == "cpu":
                dfs, coords = dfs.pin_memory(), coords.pin_memory()
            if not self.preload:
                dfs, coords = dfs.to(self.device), coords.to(self.device)

            yield dfs, coords

    def to(self, device: torch.device):
        if self.preload:
            self.dataset.to(device)
        else:
            self.device = device


class CycloneNFDataLoaderDecay(CycloneNFDataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int,
        final_batch_size: int,
        decay_rate: float,
        total_steps: int,
        preload: bool = False,
        shuffle: bool = False,
        pin_memory: bool = False,
        subsample: float = 1.0,
        prefetch_factor: int = 2,
    ):
        super().__init__(
            dataset,
            batch_size,
            preload,
            shuffle,
            pin_memory,
            subsample,
            prefetch_factor,
        )

        self.final_batch_size = final_batch_size
        self.decay_rate = decay_rate
        self.total_steps = total_steps
        self.step = 0

    def __len__(self):
        return int(
            (len(self.dataset) + self.final_batch_size - 1)
            // self.final_batch_size
            * self.subsample
        )

    @property
    def batch_size(self):
        t = min(self.step, self.total_steps)
        batch_size = self.final_batch_size + (
            self.batch_size - self.final_batch_size
        ) * exp(-self.decay_rate * t)
        return max(1, int(round(batch_size)))

    def __iter__(self):
        super().__iter__()
        self.step += 1
