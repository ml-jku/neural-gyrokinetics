from typing import Optional, Dict, Sequence
import h5py
import os
import pickle
from tqdm import tqdm
import hashlib
from dataclasses import dataclass
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import torch.distributed as dist
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

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
    def __init__(
        self,
        *args,
        conditions: Sequence[str],
        precomputed_latents: Optional[Dict] = None,
        autoencoder: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.conditions = conditions
        self.precomputed_latents = precomputed_latents
        self.autoencoder = autoencoder

    def _recompute_stats(self, key: str, offset: int = 0):
        def process_t_idx(t_idx, key):
            file_index, t_index = self.flat_index_to_file_and_tstep[t_idx]
            with h5py.File(self.files[file_index], "r", swmr=True) as f:
                sample = self._load_data(f, file_index, t_index)
            if key == "df":
                x = sample["x"]
                if self.decouple_mu:
                    norm_axes = (1, 3, 4, 5)
                else:
                    norm_axes = (1, 2, 3, 4, 5)
            elif key == "phi":
                x = sample["phi"]
                if x.ndim == 3:
                    x = np.expand_dims(x, 0)
                norm_axes = (1, 2, 3)
            else:
                x = np.array([sample["flux"]], dtype=np.float32)
                norm_axes = (0,)

            if self.separate_zf and key == "df":
                x = self._separate_zf(x)

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
        has_mu = "_mu" if self.decouple_mu else ""
        std_filter_tag = (
            f"_std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )
        stats_dump_pkl = (
            f"diff_{key}_offset{offset}{has_mu}{std_filter_tag}_{file_hash}_stats.pkl"
        )
        stats_dump_pkl = os.path.join(self.dir, stats_dump_pkl)

        if os.path.exists(stats_dump_pkl):
            stats = pickle.load(open(stats_dump_pkl, "rb"))
        else:
            process_inds = partial(process_t_idx, key=key)
            stats = None
            with ThreadPoolExecutor(self.num_workers) as executor:
                metrics_gen = tqdm(
                    executor.map(process_inds, t_indices),
                    total=len(t_indices),
                    desc=f"computing stats for {key}",
                )

                for metrics in metrics_gen:
                    x_mean, x_var, x_min, x_max = metrics
                    if stats is None:
                        stats = RunningMeanStd(shape=x_mean.shape)

                    # update moments
                    stats.update(x_mean, x_var, x_min, x_max)

            pickle.dump(stats, open(stats_dump_pkl, "wb"))
            print(f"saved recomputed stats to {stats_dump_pkl}")
        return stats

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneAESample:
        # lookup file index and time index from flat index
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if getattr(self, "precomputed_latents", None) is not None:
            sample = self._load_data(None, file_index, t_index)
            x = sample["x"]
        else:
            with h5py.File(self.files[file_index], "r", swmr=True) as f:
                sample = self._load_data(f, file_index, t_index)
            x = sample["x"]
            if x is not None and self.separate_zf:
                x = self._separate_zf(x)
        # accessory fields
        phi, flux = sample["phi"], sample["flux"]
        avg_flux = self.get_avg_flux(file_index)
        # conditioning fields
        timestep = sample["timestep"]
        conditioning = None
        if self.conditions is not None and len(self.conditions) > 0:
            conditioning = torch.stack(
                [torch.tensor(sample[k], dtype=self.dtype) for k in self.conditions],
                dim=-1,
            )
        geometry = sample["geometry"]

        if self.normalization is not None and get_normalized:
            # skip normalization if latents are precomputed
            if x is not None and getattr(self, "precomputed_latents", None) is None:
                x = self.normalize(file_index, df=x)

            if phi is not None:
                phi = self.normalize(file_index, phi=phi)

            if flux is not None:
                flux = self.normalize(file_index, flux=flux)

        if phi is not None and phi.ndim == 3:
            phi = phi[None]

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

    def _load_data(self, data, file_index, t_index) -> Dict:
        if getattr(self, "precomputed_latents", None) is not None:
            return self.precomputed_latents[(file_index, t_index)]

        orig_t_index = t_index + self.offsets[file_index]
        xs, phis, fluxes = [], [], []
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
                phis.append(data[f"data/{phi_name}"][:])

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
            phis = phis[0] if self.bundle_seq_length == 1 else np.stack(phis, axis=1)
        else:
            phis = None

        sample["x"] = xs
        sample["phi"] = phis
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
        **kwargs,
    ):
        if df is not None:
            if self.autoencoder is not None:
                condition = kwargs["condition"]
                ch = 2 + 2 * self.separate_zf
                dummy = torch.zeros((1, ch, *self.resolution), device=condition.device)
                dummy_cond = condition[0:1]
                _, pad_axes = self.autoencoder.encode(dummy, condition=dummy_cond)
                df = self.autoencoder.decode(df, pad_axes, condition=condition)["df"]
            field = "df"
            x = df
        elif phi is not None:
            field = "phi"
            x = phi
        elif flux is not None:
            return flux
        else:
            raise ValueError

        scale, shift = self._get_scale_shift(file_index, field, x)
        return x * scale + shift

    def collate(self, batch: Sequence[CycloneAESample]):
        # batch is a list of cyclonesamples
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
            conditioning=(
                torch.stack([sample.conditioning for sample in batch])
                if batch[0].conditioning is not None
                else None
            ),
            geometry=tree_map(lambda *x: torch.stack(x), *[s.geometry for s in batch]),
        )

    @torch.no_grad()
    def precompute_latents(
        self,
        rank: int,
        dataloader: DataLoader,
        autoencoder: torch.nn.Module,
        device: torch.device = "cuda",
        latent_stats: Optional[RunningMeanStd] = None,
    ):
        self.autoencoder = autoencoder
        # unique filenames
        file_hash = hashlib.sha256("".join(sorted(self.files)).encode()).hexdigest()[:8]
        latents_dump_pkl = os.path.join(
            self.dir, f"diff_{self.split}_latents_{file_hash}.pkl"
        )
        # load latents from disk if already computed
        if os.path.exists(latents_dump_pkl):
            if rank == 0:
                print(f"Loading precomputed latents from {latents_dump_pkl}")
            with open(latents_dump_pkl, "rb") as f:
                self.precomputed_latents = pickle.load(f)
            if dist.is_initialized():
                dist.barrier()
        else:
            # TODO(gg) recreate to avoid deadlock
            tmp_loader = DataLoader(
                dataset=dataloader.dataset,
                batch_size=32,
                num_workers=4,
                pin_memory=True,
                collate_fn=dataloader.collate_fn,
                drop_last=dataloader.drop_last,
                shuffle=False,
            )
            autoencoder.eval()
            autoencoder.to(device)
            latents_dict = {}
            desc = f"Precomputing {self.split} latents (rank:{rank})"
            for batch in tqdm(tmp_loader, desc=desc):
                df = batch.df.to(device)
                cond = (
                    batch.conditioning.to(device)
                    if hasattr(batch, "conditioning")
                    else None
                )
                z, _ = autoencoder.encode(df, condition=cond)
                z = z.cpu().numpy()
                # cache local batch latents
                for i in range(len(batch.file_index)):
                    f_idx = batch.file_index[i].item()
                    t_idx = batch.timestep_index[i].item()
                    with h5py.File(self.files[f_idx], "r", swmr=True) as f:
                        cached_latents = self.precomputed_latents
                        self.precomputed_latents = None
                        sample = self._load_data(f, f_idx, t_idx)
                        self.precomputed_latents = cached_latents
                    sample["x"] = z[i]
                    latents_dict[(f_idx, t_idx)] = sample
            # ddp merge
            if dist.is_initialized():
                gathered_dict = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_dict, latents_dict)
                full_latents_dict = {}
                for d in gathered_dict:
                    full_latents_dict.update(d)
                self.precomputed_latents = full_latents_dict
            else:
                self.precomputed_latents = latents_dict
            # save latents on main process
            if rank == 0:
                with open(latents_dump_pkl, "wb") as f:
                    pickle.dump(self.precomputed_latents, f)
                print(f"Saved precomputed latents to {latents_dump_pkl}")
            if dist.is_initialized():
                dist.barrier()

        if self.split == "train":
            stats = None
            l2_norms = []
            for sample in self.precomputed_latents.values():
                x = sample["x"]
                norm_axes = tuple(range(0, x.ndim))
                x_mean = np.mean(x, axis=norm_axes, keepdims=True)
                x_var = np.var(x, axis=norm_axes, keepdims=True)
                x_min = np.min(x, axis=norm_axes, keepdims=True)
                x_max = np.max(x, axis=norm_axes, keepdims=True)
                l2_norms.append(np.sqrt(np.sum(x**2, axis=norm_axes, keepdims=True)))
                if stats is None:
                    stats = RunningMeanStd(shape=x_mean.shape)
                stats.update(x_mean, x_var, x_min, x_max)
            self.latent_stats = stats
            l2_norm = np.mean(l2_norms, axis=0)
            if rank == 0:
                print(f"Latent mean: {np.squeeze(stats.mean)}")
                print(f"Latent var: {np.squeeze(stats.var)}")
                print(f"Latent l2 norm: {np.squeeze(l2_norm)}")
            # wait until done
            if dist.is_initialized():
                dist.barrier()
        else:
            assert latent_stats is not None
            self.latent_stats = latent_stats
