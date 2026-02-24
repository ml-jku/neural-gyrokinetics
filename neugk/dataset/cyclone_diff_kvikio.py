from typing import Optional, Dict, Sequence

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

from neugk.utils import RunningMeanStd
from neugk.dataset.cyclone_kvikio import KvikioCycloneDataset, read_cupy_bin
from neugk.dataset.cyclone_diff import CycloneAESample


class KvikioCycloneAEDataset(KvikioCycloneDataset):
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
            sample = self._load_data(file_index, t_index, use_kvikio=False)

            # force to numpy for math
            if isinstance(sample["x"], torch.Tensor):
                sample["x"] = sample["x"].numpy()
            if sample.get("phi") is not None and isinstance(
                sample["phi"], torch.Tensor
            ):
                sample["phi"] = sample["phi"].numpy()
            if sample.get("flux") is not None and isinstance(
                sample["flux"], torch.Tensor
            ):
                sample["flux"] = sample["flux"].numpy()

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
                    stats.update(x_mean, x_var, x_min, x_max)

            pickle.dump(stats, open(stats_dump_pkl, "wb"))
            print(f"saved recomputed stats to {stats_dump_pkl}")
        return stats

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneAESample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if getattr(self, "precomputed_latents", None) is not None:
            sample = self.precomputed_latents[(file_index, t_index)]
            x = sample["x"]
        else:
            # load from raw bin files via kvikio
            sample = self._load_data(file_index, t_index, use_kvikio=self.load_with_kvikio)
            x = sample["x"]
            if x is not None and self.separate_zf:
                x = self._separate_zf(x)

        phi = sample["phi"]
        flux = sample["flux"]
        timestep = sample["timestep"]
        geometry = sample["geometry"]

        avg_flux = self.get_avg_flux(file_index)

        conditioning = None
        if self.conditions is not None and len(self.conditions) > 0:
            cond_list = []
            for k in self.conditions:
                val = sample[k]
                if isinstance(val, torch.Tensor):
                    cond_list.append(val.to(dtype=self.dtype))
                else:
                    cond_list.append(torch.tensor(val, dtype=self.dtype))
            conditioning = torch.stack(cond_list, dim=-1)

        if self.normalization is not None and get_normalized:
            if x is not None and getattr(self, "precomputed_latents", None) is None:
                x, _, _ = self.normalize(file_index, df=x)

            if phi is not None:
                phi, _, _ = self.normalize(file_index, phi=phi)

        if phi is not None and phi.ndim == 3:
            phi = (
                phi.unsqueeze(0)
                if isinstance(phi, torch.Tensor)
                else np.expand_dims(phi, 0)
            )

        x_out = (
            torch.tensor(x, dtype=self.dtype)
            if not isinstance(x, torch.Tensor) and x is not None
            else x
        )
        if x_out is not None:
            x_out = x_out.to(dtype=self.dtype)

        phi_out = (
            torch.tensor(phi, dtype=self.dtype)
            if not isinstance(phi, torch.Tensor) and phi is not None
            else phi
        )
        if phi_out is not None:
            phi_out = phi_out.to(dtype=self.dtype)

        return CycloneAESample(
            df=x_out,
            phi=phi_out,
            flux=torch.as_tensor(flux, dtype=self.dtype),
            avg_flux=torch.as_tensor(avg_flux, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            geometry=tree_map(lambda x: torch.as_tensor(x, dtype=self.dtype), geometry),
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
        )
        
        
    def _load_data(
        self, file_index: int, t_index: int, use_kvikio: bool = True
    ) -> dict:
        original_t_index = t_index + self.offsets[file_index]
        meta = self.metadata[file_index]
        data_dir = os.path.join(self.files[file_index], "data")

        xs, phis, fluxes = [], [], []

        for i in range(self.bundle_seq_length):
            t_str = str(original_t_index + i).zfill(5)

            kfile = os.path.join(data_dir, f"timestep_{t_str}.bin")
            phifile = os.path.join(data_dir, f"poten_{t_str}.bin")

            if "df" in self.input_fields:
                k = read_cupy_bin(
                    file=kfile,
                    shape=self.df_shape,
                    rank=self.rank,
                    use_kvikio=use_kvikio
                )

                if all(self.active_keys == np.array([0, 1])):
                    xs.append(k)
                else:
                    xs.append(k[self.active_keys])

            if "phi" in self.input_fields:
                phi = read_cupy_bin(
                    file=phifile,
                    shape=self.phi_resolution,
                    rank=self.rank,
                    use_kvikio=use_kvikio
                )

                phis.append(phi)

            flux = meta["fluxes"][original_t_index + self.bundle_seq_length + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.input_fields:
            # stack to shape (c, t, v1, v2, s, x, y)
            xs = xs[0] if self.bundle_seq_length == 1 else torch.stack(xs, 1)
        else:
            xs = None

        if "phi" in self.input_fields:
            phis = phis[0] if self.bundle_seq_length == 1 else torch.stack(phis, 1)
        else:
            phis = None

        sample["x"] = xs
        sample["phi"] = phis
        sample["flux"] = torch.tensor(fluxes).squeeze()

        sample["timestep"] = meta["timesteps"][original_t_index]
        sample["itg"] = meta["ion_temp_grad"]
        sample["dg"] = meta["density_grad"]
        sample["s_hat"] = meta["s_hat"]
        sample["q"] = meta["q"]
        sample["geometry"] = meta["geometry"]

        return sample

    def collate(self, batch: Sequence[CycloneAESample]):
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
        file_hash = hashlib.sha256("".join(sorted(self.files)).encode()).hexdigest()[:8]
        latents_dump_pkl = os.path.join(
            self.dir, f"diff_{self.split}_latents_{file_hash}.pkl"
        )

        if os.path.exists(latents_dump_pkl):
            if rank == 0:
                print(f"Loading precomputed latents from {latents_dump_pkl}")
            with open(latents_dump_pkl, "rb") as f:
                self.precomputed_latents = pickle.load(f)
            if dist.is_initialized():
                dist.barrier()
        else:
            # bypass pin_memory/worker issues for initial pass
            tmp_loader = DataLoader(
                dataset=dataloader.dataset,
                batch_size=32,
                num_workers=0,  # forced to 0 to prevent deadlocks with gds
                pin_memory=False,
                collate_fn=dataloader.collate_fn,
                drop_last=dataloader.drop_last,
                shuffle=False,
            )
            autoencoder.eval()
            autoencoder.to(device)
            latents_dict = {}
            desc = f"Precomputing {self.split} latents (rank:{rank})"

            for batch in tqdm(tmp_loader, desc=desc):
                # df is likely already on device if using kvikio, but enforce it
                df = batch.df.to(device)
                cond = (
                    batch.conditioning.to(device)
                    if hasattr(batch, "conditioning")
                    else None
                )
                z, _ = autoencoder.encode(df, condition=cond)
                z = z.cpu().numpy()

                for i in range(len(batch.file_index)):
                    f_idx = batch.file_index[i].item()
                    t_idx = batch.timestep_index[i].item()

                    # load raw sample data without kvikio to cache it in RAM
                    sample = self._load_data(f_idx, t_idx, use_kvikio=False)

                    # format to match expected dict structure
                    sample["x"] = z[i]
                    sample["flux"] = sample.pop("flux")  # rename to match logic

                    # convert torch/numpy fields cleanly
                    if isinstance(sample["phi"], torch.Tensor):
                        sample["phi"] = sample["phi"].numpy()
                    if isinstance(sample["flux"], torch.Tensor):
                        sample["flux"] = sample["flux"].numpy()

                    latents_dict[(f_idx, t_idx)] = sample

            if dist.is_initialized():
                gathered_dict = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_dict, latents_dict)
                full_latents_dict = {}
                for d in gathered_dict:
                    full_latents_dict.update(d)
                self.precomputed_latents = full_latents_dict
            else:
                self.precomputed_latents = latents_dict

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
            if dist.is_initialized():
                dist.barrier()
        else:
            assert latent_stats is not None
            self.latent_stats = latent_stats
