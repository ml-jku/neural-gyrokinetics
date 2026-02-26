from typing import Optional, Dict, Sequence, Any
import os
import pickle
from tqdm import tqdm
import hashlib
from dataclasses import dataclass

import torch
import numpy as np
import torch.distributed as dist
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

from neugk.utils import RunningMeanStd
from neugk.dataset.cyclone import CycloneDataset


@dataclass
class CycloneAESample:
    df: torch.Tensor
    phi: torch.Tensor
    flux: torch.Tensor
    avg_flux: torch.Tensor
    file_index: torch.Tensor
    timestep_index: torch.Tensor
    timestep: torch.Tensor
    conditioning: torch.Tensor
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
        filter_tag = (
            f"std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )
        return super()._recompute_stats(key, offset, prefix="diff", suffix=filter_tag)

    def __getitem__(self, index: int, get_normalized: bool = True) -> CycloneAESample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if getattr(self, "precomputed_latents", None) is not None:
            sample = self.precomputed_latents[(file_index, t_index)]
            x = sample["x"]
        else:
            with self.backend.open(self.files[file_index]) as f:
                sample = self._load_data(f, file_index, t_index)
            x = sample["x"]
            if x is not None and self.separate_zf:
                x = self._separate_zf(x)

        phi = sample["phi"]
        flux = sample["flux"]
        timestep = sample["timestep"]
        geom = sample["geometry"]

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
            if x is not None and self.precomputed_latents is None:
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
            geometry=tree_map(lambda g: torch.as_tensor(g, dtype=torch.float64), geom),
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
        )

    def _load_data(self, f: Any, file_index: int, t_index: int) -> Dict:
        original_t_index = t_index + self.offsets[file_index]
        meta = self.metadata[file_index]

        xs, phis, fluxes = [], [], []

        for i in range(self.bundle_seq_length):
            t_str = str(original_t_index + i).zfill(5)

            if "df" in self.input_fields:
                k = self.backend.read_df(f, t_str, self.df_shape, self.active_keys)
                xs.append(k)

            if "phi" in self.input_fields:
                phi = self.backend.read_phi(f, t_str, self.phi_resolution)
                phis.append(phi)

            flux = meta["fluxes"][original_t_index + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.input_fields:
            if self.bundle_seq_length == 1:
                xs = xs[0]
            else:
                xs = (
                    torch.stack(xs, axis=1)
                    if isinstance(xs[0], torch.Tensor)
                    else np.stack(xs, axis=1)
                )
        else:
            xs = None

        if "phi" in self.input_fields:
            if self.bundle_seq_length == 1:
                phis = phis[0]
            else:
                phis = (
                    torch.stack(phis, axis=1)
                    if isinstance(phis[0], torch.Tensor)
                    else np.stack(phis, axis=1)
                )
        else:
            phis = None

        sample["x"] = xs
        sample["phi"] = phis
        sample["flux"] = (
            torch.tensor(fluxes).squeeze()
            if isinstance(fluxes[0], torch.Tensor)
            else np.array(fluxes).squeeze()
        )

        sample["timestep"] = meta["timesteps"][original_t_index]
        sample["itg"] = meta["ion_temp_grad"].squeeze()
        sample["dg"] = meta["density_grad"].squeeze()
        sample["s_hat"] = meta["s_hat"].squeeze()
        sample["q"] = meta["q"].squeeze()
        sample["geometry"] = meta["geometry"]
        return sample

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

        def stack_batch(_b: Sequence[CycloneAESample], key: str):
            if getattr(_b[0], key, None) is not None:
                return torch.stack([getattr(sample, key) for sample in _b])
            return None

        return CycloneAESample(
            df=stack_batch(batch, "df"),
            phi=stack_batch(batch, "phi"),
            flux=stack_batch(batch, "flux"),
            avg_flux=stack_batch(batch, "avg_flux"),
            timestep=stack_batch(batch, "timestep"),
            file_index=stack_batch(batch, "file_index"),
            timestep_index=stack_batch(batch, "timestep_index"),
            conditioning=stack_batch(batch, "conditioning"),
            geometry=tree_map(
                lambda *x: torch.stack([torch.as_tensor(v) for v in x]),
                *[s.geometry for s in batch],
            ),
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
        ae_config_str = str(
            [(k, tuple(v.shape)) for k, v in autoencoder.state_dict().items()]
        )
        hash_str = "".join(sorted(self.files)) + ae_config_str
        file_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:12]
        latents_dump_pkl = os.path.join(
            self.dir, f"diff_{self.split}_latents_{file_hash}.pkl"
        )

        if os.path.exists(latents_dump_pkl):
            if rank == 0:
                print(f"loading precomputed latents from {latents_dump_pkl}")
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
            desc = f"precomputing {self.split} latents (rank:{rank})"

            for batch in tqdm(tmp_loader, desc=desc):
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

                    # load raw sample data directly
                    with self.backend.open(self.files[f_idx]) as f:
                        sample = self._load_data(f, f_idx, t_idx)

                    sample["x"] = z[i]

                    # convert torch/numpy fields cleanly
                    if isinstance(sample["phi"], torch.Tensor):
                        sample["phi"] = sample["phi"].cpu().numpy()
                    if isinstance(sample["flux"], torch.Tensor):
                        sample["flux"] = sample["flux"].cpu().numpy()

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
                print(f"saved precomputed latents to {latents_dump_pkl}")
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
                print(f"latent mean: {np.squeeze(stats.mean)}")
                print(f"latent var: {np.squeeze(stats.var)}")
                print(f"latent l2 norm: {np.squeeze(l2_norm)}")
            if dist.is_initialized():
                dist.barrier()
        else:
            assert latent_stats is not None
            self.latent_stats = latent_stats


@dataclass
class CycloneSimSiamSample(CycloneAESample):
    df_aug: torch.Tensor = None
    timestep_index_aug: torch.Tensor = None


# TODO(gg) did not test
class CycloneSimSiamDataset(CycloneAEDataset):
    def __getitem__(
        self, index: int, get_normalized: bool = True
    ) -> CycloneSimSiamSample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        with self.backend.open(self.files[file_index]) as f:
            sample = self._load_data(f, file_index, t_index)

        x, x_aug = sample["x"], sample["x_aug"]
        t_index_aug = sample["t_index_aug"]

        if x is not None and self.separate_zf:
            x = self._separate_zf(x)
            x_aug = self._separate_zf(x_aug)

        phi, flux = sample["phi"], sample["flux"]
        avg_flux = self.get_avg_flux(file_index)
        timestep = sample["timestep"]
        geom = sample["geometry"]

        cond_list = []
        for k in self.conditions:
            val = sample[k]
            if isinstance(val, torch.Tensor):
                cond_list.append(val.to(dtype=self.dtype))
            else:
                cond_list.append(torch.tensor(val, dtype=self.dtype))
        conditioning = torch.stack(cond_list, dim=-1)

        if self.normalization is not None and get_normalized:
            if x is not None:
                x, _, _ = self.normalize(file_index, df=x)
                x_aug, _, _ = self.normalize(file_index, df=x_aug)
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

        x_aug_out = (
            torch.tensor(x_aug, dtype=self.dtype)
            if not isinstance(x_aug, torch.Tensor) and x_aug is not None
            else x_aug
        )
        if x_aug_out is not None:
            x_aug_out = x_aug_out.to(dtype=self.dtype)

        phi_out = (
            torch.tensor(phi, dtype=self.dtype)
            if not isinstance(phi, torch.Tensor) and phi is not None
            else phi
        )
        if phi_out is not None:
            phi_out = phi_out.to(dtype=self.dtype)

        return CycloneSimSiamSample(
            df=x_out,
            df_aug=x_aug_out,
            phi=phi_out,
            flux=torch.as_tensor(flux, dtype=self.dtype),
            avg_flux=torch.as_tensor(avg_flux, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            timestep_index_aug=torch.tensor(t_index_aug, dtype=torch.long),
            geometry=tree_map(lambda g: torch.as_tensor(g, dtype=torch.float64), geom),
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
        )

    def _load_data(self, f, file_index, t_index) -> dict:
        offset, n_ts = self.offsets[file_index], self.file_num_timesteps[file_index]
        orig_t_index = t_index + offset
        meta = self.metadata[file_index]

        xs, xs2, phis, fluxes = [], [], [], []
        t_aug = []

        for i in range(self.bundle_seq_length):
            t_str = str(orig_t_index + i).zfill(5)

            while (rnd := np.random.randint(offset, n_ts + offset)) == orig_t_index + i:
                pass

            t_str_aug = str(rnd).zfill(5)
            t_aug.append(rnd)

            if "df" in self.input_fields:
                k = self.backend.read_df(
                    f, t_str, self.df_shape, self.active_keys, self.rank
                )
                k2 = self.backend.read_df(
                    f, t_str_aug, self.df_shape, self.active_keys, self.rank
                )
                xs.append(k)
                xs2.append(k2)

            if "phi" in self.input_fields:
                phi = self.backend.read_phi(f, t_str, self.phi_resolution, self.rank)
                phis.append(phi)

            flux = meta["fluxes"][orig_t_index + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.input_fields:
            if self.bundle_seq_length == 1:
                xs, xs2 = xs[0], xs2[0]
            else:
                xs = (
                    torch.stack(xs, axis=1)
                    if isinstance(xs[0], torch.Tensor)
                    else np.stack(xs, axis=1)
                )
                xs2 = (
                    torch.stack(xs2, axis=1)
                    if isinstance(xs2[0], torch.Tensor)
                    else np.stack(xs2, axis=1)
                )
        else:
            xs = xs2 = None

        if "phi" in self.input_fields:
            if self.bundle_seq_length == 1:
                phis = phis[0]
            else:
                phis = (
                    torch.stack(phis, axis=1)
                    if isinstance(phis[0], torch.Tensor)
                    else np.stack(phis, axis=1)
                )
        else:
            phis = None

        t_aug = t_aug[0] if self.bundle_seq_length == 1 else np.stack(t_aug, axis=1)

        sample["x"] = xs
        sample["x_aug"] = xs2
        sample["phi"] = phis
        sample["flux"] = (
            torch.tensor(fluxes).squeeze()
            if isinstance(fluxes[0], torch.Tensor)
            else np.array(fluxes).squeeze()
        )

        sample["timestep"] = meta["timesteps"][orig_t_index]
        sample["t_index_aug"] = t_aug
        sample["itg"] = meta["ion_temp_grad"].squeeze()
        sample["dg"] = meta["density_grad"].squeeze()
        sample["s_hat"] = meta["s_hat"].squeeze()
        sample["q"] = meta["q"].squeeze()
        sample["geometry"] = meta["geometry"]

        return sample

    def collate(self, batch: Sequence[CycloneSimSiamSample]):

        def stack_batch(_b: Sequence[CycloneSimSiamSample], key: str):
            if hasattr(_b[0], key) is not None:
                return torch.stack([getattr(sample, key) for sample in _b])
            return None

        return CycloneSimSiamSample(
            df=stack_batch(batch, "df"),
            df_aug=stack_batch(batch, "df_aug"),
            phi=stack_batch(batch, "phi"),
            flux=stack_batch(batch, "flux"),
            avg_flux=stack_batch(batch, "avg_flux"),
            timestep=stack_batch(batch, "timestep"),
            file_index=stack_batch(batch, "file_index"),
            timestep_index=stack_batch(batch, "timestep_index"),
            conditioning=stack_batch(batch, "conditioning"),
            geometry=tree_map(
                lambda *x: torch.stack([torch.as_tensor(v) for v in x]),
                *[s.geometry for s in batch],
            ),
        )
