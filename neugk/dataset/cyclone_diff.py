from typing import Optional, Dict, Sequence, Any, Union, List

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

from neugk.utils import RunningMeanStd, separate_zf as separate_zf_fn
from neugk.dataset.cyclone import CycloneDataset, resolve_trajectories
from neugk.dataset.backend import KvikIOBackend


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
    # optional served GT turbulence spectra (per-timestep, per-mode)
    kyspec: Optional[torch.Tensor] = None
    fluxspec: Optional[torch.Tensor] = None

    def pin_memory(self):
        if self.df is not None:
            self.df = self.df.pin_memory()
        if self.phi is not None:
            self.phi = self.phi.pin_memory()
        return self


_VALID_LATENT_SCALING_MODES = ("global", "per_channel", "per_token")


def _latent_norm_axes(mode: str, ndim: int) -> tuple:
    """Reduction axes used to derive the latent_scale stats from one sample.

    Per-sample latents have no batch axis (shape `(C, *spatial)`).
        - global       -> reduce over all dims  -> stats shape (1, 1, ...)  scalar broadcast
        - per_channel  -> keep channel axis 0   -> stats shape (C, 1, ...)
        - per_token    -> reduce nothing        -> stats shape (C, *spatial) (per-element)
    """
    if mode == "global":
        return tuple(range(0, ndim))
    if mode == "per_channel":
        return tuple(range(1, ndim))
    if mode == "per_token":
        return ()
    raise ValueError(
        f"latent_scaling_mode must be one of {_VALID_LATENT_SCALING_MODES}, got '{mode}'"
    )


class CycloneAEDataset(CycloneDataset):
    def __init__(
        self,
        *args,
        conditions: Sequence[str],
        precomputed_latents: Optional[Dict] = None,
        autoencoder: Optional[torch.nn.Module] = None,
        ae_cfg=None,
        latent_scaling_mode: str = "global",
        **kwargs,
    ):
        if latent_scaling_mode not in _VALID_LATENT_SCALING_MODES:
            raise ValueError(
                f"latent_scaling_mode must be one of {_VALID_LATENT_SCALING_MODES}, "
                f"got '{latent_scaling_mode}'"
            )
        self.conditions = conditions
        self.precomputed_latents = precomputed_latents
        self.autoencoder = autoencoder
        self._ae_cfg = ae_cfg
        self.latent_scaling_mode = latent_scaling_mode
        super().__init__(*args, **kwargs)

    def _recompute_stats(
        self, keys: Union[str, List[str]], offset: int = 0
    ) -> Dict[str, RunningMeanStd]:
        filter_tag = (
            f"std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )

        if self._ae_cfg is None:
            return super()._recompute_stats(
                keys, offset, prefix="diff", suffix=filter_tag
            )

        # diffusion dataset may use different training_trajectories than the AE
        ae_ds = self._ae_cfg.dataset
        ae_offset = getattr(ae_ds, "offset", offset)
        ae_decouple_mu = getattr(ae_ds, "norm_decouple_mu", self.decouple_mu)
        ae_filter_tag = (
            f"std{ae_ds.timestep_std_filter}"
            if getattr(ae_ds, "timestep_std_filter", None)
            else ""
        )

        # build AE file list from config
        raw_ae_files = resolve_trajectories(self.dir, ae_ds.training_trajectories)
        ae_files = sorted(
            set(
                self.backend.format_path(
                    f,
                    ae_ds.spatial_ifft,
                    getattr(ae_ds, "split_into_bands", None),
                    getattr(ae_ds, "real_potens", True),
                )
                for f in raw_ae_files
            )
        )

        # try agg cache with unfiltered files first (avoids NFS stat calls)
        _unfiltered_hash = hashlib.sha256(
            "".join(sorted(os.path.basename(f) for f in ae_files)).encode()
        ).hexdigest()[:8]
        _keys_list = sorted(keys if isinstance(keys, (list, tuple)) else [keys])
        _keys_tag = "_".join(_keys_list)
        _tmu = "mu" if ae_decouple_mu else ""
        _norm_tag = "_".join(
            f"{k}{''.join(str(a) for a in self.normalizers[k]['agg_axes'])}"
            for k in _keys_list
            if self.normalizers[k]["agg_axes"]
        )
        for _h in [_unfiltered_hash]:
            _segs = [
                "diff",
                _keys_tag,
                f"offset{ae_offset}",
                _tmu,
                ae_filter_tag,
                _h,
                _norm_tag,
                "agg_stats",
            ]
            _agg_path = os.path.join(
                self.dir, "_".join(filter(None, (str(s) for s in _segs))) + ".pkl"
            )
            if os.path.exists(_agg_path):
                print(f"loading aggregated stats from {_agg_path}")
                with open(_agg_path, "rb") as f:
                    return pickle.load(f)

        # validate files (slow on NFS, but only if no cache hit)
        ae_files = [f for f in ae_files if self.backend.is_valid(f)]
        ae_cond_filters = getattr(ae_ds, "training_cond_filters", None)
        if ae_cond_filters:
            # if diffusion files == AE files (already filtered), reuse them directly
            ae_basenames = {os.path.basename(f) for f in ae_files}
            self_basenames = {os.path.basename(f) for f in self.files}
            if self_basenames == ae_basenames:
                ae_files = self.files
            else:
                ae_threshold = ae_offset if ae_offset > 0 else 80
                orig_cond_filters = self.cond_filters
                self.cond_filters = ae_cond_filters
                ae_files = [
                    f for f in ae_files if self._conditioning_filter(f, ae_threshold)
                ]
                self.cond_filters = orig_cond_filters

        # build expected stats filename - use basenames
        file_basenames = sorted(os.path.basename(f) for f in ae_files)
        file_hash = hashlib.sha256("".join(file_basenames).encode()).hexdigest()[:8]
        keys_tag = "_".join(sorted(keys if isinstance(keys, (list, tuple)) else [keys]))
        tmu = "mu" if ae_decouple_mu else ""
        norm_tag = "_".join(
            f"{k}{''.join(str(a) for a in self.normalizers[k]['agg_axes'])}"
            for k in sorted(keys if isinstance(keys, (list, tuple)) else [keys])
            if self.normalizers[k]["agg_axes"]
        )
        segments = [
            "diff",
            keys_tag,
            f"offset{ae_offset}",
            tmu,
            ae_filter_tag,
            file_hash,
            "stats",
        ]
        stats_filename = "_".join(filter(None, (str(s) for s in segments))) + ".pkl"
        stats_path = os.path.join(self.dir, stats_filename)
        self.raw_stats_path = stats_path

        # fast path: load small aggregated cache (avoids loading 681MB+ raw stats pkl)
        agg_segments = segments[:-1] + [norm_tag, "agg_stats"]
        agg_filename = "_".join(filter(None, (str(s) for s in agg_segments))) + ".pkl"
        agg_path = os.path.join(self.dir, agg_filename)

        if os.path.exists(agg_path):
            print(f"loading aggregated stats from {agg_path}")
            with open(agg_path, "rb") as f:
                return pickle.load(f)

        if os.path.exists(stats_path):
            # raw stats exist but no aggregated cache yet — load, aggregate, save small cache
            print(f"loading raw stats from {stats_path} (building aggregated cache...)")
            with open(stats_path, "rb") as f:
                stats_dict = pickle.load(f)
            for key in (keys if isinstance(keys, (list, tuple)) else [keys]):
                stats = stats_dict[key]
                if self.normalizers[key]["agg_axes"]:
                    norm_axes = tuple(self.normalizers[key]["agg_axes"])
                    mean, var, traj_min, traj_max = stats.aggregate_stats(
                        stats.mean, stats.var, stats.min, stats.max, agg_axes=norm_axes
                    )
                    if key == "phi":
                        mean = np.expand_dims(mean, axis=0)
                        var = np.expand_dims(var, axis=0)
                        traj_min = np.expand_dims(traj_min, axis=0)
                        traj_max = np.expand_dims(traj_max, axis=0)
                    stats.mean = mean
                    stats.var = var
                    stats.min = traj_min
                    stats.max = traj_max
            with open(agg_path, "wb") as f:
                pickle.dump(stats_dict, f)
            print(f"saved aggregated stats to {agg_path}")
            return stats_dict

        # swap self state so the parent uses AE files/config for the hash + pkl
        saved = (
            self.files,
            self.decouple_mu,
            self.flat_index_to_file_and_tstep,
            self.length,
            self.offsets,
            self.metadata,
        )
        self.files = ae_files
        self.decouple_mu = ae_decouple_mu

        # build full index + metadata
        input_keys = list(keys) if isinstance(keys, (list, tuple)) else [keys]
        ae_metadata = {}
        ae_flat_index = {}
        flat_idx = 0
        for file_idx, ae_file in enumerate(ae_files):
            meta = self.backend.read_metadata(ae_file, input_fields=input_keys)
            ae_metadata[file_idx] = meta
            n_t = len(meta.get("timesteps", []))
            for t_idx in range(max(0, n_t - ae_offset)):
                ae_flat_index[flat_idx] = (file_idx, t_idx)
                flat_idx += 1
        self.flat_index_to_file_and_tstep = ae_flat_index
        self.length = flat_idx
        self.offsets = [ae_offset] * len(ae_files)
        self.metadata = ae_metadata

        try:
            result = super()._recompute_stats(
                keys, ae_offset, prefix="diff", suffix=ae_filter_tag
            )
        finally:
            (
                self.files,
                self.decouple_mu,
                self.flat_index_to_file_and_tstep,
                self.length,
                self.offsets,
                self.metadata,
            ) = saved

        return result

    def __getitem__(
        self, index: int, get_normalized: bool = True, override_latens: bool = False
    ) -> CycloneAESample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if (
            getattr(self, "precomputed_latents", None) is not None
            and not override_latens
        ):
            sample = self.precomputed_latents[(file_index, t_index)]
            # fast path: pre-tensorized latents (after _tensorize_latents)
            if "_sample" in sample:
                return sample["_sample"]
            x = sample["x"]
        else:
            with self.backend.open(self.files[file_index]) as f:
                sample = self._load_data(f, file_index, t_index)
            x = sample["x"]
            if x is not None and self.separate_zf:
                x = separate_zf_fn(x, dim=0)

        phi = sample["phi"]
        flux = sample["flux"]
        timestep = sample["timestep"]

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

        if get_normalized:
            # skip normalization if latents are precomputed (unless overriding)
            if x is not None and (self.precomputed_latents is None or override_latens):
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
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
            kyspec=(
                torch.as_tensor(sample["kyspec"], dtype=self.dtype)
                if "kyspec" in sample
                else None
            ),
            fluxspec=(
                torch.as_tensor(sample["fluxspec"], dtype=self.dtype)
                if "fluxspec" in sample
                else None
            ),
        )

    def _load_data(self, f: Any, file_index: int, t_index: int) -> Dict:
        original_t_index = t_index + self.offsets[file_index]
        meta = self.metadata[file_index]

        xs, phis, fluxes = [], [], []

        for i in range(self.bundle_seq_length):
            t_str = str(original_t_index + i).zfill(5)

            if "df" in self.fields_to_load:
                k = self.backend.read_df(f, t_str, self.df_shape, self.active_keys)
                xs.append(k)

            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(f, t_str, self.phi_resolution)
                phis.append(phi)

            flux = meta["flux"][original_t_index + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.fields_to_load:
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

        if "phi" in self.fields_to_load:
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
        # served GT spectra at the (input == reconstructed) timestep, gated by
        # fields_to_load so default behaviour (df/phi/flux only) is unchanged.
        for skey in self.SPECTRAL_KEYS:
            if skey in self.fields_to_load and skey in meta:
                sample[skey] = self.get_spectrum(file_index, original_t_index, skey)
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
                # TODO naive, can improve: pad axes can be precomputed from resolution
                condition = kwargs["condition"]
                # handle single sample input from evaluate.py loops
                if df.ndim == 5:  # (C, Vp, Vm, S, X, Y) -> (1, C, ...)
                    df = df.unsqueeze(0)
                if condition.ndim == 1:  # (N_cond,) -> (1, N_cond)
                    condition = condition.unsqueeze(0)

                df = self.autoencoder.decode(df, condition=condition)["df"]
                df = df.squeeze(0)
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
            kyspec=stack_batch(batch, "kyspec"),
            fluxspec=stack_batch(batch, "fluxspec"),
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

        config_keys = ["cond_filters", "subsample", "separate_zf"]
        config_str = "".join(str(getattr(self, k, "")) for k in config_keys)
        model_str = str({k: v.shape for k, v in autoencoder.state_dict().items()})
        ae_checkpoint_path = str(
            getattr(autoencoder, "checkpoint_path", "")
            or getattr(autoencoder, "_checkpoint_path", "")
        )

        file_basenames = sorted(os.path.basename(f) for f in self.files)
        file_hash = hashlib.sha256("".join(file_basenames).encode()).hexdigest()[:12]

        tmu = "mu" if self.decouple_mu else ""
        offset = self.offsets[0]
        filter_tag = (
            f"std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )
        ae_tag = "ae" + ae_checkpoint_path.split("_")[-1] if ae_checkpoint_path else ""

        segments = [
            "diff",
            f"{self.split}_latents",
            f"offset{offset}",
            tmu,
            filter_tag,
            file_hash,
            "latents",
            ae_tag,
        ]
        latents_dump_pkl = os.path.join(
            self.dir, "_".join(filter(None, (str(s) for s in segments))) + ".pkl"
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
            # tmp_loader = DataLoader(
            #     dataset=dataloader.dataset,
            #     batch_size=128,
            #     num_workers=dataloader.num_workers,
            #     prefetch_factor=1,
            #     pin_memory=False,
            #     collate_fn=dataloader.collate_fn,
            #     drop_last=dataloader.drop_last,
            #     shuffle=False,
            # )
            tmp_loader = dataloader
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

                    sample = {"x": z[i]}

                    # extract from batch instead of re-reading from disk
                    if batch.phi is not None:
                        phi_i = batch.phi[i]
                        sample["phi"] = (
                            phi_i.cpu().numpy()
                            if isinstance(phi_i, torch.Tensor)
                            else np.asarray(phi_i)
                        )
                    else:
                        sample["phi"] = None

                    flux_i = batch.flux[i]
                    sample["flux"] = (
                        flux_i.cpu().numpy()
                        if isinstance(flux_i, torch.Tensor)
                        else np.asarray(flux_i)
                    )

                    ts_i = batch.timestep[i]
                    sample["timestep"] = (
                        ts_i.cpu().numpy()
                        if isinstance(ts_i, torch.Tensor)
                        else np.asarray(ts_i)
                    )

                    if batch.conditioning is not None:
                        cond_i = batch.conditioning[i]
                        if isinstance(cond_i, torch.Tensor):
                            cond_i = cond_i.cpu().numpy()
                        for j, key in enumerate(self.conditions):
                            sample[key] = cond_i[j]

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
                norm_axes = _latent_norm_axes(self.latent_scaling_mode, x.ndim)
                if norm_axes:
                    x_mean = np.mean(x, axis=norm_axes, keepdims=True)
                    x_var = np.var(x, axis=norm_axes, keepdims=True)
                    x_min = np.min(x, axis=norm_axes, keepdims=True)
                    x_max = np.max(x, axis=norm_axes, keepdims=True)
                    l2_norms.append(
                        np.sqrt(np.sum(x**2, axis=norm_axes, keepdims=True))
                    )
                else:
                    # per_token: per-element stats; one sample contributes (mean=x, var=0)
                    x_f = x.astype(np.float32, copy=False)
                    x_mean = x_f.copy()
                    x_var = np.zeros_like(x_f)
                    x_min = x_f.copy()
                    x_max = x_f.copy()
                    l2_norms.append(np.abs(x_f))
                if stats is None:
                    stats = RunningMeanStd(shape=x_mean.shape)
                stats.update(x_mean, x_var, x_min, x_max)
            self.latent_stats = stats
            l2_norm = np.mean(l2_norms, axis=0)
            if rank == 0:
                print(f"latent_scaling_mode: {self.latent_scaling_mode}")
                print(f"latent stats shape: {stats.mean.shape}")
                print(f"latent mean: {np.squeeze(stats.mean)}")
                print(f"latent var: {np.squeeze(stats.var)}")
                print(f"latent l2 norm: {np.squeeze(l2_norm)}")
            if dist.is_initialized():
                dist.barrier()
        else:
            assert latent_stats is not None
            self.latent_stats = latent_stats

        # update backend to not use kvikio, not needed for diffusion beyond this point
        if isinstance(self.backend, KvikIOBackend):
            self.backend = KvikIOBackend(self.rank, use_kvikio=False)

        # pre-tensorize latents and cache avg_flux for fast __getitem__
        self._tensorize_latents()

    def _tensorize_latents(self):
        """Pre-build CycloneAESample objects for zero-alloc __getitem__."""
        if self.precomputed_latents is None:
            return
        # cache avg_flux per file
        avg_flux_cache = {}
        for f_id in self.metadata:
            fluxes = self.metadata[f_id]["flux"]
            avg_flux_cache[f_id] = torch.tensor(
                float(np.mean(fluxes[-80:])), dtype=self.dtype
            )

        for (file_index, t_index), sample in self.precomputed_latents.items():
            x = sample["x"]
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=self.dtype)
            flux = sample["flux"]
            if isinstance(flux, np.ndarray):
                flux = torch.as_tensor(flux, dtype=self.dtype)
            elif not isinstance(flux, torch.Tensor):
                flux = torch.tensor(flux, dtype=self.dtype)
            timestep = sample["timestep"]
            if isinstance(timestep, np.ndarray):
                timestep = torch.as_tensor(timestep, dtype=self.dtype)
            elif not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor(timestep, dtype=self.dtype)
            conditioning = None
            if self.conditions:
                cond_vals = []
                for k in self.conditions:
                    val = sample[k]
                    if isinstance(val, torch.Tensor):
                        cond_vals.append(val.to(dtype=self.dtype))
                    else:
                        cond_vals.append(torch.tensor(val, dtype=self.dtype))
                conditioning = torch.stack(cond_vals, dim=-1)

            sample["_sample"] = CycloneAESample(
                df=x,
                phi=None,  # not used in diffusion training
                flux=flux,
                avg_flux=avg_flux_cache.get(file_index, torch.tensor(0.0)),
                file_index=torch.tensor(file_index, dtype=torch.long),
                timestep_index=torch.tensor(t_index, dtype=torch.long),
                timestep=timestep,
                conditioning=conditioning,
            )


class CycloneVAEDataset(CycloneAEDataset):
    def __init__(
        self,
        *args,
        latent_sampling_mode: str = "stochastic",
        **kwargs,
    ):
        self.latent_sampling_mode = latent_sampling_mode.lower()
        if self.latent_sampling_mode not in {"stochastic", "deterministic"}:
            raise ValueError(
                "latent_sampling_mode must be either 'stochastic' or 'deterministic'."
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _sample_from_mu_var(
        mu: Union[torch.Tensor, np.ndarray],
        var: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        mu_np = (
            mu.detach().cpu().numpy()
            if isinstance(mu, torch.Tensor)
            else np.asarray(mu)
        )
        var_np = (
            var.detach().cpu().numpy()
            if isinstance(var, torch.Tensor)
            else np.asarray(var)
        )
        var_np = np.clip(var_np, 1e-12, None)
        return mu_np + np.sqrt(var_np) * np.random.randn(*mu_np.shape)

    def __getitem__(
        self, index: int, get_normalized: bool = True, override_latens: bool = False
    ) -> CycloneAESample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if (
            getattr(self, "precomputed_latents", None) is not None
            and not override_latens
        ):
            sample = self.precomputed_latents[(file_index, t_index)]
            if "mu" in sample and "var" in sample:
                if self.latent_sampling_mode == "stochastic":
                    x = self._sample_from_mu_var(sample["mu"], sample["var"])
                else:
                    x = (
                        sample["mu"].detach().cpu().numpy()
                        if isinstance(sample["mu"], torch.Tensor)
                        else np.asarray(sample["mu"])
                    )
            else:
                x = sample["x"]
        else:
            with self.backend.open(self.files[file_index]) as f:
                sample = self._load_data(f, file_index, t_index)
            x = sample["x"]
            if x is not None and self.separate_zf:
                x = separate_zf_fn(x, dim=0)

        phi = sample["phi"]
        flux = sample["flux"]
        timestep = sample["timestep"]

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

        if get_normalized:
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
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
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

        config_keys = ["cond_filters", "subsample", "separate_zf"]
        config_str = "".join(str(getattr(self, k, "")) for k in config_keys)
        model_str = str({k: v.shape for k, v in autoencoder.state_dict().items()})
        vae_checkpoint_path = str(
            getattr(autoencoder, "checkpoint_path", "")
            or getattr(autoencoder, "_checkpoint_path", "")
        )
        hash_str = (
            "".join(sorted(self.files))
            + config_str
            + model_str
            + vae_checkpoint_path
            + "vae"
        )
        file_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:12]

        tmu = "mu" if self.decouple_mu else ""
        offset = self.offsets[0]
        filter_tag = (
            f"std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )
        vae_tag = (
            "vae" + vae_checkpoint_path.split("_")[-1] if vae_checkpoint_path else ""
        )

        segments = [
            "diff",
            f"{self.split}_latents",
            f"offset{offset}",
            tmu,
            filter_tag,
            file_hash,
            "latents",
            vae_tag,
        ]
        latents_dump_pkl = os.path.join(
            self.dir, "_".join(filter(None, (str(s) for s in segments))) + ".pkl"
        )

        if os.path.exists(latents_dump_pkl):
            if rank == 0:
                print(f"loading precomputed latents from {latents_dump_pkl}")
            with open(latents_dump_pkl, "rb") as f:
                loaded = pickle.load(f)
            self.precomputed_latents = (
                loaded["samples"]
                if isinstance(loaded, dict) and "samples" in loaded
                else loaded
            )
            if dist.is_initialized():
                dist.barrier()
        else:
            tmp_loader = dataloader
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

                mu = getattr(autoencoder, "_mu", None)
                logvar = getattr(autoencoder, "_logvar", None)
                if mu is None or logvar is None:
                    raise RuntimeError(
                        "VAE latent precompute expects encoder to expose _mu/_logvar."
                    )
                mu_np = mu.detach().cpu().numpy()
                var_np = np.exp(logvar.detach().cpu().numpy())

                for i in range(len(batch.file_index)):
                    f_idx = batch.file_index[i].item()
                    t_idx = batch.timestep_index[i].item()

                    sample = {"x": z[i], "mu": mu_np[i], "var": var_np[i]}

                    if batch.phi is not None:
                        phi_i = batch.phi[i]
                        sample["phi"] = (
                            phi_i.cpu().numpy()
                            if isinstance(phi_i, torch.Tensor)
                            else np.asarray(phi_i)
                        )
                    else:
                        sample["phi"] = None

                    flux_i = batch.flux[i]
                    sample["flux"] = (
                        flux_i.cpu().numpy()
                        if isinstance(flux_i, torch.Tensor)
                        else np.asarray(flux_i)
                    )

                    ts_i = batch.timestep[i]
                    sample["timestep"] = (
                        ts_i.cpu().numpy()
                        if isinstance(ts_i, torch.Tensor)
                        else np.asarray(ts_i)
                    )

                    if batch.conditioning is not None:
                        cond_i = batch.conditioning[i]
                        if isinstance(cond_i, torch.Tensor):
                            cond_i = cond_i.cpu().numpy()
                        for j, key in enumerate(self.conditions):
                            sample[key] = cond_i[j]

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
                mu = (
                    sample["mu"].detach().cpu().numpy()
                    if isinstance(sample["mu"], torch.Tensor)
                    else np.asarray(sample["mu"])
                )
                var = (
                    sample["var"].detach().cpu().numpy()
                    if isinstance(sample["var"], torch.Tensor)
                    else np.asarray(sample["var"])
                )
                var = np.clip(var, 1e-12, None)
                norm_axes = _latent_norm_axes(self.latent_scaling_mode, mu.ndim)

                if self.latent_sampling_mode == "stochastic":
                    if norm_axes:
                        x_mean = np.mean(mu, axis=norm_axes, keepdims=True)
                        mu2_mean = np.mean(mu**2, axis=norm_axes, keepdims=True)
                        var_mean = np.mean(var, axis=norm_axes, keepdims=True)
                        x_var = var_mean + mu2_mean - x_mean**2
                        std = np.sqrt(var)
                        x_min = np.min(mu - 3.0 * std, axis=norm_axes, keepdims=True)
                        x_max = np.max(mu + 3.0 * std, axis=norm_axes, keepdims=True)
                        l2_norms.append(
                            np.sqrt(np.sum(mu**2 + var, axis=norm_axes, keepdims=True))
                        )
                    else:
                        # per_token: x = mu + eps*sqrt(var) -> per-element E[x]=mu, Var[x]=var
                        x_mean = mu.astype(np.float32, copy=False).copy()
                        x_var = var.astype(np.float32, copy=False).copy()
                        std = np.sqrt(var)
                        x_min = (mu - 3.0 * std).astype(np.float32, copy=False)
                        x_max = (mu + 3.0 * std).astype(np.float32, copy=False)
                        l2_norms.append(
                            np.sqrt(mu**2 + var).astype(np.float32, copy=False)
                        )
                else:
                    if norm_axes:
                        x_mean = np.mean(mu, axis=norm_axes, keepdims=True)
                        x_var = np.var(mu, axis=norm_axes, keepdims=True)
                        x_min = np.min(mu, axis=norm_axes, keepdims=True)
                        x_max = np.max(mu, axis=norm_axes, keepdims=True)
                        l2_norms.append(
                            np.sqrt(np.sum(mu**2, axis=norm_axes, keepdims=True))
                        )
                    else:
                        # per_token deterministic: one sample per element -> mean=mu, var=0
                        mu_f = mu.astype(np.float32, copy=False)
                        x_mean = mu_f.copy()
                        x_var = np.zeros_like(mu_f)
                        x_min = mu_f.copy()
                        x_max = mu_f.copy()
                        l2_norms.append(np.abs(mu_f))

                if stats is None:
                    stats = RunningMeanStd(shape=x_mean.shape)
                stats.update(x_mean, x_var, x_min, x_max)

            self.latent_stats = stats
            l2_norm = np.mean(l2_norms, axis=0)
            if rank == 0:
                print(f"latent_scaling_mode: {self.latent_scaling_mode}")
                print(f"latent stats shape: {stats.mean.shape}")
                print(f"latent mean: {np.squeeze(stats.mean)}")
                print(f"latent var: {np.squeeze(stats.var)}")
                print(f"latent l2 norm: {np.squeeze(l2_norm)}")
            if dist.is_initialized():
                dist.barrier()
        else:
            assert latent_stats is not None
            self.latent_stats = latent_stats

        if isinstance(self.backend, KvikIOBackend):
            self.backend = KvikIOBackend(self.rank, use_kvikio=False)


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
            x = separate_zf_fn(x, dim=0)
            x_aug = separate_zf_fn(x_aug, dim=0)

        phi, flux = sample["phi"], sample["flux"]
        avg_flux = self.get_avg_flux(file_index)
        timestep = sample["timestep"]

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

        if get_normalized:
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

            if "df" in self.fields_to_load:
                k = self.backend.read_df(f, t_str, self.df_shape, self.active_keys)
                k2 = self.backend.read_df(f, t_str_aug, self.df_shape, self.active_keys)
                xs.append(k)
                xs2.append(k2)

            if "phi" in self.fields_to_load:
                phi = self.backend.read_phi(f, t_str, self.phi_resolution)
                phis.append(phi)

            flux = meta["flux"][orig_t_index + i]
            fluxes.append(flux)

        sample = {}
        if "df" in self.fields_to_load:
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

        if "phi" in self.fields_to_load:
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

        return sample

    def collate(self, batch: Sequence[CycloneSimSiamSample]):
        def stack_batch(_b: Sequence[CycloneSimSiamSample], key: str):
            if getattr(_b[0], key, None) is not None:
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
        )


class CycloneVQVAEDataset(CycloneAEDataset):
    """Dataset that precomputes and serves discrete VQVAE token indices."""

    def __getitem__(
        self, index: int, get_normalized: bool = True, override_latens: bool = False
    ) -> CycloneAESample:
        file_index, t_index = self.flat_index_to_file_and_tstep[index]

        if (
            getattr(self, "precomputed_latents", None) is not None
            and not override_latens
        ):
            sample = self.precomputed_latents[(file_index, t_index)]
            if "_sample" in sample:
                return sample["_sample"]
            x = sample["x"]  # already flattened int64 indices
        else:
            with self.backend.open(self.files[file_index]) as f:
                sample = self._load_data(f, file_index, t_index)
            x = sample["x"]
            if x is not None and self.separate_zf:
                x = separate_zf_fn(x, dim=0)

        phi = sample.get("phi")
        flux = sample["flux"]
        timestep = sample["timestep"]
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

        # for precomputed indices: x is already int64, skip normalization
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x is not None and x.dtype not in (torch.long, torch.int64, torch.int32):
            # not precomputed — normalize like parent
            if get_normalized and (self.precomputed_latents is None or override_latens):
                x, _, _ = self.normalize(file_index, df=x)
            x = torch.as_tensor(x, dtype=self.dtype)
        else:
            x = x.long() if x is not None else x

        phi_out = None
        if phi is not None:
            if get_normalized:
                phi, _, _ = self.normalize(file_index, phi=phi)
            if phi.ndim == 3:
                phi = (
                    phi.unsqueeze(0)
                    if isinstance(phi, torch.Tensor)
                    else np.expand_dims(phi, 0)
                )
            phi_out = (
                torch.tensor(phi, dtype=self.dtype)
                if not isinstance(phi, torch.Tensor)
                else phi.to(dtype=self.dtype)
            )

        return CycloneAESample(
            df=x,
            phi=phi_out,
            flux=torch.as_tensor(flux, dtype=self.dtype),
            avg_flux=torch.as_tensor(avg_flux, dtype=self.dtype),
            file_index=torch.tensor(file_index, dtype=torch.long),
            timestep_index=torch.tensor(t_index, dtype=torch.long),
            timestep=torch.as_tensor(timestep, dtype=self.dtype),
            conditioning=conditioning,
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

        ae_checkpoint_path = str(
            getattr(autoencoder, "checkpoint_path", "")
            or getattr(autoencoder, "_checkpoint_path", "")
        )

        file_basenames = sorted(os.path.basename(f) for f in self.files)
        file_hash = hashlib.sha256("".join(file_basenames).encode()).hexdigest()[:12]

        tmu = "mu" if self.decouple_mu else ""
        offset = self.offsets[0]
        filter_tag = (
            f"std{self.timestep_std_filter}" if self.timestep_std_filter else ""
        )
        vqvae_tag = (
            "vqvae" + ae_checkpoint_path.split("_")[-1] if ae_checkpoint_path else ""
        )

        segments = [
            "diff",
            f"{self.split}_indices",
            f"offset{offset}",
            tmu,
            filter_tag,
            file_hash,
            "indices",
            vqvae_tag,
        ]
        indices_dump_pkl = os.path.join(
            self.dir, "_".join(filter(None, (str(s) for s in segments))) + ".pkl"
        )

        if os.path.exists(indices_dump_pkl):
            if rank == 0:
                print(f"loading precomputed VQ indices from {indices_dump_pkl}")
            with open(indices_dump_pkl, "rb") as f:
                self.precomputed_latents = pickle.load(f)
            if dist.is_initialized():
                dist.barrier()
        else:
            tmp_loader = dataloader
            autoencoder.eval()
            autoencoder.to(device)
            latents_dict = {}
            desc = f"precomputing {self.split} VQ indices (rank:{rank})"

            for batch in tqdm(tmp_loader, desc=desc):
                df = batch.df.to(device)
                cond = (
                    batch.conditioning.to(device)
                    if hasattr(batch, "conditioning") and batch.conditioning is not None
                    else None
                )
                # encode to get VQ indices
                _z, _ = autoencoder.encode(df, condition=cond)
                indices = autoencoder.get_indices()  # (B, *grid_size)
                indices_flat = indices.view(indices.shape[0], -1).cpu()  # (B, seq_len)

                for i in range(len(batch.file_index)):
                    f_idx = batch.file_index[i].item()
                    t_idx = batch.timestep_index[i].item()

                    sample = {"x": indices_flat[i].numpy().astype(np.int64)}

                    if batch.phi is not None:
                        phi_i = batch.phi[i]
                        sample["phi"] = (
                            phi_i.cpu().numpy()
                            if isinstance(phi_i, torch.Tensor)
                            else np.asarray(phi_i)
                        )
                    else:
                        sample["phi"] = None

                    flux_i = batch.flux[i]
                    sample["flux"] = (
                        flux_i.cpu().numpy()
                        if isinstance(flux_i, torch.Tensor)
                        else np.asarray(flux_i)
                    )

                    ts_i = batch.timestep[i]
                    sample["timestep"] = (
                        ts_i.cpu().numpy()
                        if isinstance(ts_i, torch.Tensor)
                        else np.asarray(ts_i)
                    )

                    if batch.conditioning is not None:
                        cond_i = batch.conditioning[i]
                        if isinstance(cond_i, torch.Tensor):
                            cond_i = cond_i.cpu().numpy()
                        for j, key in enumerate(self.conditions):
                            sample[key] = cond_i[j]

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
                with open(indices_dump_pkl, "wb") as f:
                    pickle.dump(self.precomputed_latents, f)
                print(f"saved precomputed VQ indices to {indices_dump_pkl}")
            if dist.is_initialized():
                dist.barrier()

        # store dummy latent stats (discrete tokens don't need normalization)
        self.latent_stats = RunningMeanStd(shape=(1,))
        self.latent_stats.update(
            np.zeros((1,)), np.ones((1,)), np.zeros((1,)), np.ones((1,))
        )
        self.seq_len = next(iter(self.precomputed_latents.values()))["x"].shape[0]

        if rank == 0:
            print(
                f"VQ indices: seq_len={self.seq_len}, "
                f"codebook_size={autoencoder.vq.codebook_size}"
            )

        if isinstance(self.backend, KvikIOBackend):
            self.backend = KvikIOBackend(self.rank, use_kvikio=False)

        self._tensorize_latents()

    def _tensorize_latents(self):
        """Pre-build CycloneAESample objects with int64 index tensors."""
        if self.precomputed_latents is None:
            return
        avg_flux_cache = {}
        for f_id in self.metadata:
            fluxes = self.metadata[f_id]["flux"]
            avg_flux_cache[f_id] = torch.tensor(
                float(np.mean(fluxes[-80:])), dtype=self.dtype
            )

        for (file_index, t_index), sample in self.precomputed_latents.items():
            x = sample["x"]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).long()
            elif isinstance(x, torch.Tensor):
                x = x.long()

            flux = sample["flux"]
            if isinstance(flux, np.ndarray):
                flux = torch.as_tensor(flux, dtype=self.dtype)
            elif not isinstance(flux, torch.Tensor):
                flux = torch.tensor(flux, dtype=self.dtype)

            timestep = sample["timestep"]
            if isinstance(timestep, np.ndarray):
                timestep = torch.as_tensor(timestep, dtype=self.dtype)
            elif not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor(timestep, dtype=self.dtype)

            conditioning = None
            if self.conditions:
                cond_vals = []
                for k in self.conditions:
                    val = sample[k]
                    if isinstance(val, torch.Tensor):
                        cond_vals.append(val.to(dtype=self.dtype))
                    else:
                        cond_vals.append(torch.tensor(val, dtype=self.dtype))
                conditioning = torch.stack(cond_vals, dim=-1)

            sample["_sample"] = CycloneAESample(
                df=x,
                phi=None,
                flux=flux,
                avg_flux=avg_flux_cache.get(file_index, torch.tensor(0.0)),
                file_index=torch.tensor(file_index, dtype=torch.long),
                timestep_index=torch.tensor(t_index, dtype=torch.long),
                timestep=timestep,
                conditioning=conditioning,
            )
