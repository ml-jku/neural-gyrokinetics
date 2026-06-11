import os
from math import exp
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from neugk.dataset.backend import H5Backend, KvikIOBackend
from neugk.physics.integrals import FluxIntegral


class CycloneNFDataset(Dataset):
    def __init__(
        self,
        trajectory: str,
        timesteps: Union[int, Sequence[int]],
        path: str = "/restricteddata/ukaea/gyrokinetics/preprocessed",
        realpotens: bool = False,
        normalize: Optional[str] = None,
        normalize_coords: bool = False,
        norm_axes: Sequence[int] = (-4,),
        beta1: float = 1.0,
        beta2: float = 0.0,
        flux_fields: bool = False,
        flux_fields_train: bool = False,
        backend: str = "gds",
        prefer_dtype: Optional[str] = None,
    ):
        super().__init__()

        self.normalize = normalize
        # field axes to keep separate stats along, counted from the end:
        # vpar=-5, mu=-4, s=-3, x=-2, y=-1 (channel axis 0 is always kept).
        self.norm_axes = tuple(norm_axes)
        self.flux_fields = flux_fields
        self.flux_fields_train = flux_fields and flux_fields_train
        self.realpotens = realpotens
        self.beta1 = beta1
        self.beta2 = beta2
        self.backend = backend
        # None / "fp32" keeps the unchanged fp32 path; "bf16" prefers the
        # .bf16.bin siblings (half the bytes, ~2x faster) and falls back to
        # fp32 silently when they are absent.
        self.prefer_dtype = prefer_dtype

        trajectory = trajectory.replace(".h5", "")

        if isinstance(timesteps, int):
            timesteps = [timesteps]
        self.timesteps = timesteps

        # load via the shared neugk/dataset backends (kvikio/GDS or h5)
        self._backend = (
            KvikIOBackend(
                use_kvikio=(backend == "gds"), prefer_dtype=self.prefer_dtype
            )
            if backend in ("kvikio", "gds")
            else H5Backend()
        )
        self.raw_path = os.path.join(path, trajectory)
        self.df, self.phi, self.flux, self.geom = self._load(self.raw_path, timesteps)

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
        self.f_grid = rearrange(self.grid, "... d -> (...) d")

        # per-sample normalization, reducing the spatial dims but keeping the
        # velocity structure (C, [t,] vpar, mu); helps the INR fit. The raw
        # self.df is kept intact (full_df / CR stay in physical units).
        self.scale, self.shift = {}, {}
        self.f_df, self.scale["df"], self.shift["df"] = self._norm_field(self.df)
        if self.flux_fields_train:
            self.f_flux, self.scale["flux"], self.shift["flux"] = self._norm_field(
                self.flux
            )

        # strip _Lin or similar suffixes to find raw trajectory name
        # parallel-grid spacing: prefer metadata, fall back to the raw sgrid file
        if getattr(self, "_meta_ds", None) is not None:
            self.ds = self._meta_ds
        else:
            raw_traj = trajectory.split("_Lin")[0].split("_ifft")[0]
            raw_path = f"/restricteddata/ukaea/gyrokinetics/raw/{raw_traj}"
            if not os.path.isdir(raw_path):
                raw_path = f"/restricteddata/ukaea/gyrokinetics/raw/{trajectory}"
            sgrid = np.loadtxt(f"{raw_path}/sgrid")
            self.ds = float(sgrid[1] - sgrid[0])

    def _resolve_path(self, traj_path: str) -> str:
        """Try the formatted path first, then fall back to the bare path.

        ``format_path`` appends ``_ifft_realpotens`` (or similar) when
        ``spatial_ifft=True``. If the data was preprocessed without that
        suffix, the directory (KvikIO) or file (H5) won't exist.  Fall back to
        the original path so the eval runner works on both naming conventions.
        """
        be = self._backend
        formatted = be.format_path(
            traj_path, spatial_ifft=True, real_potens=self.realpotens
        )
        if be.exists(formatted):
            return formatted
        # try the bare path (already IFFT / real-potens on disk)
        if hasattr(be, "_strip_h5"):
            bare = be._strip_h5(traj_path)
        else:
            # H5Backend: keep .h5 extension
            bare = traj_path
        if be.exists(bare):
            return bare
        raise FileNotFoundError(
            f"Cannot find trajectory data for {traj_path!r}. "
            f"Tried: {formatted!r}, {bare!r}"
        )

    def _load(self, traj_path: str, timesteps: Sequence[int]):
        be = self._backend
        path = self._resolve_path(traj_path)
        meta = be.read_metadata(path, input_fields=["df"])
        self._meta_ds = float(meta["ds"]) if "ds" in meta else None

        # per-mode log1p std of the served GT turbulence spectra, computed once
        # from the metadata over the (offset) trajectory. Exposed via
        # `spectral_stds` so the spectral loss can std-normalise per mode,
        # identically to the autoencoder path. kyspec drives the kyspec loss,
        # fluxspec the qspec loss. None when the spectrum is absent.
        self.spectral_stds = {}
        _spec_offset = 80
        for _sk, _lk in (("kyspec", "kyspec"), ("fluxspec", "qspec")):
            if _sk in meta:
                _arr = np.log1p(np.asarray(meta[_sk], dtype=np.float64)[_spec_offset:])
                self.spectral_stds[_lk] = torch.as_tensor(
                    np.std(_arr, axis=0), dtype=torch.float32
                )

        res = tuple(int(x) for x in meta["resolution"])  # (nvpar, nmu, ns, nkx, nky)
        df_shape = (2, *res)
        phi_shape = tuple(meta["phi_mean"].shape) if "phi_mean" in meta else res[2:]
        flux_arr = meta["flux"] if "flux" in meta else meta["fluxes"]
        active = np.array([0, 1])

        # When prefer_dtype="bf16" the backend reads the .bf16.bin sibling and
        # hands back a torch.bfloat16 tensor (half the bytes off disk, no upcast
        # in the read). We keep the served df/phi in that read dtype here -- the
        # only consumer that strictly needs float32 is the FFT-based
        # FluxIntegral below, which gets its own upcast copy. Default (fp32) is
        # byte-identical to the old `.float()` behaviour.
        keep_bf16 = self.prefer_dtype == "bf16"

        def _as_read_dtype(t):
            # legacy fp32 path: unconditional .float(); bf16 path: keep bf16.
            return t.cpu() if keep_bf16 else t.float().cpu()

        dfs, phis, fluxes = [], [], []
        with be.open(path) as f:
            for t in timesteps:
                ts = str(t).zfill(5)
                df = _as_read_dtype(torch.as_tensor(be.read_df(f, ts, df_shape, active)))
                phi = _as_read_dtype(torch.as_tensor(be.read_phi(f, ts, phi_shape)))
                if phi.shape[0] != 2:
                    phi = torch.stack([phi, torch.zeros_like(phi)], dim=0)
                dfs.append(df.reshape(df_shape))
                phis.append(phi)
                fluxes.append(float(flux_arr[t]))

        dfs = torch.stack(dfs, 0).squeeze(0)
        phis = torch.stack(phis, 0).squeeze(0)
        fluxes = torch.tensor(fluxes).squeeze(0)
        geom = {
            k: torch.as_tensor(np.array(v)).squeeze(0)
            for k, v in meta["geometry"].items()
        }

        # sanity: df should be non-trivial for turbulent trajectories
        df_abs_mean = float(dfs.abs().mean())
        if df_abs_mean < 1e-20:
            raise RuntimeError(
                f"Loaded df is all zeros (mean abs = {df_abs_mean:.2e}) from "
                f"{path!r}. Check that the trajectory path and timesteps are correct."
            )
        if len(timesteps) > 1:
            dfs = rearrange(dfs, "t c ... -> c t ...")

        if self.flux_fields or self.realpotens:
            geom_ = {k: g[None] for k, g in geom.items()}
            # FluxIntegral / get_integrals use torch FFTs that do not support
            # bfloat16; give them a float32 copy. The served self.df keeps its
            # read dtype (bf16 when prefer_dtype="bf16").
            dfs_ = dfs.float().clone()
            if len(timesteps) == 1:
                dfs_ = dfs_[:, None]
            phis_int, fluxes_int = [], []
            for t_idx in range(len(timesteps)):
                assert dfs_[:, t_idx].shape[0] == 2
                integrator = FluxIntegral(flux_fields=self.flux_fields)
                phi_t, (_, fluxes_t, _) = integrator(geom_, df=dfs_[None, :, t_idx])
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

        return dfs, phis, fluxes, geom

    def _norm_field(self, field: torch.Tensor):
        """Normalize keeping separate stats along the channel axis and
        self.norm_axes (e.g. mu), reducing the rest. Returns the flattened
        normalized field and broadcastable (scale, shift)."""
        keep = {0} | {ax % field.ndim for ax in self.norm_axes}
        dims = tuple(d for d in range(field.ndim) if d not in keep)
        if self.normalize == "minmax":
            lo, hi = field.amin(dims, keepdim=True), field.amax(dims, keepdim=True)
            scale = (hi - lo) / self.beta1
            shift = lo + scale * self.beta2
        elif self.normalize == "zscore":
            scale = field.std(dims, keepdim=True) / self.beta1
            shift = field.mean(dims, keepdim=True) + scale * self.beta2
        else:
            shape = [1 if d in dims else s for d, s in enumerate(field.shape)]
            scale, shift = torch.ones(shape), torch.zeros(shape)
        scale = scale.clamp_min(1e-12)
        fieldn = rearrange((field - shift) / scale, "c ... -> c (...)")
        return fieldn, scale, shift

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, int):
            index = [index]
        f = self.f_flux if self.flux_fields_train else self.f_df
        return f[:, index].T, self.f_grid[index, :]

    def to(self, device: torch.device):
        # df/grid back the full_df ground truth and sample_field coords; without
        # them on-device the eval's GT FluxIntegral silently runs on CPU (~5x).
        self.df = self.df.to(device)
        self.grid = self.grid.to(device)
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
        if getattr(self, "spectral_stds", None):
            self.spectral_stds = {
                k: v.to(device) for k, v in self.spectral_stds.items()
            }
        return self

    def shuffle(self):
        # perm on the data's device: a CPU randperm here forces a host->device
        # index copy + slow gather (~200ms for 11M points), vs ~2ms on-device.
        perm = torch.randperm(self.f_grid.shape[0], device=self.f_df.device)
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
    def grid_size(self) -> Tuple[int, ...]:
        return tuple(self.df.shape[1:])

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
        self._prebuilt = False

    def _prebuild(self):
        """Pre-build batch index ranges for O(1) iteration (avoids Python
        slicing overhead on every ``__getitem__``)."""
        n = int(len(self.dataset) * self.subsample)
        bs = self.batch_size
        self._batches = [(start, min(start + bs, n)) for start in range(0, n, bs)]
        self._prebuilt = True

    def __len__(self):
        if self._prebuilt:
            return len(self._batches)
        return int(
            (len(self.dataset) + self.batch_size - 1)
            // self.batch_size
            * self.subsample
        )

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        if not self._prebuilt:
            self._prebuild()
        if self.shuffle:
            self.dataset.shuffle()

        for start, end in self._batches:
            dfs, coords = self.dataset[start:end]
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
