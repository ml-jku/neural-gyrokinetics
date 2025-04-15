from typing import List, Callable, Dict, Optional, Tuple
import warnings
from functools import partial
import torch
from torch.special import bessel_j0 as j0, i0
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.distributed import get_rank, is_initialized
from einops import rearrange

from concurrent.futures import ThreadPoolExecutor

from utils import save_model_and_config
from dataset.cyclone import CycloneDataset, CycloneSample


def relative_norm_mse(x, y, dim_to_keep=None, squared=True):
    if x.ndim == 2 and y.ndim == 1:
        y = y.unsqueeze(1)
    assert x.shape == y.shape, "Mismatch in dimensions for computing loss"
    if dim_to_keep is None:
        y = y.flatten(1)
        diff = x.flatten(1) - y
        diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        if squared:
            diff_norms, y_norms = diff_norms**2, y_norms**2
        # sum over timesteps and mean over examples in batch
        return torch.mean(diff_norms / y_norms)
    else:
        # TODO: Check if this is necessary
        y = y.flatten(2)
        diff = x.flatten(2) - y
        diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        if squared:
            diff_norms, y_norms = diff_norms**2, y_norms**2
        dims = [i for i in range(len(y_norms.shape))][dim_to_keep + 1 :]
        return torch.mean(diff_norms / y_norms, dim=dims)


class FluxIntegral(nn.Module):
    def __init__(self, geometry: Dict, dtype: torch.dtype = torch.float32):
        super().__init__()

        self.geometry = geometry

        # expand geometry constants for broadcasting
        # grids
        krho = rearrange(geometry["krho"], "y -> 1 1 1 1 y")
        self.register_buffer("krho", krho.to(dtype=dtype))
        kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 x 1")
        self.register_buffer("kxrh", kxrh.to(dtype=dtype))
        ints = rearrange(geometry["ints"], "s -> 1 1 s 1 1")
        self.register_buffer("ints", ints.to(dtype=dtype))
        intmu = rearrange(geometry["intmu"], "mu -> 1 mu 1 1 1")
        self.register_buffer("intmu", intmu.to(dtype=dtype))
        intvp = rearrange(geometry["intvp"], "par -> par 1 1 1 1")
        self.register_buffer("intvp", intvp.to(dtype=dtype))
        vpgr = rearrange(geometry["vpgr"], "par -> par 1 1 1 1")
        self.register_buffer("vpgr", vpgr.to(dtype=dtype))
        mugr = rearrange(geometry["mugr"], "mu -> 1 mu 1 1 1")
        self.register_buffer("mugr", mugr.to(dtype=dtype))
        # settings
        little_g = rearrange(geometry["little_g"], "s three -> three 1 1 s 1 1")
        self.register_buffer("little_g", little_g.to(dtype=dtype))
        bn = rearrange(geometry["bn"], "s -> 1 1 s 1 1")
        self.register_buffer("bn", bn.to(dtype=dtype))
        efun = rearrange(geometry["efun"], "s -> 1 1 s 1 1")
        self.register_buffer("efun", efun.to(dtype=dtype))
        rfun = rearrange(geometry["rfun"], "s -> 1 1 s 1 1")
        self.register_buffer("rfun", rfun.to(dtype=dtype))
        bt_frac = rearrange(geometry["bt_frac"], "s -> 1 1 s 1 1")
        self.register_buffer("bt_frac", bt_frac.to(dtype=dtype))
        parseval = rearrange(geometry["parseval"], "y -> 1 1 1 1 y")
        self.register_buffer("parseval", parseval.to(dtype=dtype))
        mas, vthrat, signz = geometry["mas"], geometry["vthrat"], geometry["signz"]
        # bessel for gyroaverage
        krloc = torch.sqrt(
            krho**2 * little_g[0]
            + 2 * krho * kxrh * little_g[1]
            + kxrh**2 * little_g[2]
        )
        bessel = j0(mas * vthrat * krloc * torch.sqrt(2.0 * mugr / bn) / signz)
        # exponentially scaled bessel i0 function
        gamma = 0.5 * ((mas * vthrat * krloc) / (signz * bn)) ** 2
        gamma = i0(gamma) * torch.exp(-gamma)
        self.register_buffer("bessel", bessel.to(dtype=dtype))
        self.register_buffer("gamma", gamma.to(dtype=dtype))

        self._fwd_vmap = torch.vmap(self.forward_single)

    def _df_fft(self, df: torch.Tensor, norm: str = "forward"):
        df = df.movedim(0, -1).contiguous()
        df = torch.view_as_complex(df)
        df = torch.fft.fftn(df, dim=(3, 4), norm=norm)
        return torch.fft.ifftshift(df, dim=(3,))

    def _phi_to_spc(self, phi: torch.Tensor, out_shape: Tuple, norm: str = "forward"):
        phi = phi.movedim(0, -1).contiguous()
        phi = torch.view_as_complex(phi)
        phi = torch.fft.fftn(phi, dim=(0, 2), norm=norm)
        phi = torch.fft.fftshift(phi, dim=(0, 2))
        # unpad (and positive half of spectra)
        if phi.shape != out_shape:
            nx, _, ny = out_shape
            phi = phi[..., phi.shape[-1] // 2 :]
            xpad = (phi.shape[0] - nx) // 2 + 1
            phi = phi[xpad : nx + xpad, :, :ny]
        return rearrange(phi, "x s y -> s x y")

    def _spc_to_phi(
        self,
        spc: torch.Tensor,
        original_shape: Tuple = (392, 16, 96),
        repad: bool = False,
        norm: str = "forward",
    ):
        spc = rearrange(spc, "s x y -> x s y")
        spc_nx, _, spc_ny = spc.shape
        if repad:
            # pad x
            nx, _, ny = original_shape
            x_pad_total = nx - spc_nx
            x_pad_left = x_pad_total // 2
            x_pad_right = x_pad_total - x_pad_left
            spc = F.pad(spc, (0, 0, 0, 0, x_pad_left, x_pad_right))
            # y full spectrum and pad
            spc_flipped_y = torch.flip(spc, dims=[-1])
            spc = torch.cat([spc_flipped_y, spc], dim=-1)
            y_pad_total = ny - spc_ny * 2
            y_pad_left = y_pad_total // 2
            y_pad_right = y_pad_total - y_pad_left
            spc = F.pad(spc, (y_pad_left, y_pad_right, 0, 0))
        # ifft
        # TODO investigate the shift for phi
        phi = torch.fft.ifftshift(spc, dim=2)
        phi = torch.fft.ifftn(phi, dim=(0, 2), norm=norm)
        phi = torch.view_as_real(phi).movedim(-1, 0).contiguous()
        return phi  # (c, x, s, y)

    def pev_fluxes(
        self,
        df: torch.Tensor,
        phi: torch.Tensor,
        magnitude: bool = False,
    ):
        """
        Computes particle, heat and momentum fluxes based on the distribution function
        and electrostatic potential.

        Args:
            df (torch.Tensor): 5D density function. Shape: (b, c, vpar, vmu, s, x, y).
            phi (torch.Tensor): 3D electrostatic potential. Shape: (1, x, s, y).
            geometry (Dict): Dictionary containing geometry parameters and settings.
            magnitude (bool, optional): Use df and phi absolutes. Default: False.
        """
        phi_gyro = self.bessel * rearrange(phi.squeeze(), "s x y -> 1 1 s x y")
        # absolute values of df and phi
        if magnitude:
            df = -1j * torch.abs(df)
            phi_gyro = torch.abs(phi_gyro)
        # grid derivatives
        dum = self.parseval * self.ints * (self.efun * self.krho) * df
        dum1 = dum * torch.conj(phi_gyro)
        dum2 = dum1 * self.bn
        d3v = self.ints * self.geometry["d2X"] * self.intmu * self.bn * self.intvp
        signB = self.geometry["signB"]
        # flux fields
        dum1 = torch.imag(dum1)
        dum2 = torch.imag(dum2)
        pflux = d3v * dum1
        eflux = d3v * (self.vpgr**2 * dum1 + 2 * self.mugr * dum2)
        vflux = d3v * (dum1 * self.vpgr * self.rfun * self.bt_frac * signB)
        # sum total fluxes
        return pflux.sum(), eflux.sum(), vflux.sum()

    def phi(self, df: torch.Tensor):
        # density of the species
        de = 1.0
        signz, tmp = self.geometry["signz"], self.geometry["tmp"]
        cfen = torch.zeros_like(self.ints)
        # poisson terms
        # integral mapping
        poisson_int = signz * de * self.intmu * self.intvp * self.bessel * self.bn
        poisson_int = torch.where(torch.abs(self.intvp) < 1e-9, 0.0, poisson_int)
        # matz and maty zonal flow correction
        diagz = (
            signz
            * de
            * (
                signz * (self.gamma - 1.0) * torch.exp(-cfen) / tmp
                - torch.exp(-cfen) / tmp
            )
        )
        matz = -self.ints / diagz
        matz[..., 1:] = 0.0  # only keep y=0 (turb)
        maty = (-matz * torch.exp(-cfen)).sum((2,), keepdim=True)
        maty = tmp / (de * torch.exp(-cfen)) + maty / torch.exp(-cfen)
        maty[..., 0, :] = 1 + 0j
        maty = torch.where(maty == 0, 1.0, maty)  # avoid infs
        maty = 1 / maty
        maty[..., 1:] = 0.0  # only keep y=0 (turb)
        # diagonal normalization term
        poisson_diag = torch.exp(-cfen) * (signz**2) * de * (self.gamma - 1.0) / tmp
        poisson_diag[..., 0, 0] = 0.0
        poisson_diag = poisson_diag + signz * torch.exp(-cfen) * de / tmp
        # first usmv
        phi = (1 + 0j) * poisson_int * df
        # integrate velocity space
        phi = phi.sum((0, 1), keepdim=True)
        # second usmv
        bufphi = (1 + 0j) * matz * phi
        # surface average
        bufphi = bufphi.sum((2, 4), keepdim=True)
        # third usmv
        phi = phi + (1 + 0j) * maty * bufphi
        # normalize
        phi = phi * poisson_diag
        return phi.squeeze()

    def forward_single(self, df: torch.Tensor, phi: Optional[torch.Tensor] = None):
        ns, nx, ny = df.shape[3:]
        # df to fourier
        df = self._df_fft(df)  # (par, mu, s, x, y)
        phi_int = self.phi(df)  # (s, x, y)
        phi_ = phi_int.clone()
        if phi is not None:
            phi_ = self._phi_to_spc(phi, out_shape=(nx, ns, ny))  # (s, x, y)
        pflux, eflux, vflux = self.pev_fluxes(df, phi_)
        # integrated phi repad and back to real
        phi_int = self._spc_to_phi(phi_int)
        return phi_int, (pflux, eflux, vflux)

    def forward(self, df: torch.Tensor, phi: Optional[torch.Tensor] = None):
        if phi is not None:
            return self._fwd_vmap(df, phi)
        else:
            return self._fwd_vmap(df)


class LossWrapper(nn.Module):
    def __init__(
        self,
        weights: Dict,
        geometry: Optional[Dict] = None,
        denormalize_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.weights = weights
        self._extras = ["flux_int", "phi_int", "flux_cross", "phi_cross"]
        self.integrator = FluxIntegral(geometry)
        self.denormalize_fn = denormalize_fn

        self.loss_fns = {
            "df": partial(self.data_loss, "df"),
            "phi": partial(self.data_loss, "phi"),
            "flux": partial(self.data_loss, "flux"),
        }

    def data_loss(
        self, key: str, preds: Dict[str, torch.Tensor], tgts: Dict[str, torch.Tensor]
    ):
        if key == "flux":
            # NOTE: regular mse for flux otherwise might get nans (for linear phase)
            return F.mse_loss(preds["flux"], tgts["flux"])
        else:
            return relative_norm_mse(preds[key], tgts[key])

    def integral_loss(
        self,
        preds: Dict[str, torch.Tensor],
        tgts: Dict[str, torch.Tensor],
        idx_data: Optional[Dict[str, torch.Tensor]] = None,
    ):
        assert self.denormalize_fn is not None
        if self.training:
            pred_df = []
            pred_phi = []
            tgt_phi = []
            for b, f in enumerate(idx_data["file_index"].tolist()):
                pred_df.append(self.denormalize_fn(f, df=preds["df"][b]))
                if "phi" in preds:
                    pred_phi.append(self.denormalize_fn(f, phi=preds["phi"][b]))
                tgt_phi.append(self.denormalize_fn(f, phi=tgts["phi"][b]))
            pred_df = torch.stack(pred_df)
            if len(pred_phi) > 0:
                pred_phi = torch.stack(pred_phi)
            else:
                pred_phi = None
            tgt_phi = torch.stack(tgt_phi)
        else:
            # already denormalized for evaluation
            pred_df = preds["df"]
            pred_phi = preds["phi"] if "phi" in preds else None
            tgt_phi = tgts["phi"]

        pred_eflux, tgt_eflux = preds["flux"], tgts["flux"]

        pphi_int, (pflux, eflux, _) = self.integrator(pred_df, pred_phi)
        int_losses = {}
        # NOTE: these losses are in unnormalized space
        int_losses["phi_int"] = relative_norm_mse(pphi_int, tgt_phi)
        # pflux -> 0, eflux -> heat flux
        int_losses["flux_int"] = (pflux**2).mean() + F.mse_loss(eflux, tgt_eflux)
        # mimicry / cross terms in the loss (between prediction heads and integrals)
        if "phi" in preds:
            int_losses["phi_cross"] = relative_norm_mse(pred_phi, pphi_int)
        if "flux" in preds:
            int_losses["flux_cross"] = F.mse_loss(pred_eflux, eflux)

        return int_losses, {"phi": pphi_int, "pflux": pflux, "eflux": eflux}

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        tgts: Dict[str, torch.Tensor],
        idx_data: Optional[Dict[str, torch.Tensor]] = None,
        integrals: bool = True,
    ):
        losses = {}
        int_losses = None
        # NOTE: newtwork predicts phi -> weight["phi_int"] = 0 (otherwise summed twice)
        # NOTE: if weight["phi"] > 0 -> weight["phi_int"] = 0 and vice versa
        # only compute integrals if requested by weights or in eval
        eval_integrals = not self.training and integrals
        if sum([self.weights.get(k, 0.0) for k in self._extras]) > 0 or eval_integrals:
            int_losses, integrated = self.integral_loss(preds, tgts, idx_data)
        if self.training:
            loss_keys = self.weights
        else:
            loss_keys = list(self.loss_fns.keys()) + list(int_losses.keys())
        for k in loss_keys:
            if "int" in k or "cross" in k:
                losses[k] = int_losses[k]
            else:
                losses[k] = self.loss_fns[k](preds, tgts)
        loss = sum([w * losses[k] for k, w in self.weights.items()])
        if self.training:
            return loss, losses
        else:
            return loss, losses, integrated

    @property
    def active_losses(self):
        return [k for k in self.weights if self.weights[k] > 0.0]

    @property
    def all_losses(self):
        return list(self.loss_fns.keys()) + self._extras

    def __len__(self):
        return len(self.all_losses)


def get_pushforward_fn(
    n_unrolls_schedule: List[int],
    probs_schedule: List[float],
    epoch_schedule: List[float],
    predict_delta: bool,
    dataset: CycloneDataset,
    bundle_steps: int,
    use_amp: bool = False,
    device: str = None,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        inputs: Dict,
        gts: Dict,
        conds: Dict,
        idx_data: Dict,
        epoch: int,
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(epoch_schedule)).sum()
        if not idx:
            return inputs, gts, conds

        # sample number of steps
        curr_probs = [p / sum(probs_schedule[:idx]) for p in probs_schedule[:idx]]
        pf_n_unrolls = np.random.choice(n_unrolls_schedule[:idx], p=curr_probs)

        # cap the unroll steps depending on the current max timestep
        n_unrolls = []
        for i, f_idx in enumerate(idx_data["file_index"].tolist()):
            sleft = (
                dataset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i])
            ) // bundle_steps - 1
            n_unrolls.append(min(sleft, pf_n_unrolls))
        n_unrolls = min(n_unrolls)

        if n_unrolls < 2:
            return inputs, gts, conds

        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + n_unrolls * ts_step, ts_step))
            for ts in idx_data["timestep_index"].tolist()
        ]
        tsteps = dataset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(
                file_idx.cpu(),
                ts_unrolled.cpu(),
            )

        executor = ThreadPoolExecutor(max_workers=1)
        with torch.no_grad():
            ts_unrolled = idx_data["timestep_index"] + (n_unrolls - 1) * ts_step
            future = executor.submit(
                fetch_target, dataset, idx_data["file_index"], ts_unrolled
            )

            inputs_t = inputs.copy()
            for i in range(n_unrolls - 1):
                use_bf16 = use_amp and torch.cuda.is_bf16_supported()
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    conds["timestep"] = tsteps[:, i].to(device)
                    outputs = model(**inputs_t, **conds)
                    if predict_delta:
                        for key in dataset.input_fields:
                            outputs[key] = outputs[key] + inputs[key]

                    for key in dataset.input_fields:
                        inputs_t[key] = outputs[key]

            # Get the result when needed
            unrolled: CycloneSample = future.result()

        gts = {
            k: getattr(unrolled, k).to(device, non_blocking=True)
            for k in gts.keys()
            if k is not None
        }
        conds = {
            k: getattr(unrolled, k).to(device, non_blocking=True)
            for k in conds.keys()
            if k is not None
        }
        return inputs_t, gts, conds

    return _loss_fn


def pretrain_autoencoder(model, cfg, trainloader, valloaders, writer, device):
    if cfg.training.pretraining_kwargs.target_modules == "all":
        target_modules = model.parameters()
    else:
        target_modules = []
        for n, p in model.named_parameters():
            for t in cfg.training.pretraining_kwargs.target_modules:
                if t in n:
                    target_modules.append(p)

    scaler = torch.amp.GradScaler(device=device, enabled=cfg.use_amp)
    use_bf16 = cfg.use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_ddp = is_initialized()
    if use_ddp:
        rank = get_rank()
    n_epochs = cfg.training.pretraining_kwargs.n_epochs

    opt = torch.optim.Adam(
        target_modules,
        lr=cfg.training.pretraining_kwargs.lr,
        weight_decay=cfg.training.pretraining_kwargs.weight_decay,
    )

    is_main_proc = not rank if use_ddp else True
    if cfg.training.pretraining_kwargs.scheduler is not None:
        total_steps = n_epochs * len(trainloader)
        scheduler = get_scheduler(
            name=cfg.training.pretraining_kwargs.scheduler,
            optimizer=opt,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

    use_tqdm = cfg.logging.tqdm if not use_ddp else False
    loss_val_min = torch.inf
    for epoch in range(1, cfg.training.pretraining_kwargs.n_epochs + 1):
        train_mse = 0
        if use_tqdm or (use_ddp and not rank):
            trainloader = tqdm(trainloader, "AE pretraining")
        for sample in trainloader:
            sample: CycloneSample
            x = sample.x.to(device)
            ts = sample.timestep.to(device)
            itg = sample.itg.to(device)

            with torch.autocast(cfg.device, dtype=amp_dtype, enabled=cfg.use_amp):
                if cfg.training.pretraining_kwargs.target_modules == "all":
                    pred_x = model(x, timestep=ts, itg=itg)
                else:
                    if hasattr(model, "module"):
                        z, pad_ax = model.module.patch_encode(x)
                    else:
                        z, pad_ax = model.patch_encode(x)

                    if cfg.training.pretraining_kwargs.add_noise:
                        z = z + torch.normal(0, 1e-3, size=(z.shape), device=z.device)

                    cond = {"timestep": ts, "itg": itg}
                    if hasattr(model, "module"):
                        cond = model.module.condition(cond)["condition"]
                        pred_x = model.module.patch_decode(z, cond, pad_ax)
                    else:
                        cond = model.condition(cond)["condition"]
                        pred_x = model.patch_decode(z, cond, pad_ax)
                if cfg.training.predict_delta:
                    pred_x = x + pred_x
                loss = relative_norm_mse(pred_x, x)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.training.pretraining_kwargs.clip_grad:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if cfg.training.pretraining_kwargs.scheduler is not None:
                scheduler.step()

            train_mse += loss.item()

        train_mse = train_mse / len(trainloader)

        val_log = ""
        if ((epoch % 10) == 0 or epoch == 1) and is_main_proc:
            for val_idx, valloader in enumerate(valloaders):
                valname = "holdout_trajectories" if val_idx == 0 else "holdout_samples"
                val_mse = 0
                if cfg.logging.tqdm:
                    valloader = tqdm(valloader, "AE evaluation")
                for sample in valloader:
                    sample: CycloneSample
                    x = sample.x.to(device)
                    ts = sample.timestep.to(device)
                    itg = sample.itg.to(device)
                    cond = {"timestep": ts, "itg": itg}
                    if hasattr(model, "module"):
                        z, pad_ax = model.module.patch_encode(x)
                        cond = model.module.condition(cond)["condition"]
                        pred_x = model.module.patch_decode(z, cond, pad_ax)
                    else:
                        z, pad_ax = model.module.patch_encode(x)
                        cond = model.condition(cond)["condition"]
                        pred_x = model.patch_decode(z, cond, pad_ax)
                    if cfg.training.predict_delta:
                        pred_x = x + pred_x
                    loss = relative_norm_mse(pred_x, x)
                    val_mse += loss.item()
                val_mse = val_mse / len(valloader)
                val_log += f", val_{valname}/relative_norm_mse: {val_mse:.4f}"
                if is_main_proc and writer:
                    writer.log({f"pretrain/val_{valname}_relative_norm_mse": val_mse})

            if cfg.ckpt_path is not None and is_main_proc:
                # Save model if validation loss improves
                loss_val_min = save_model_and_config(
                    model,
                    optimizer=opt,
                    cfg=cfg,
                    epoch=epoch,
                    # TODO decide target metric
                    val_loss=val_mse,
                    loss_val_min=loss_val_min,
                )
            else:
                warnings.warn("`cfg.ckpt_path` not set: checkpoints will not be stored")

        epoch_str = str(epoch).zfill(
            len(str(int(cfg.training.pretraining_kwargs.n_epochs)))
        )
        if is_main_proc and writer:
            print(
                f"AE epoch: {epoch_str}, "
                f"train/relative_norm_mse: {train_mse:.4f}{val_log}"
            )
            writer.log(
                {
                    "pretrain/train_relative_norm_mse": train_mse,
                    "pretrain/train_lr": (
                        scheduler.get_last_lr()[0]
                        if cfg.training.pretraining_kwargs.scheduler is not None
                        else cfg.training.pretraining_kwargs.lr
                    ),
                }
            )

    if cfg.training.pretraining_kwargs.freeze_after:
        # freeze patching
        print("Freezing AE weights...")
        if hasattr(model, "module"):
            model = model.module
        model.patch_embed.requires_grad_(False)
        model.unpatch.requires_grad_(False)

    print("Pretraining done!\n\n")

    return model
