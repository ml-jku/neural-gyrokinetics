from typing import List, Callable, Dict, Optional, Tuple
import warnings
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
    assert x.shape == y.shape, "Mismatch in dimensions for computing loss"
    if dim_to_keep is None:
        x = x.flatten(1)
        y = y.flatten(1)
    else:
        # inference mode
        x = x.flatten(2)
        y = y.flatten(2)
    diff = x - y
    diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
    y_norms = torch.linalg.norm(y, ord=2, dim=-1)
    if squared:
        diff_norms, y_norms = diff_norms ** 2, y_norms ** 2
    if dim_to_keep is None:
        # sum over timesteps and mean over examples in batch
        return torch.mean(diff_norms / y_norms)
    else:
        dims = [i for i in range(len(y_norms.shape))][dim_to_keep + 1 :]
        return torch.mean(diff_norms / y_norms, dim=dims)


class FluxIntegral(nn.Module):
    def __init__(self, geometry: Dict):
        super().__init__()

        self.geometry = geometry

        # expand geometry constants for broadcasting
        # grids
        krho = rearrange(geometry["krho"], "y -> 1 1 1 1 y")
        self.register_buffer("krho", krho)
        kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 x 1")
        self.register_buffer("kxrh", kxrh)
        ints = rearrange(geometry["ints"], "s -> 1 1 s 1 1")
        self.register_buffer("ints", ints)
        intmu = rearrange(geometry["intmu"], "mu -> 1 mu 1 1 1")
        self.register_buffer("intmu", intmu)
        intvp = rearrange(geometry["intvp"], "par -> par 1 1 1 1")
        self.register_buffer("intvp", intvp)
        vpgr = rearrange(geometry["vpgr"], "par -> par 1 1 1 1")
        self.register_buffer("vpgr", vpgr)
        mugr = rearrange(geometry["mugr"], "mu -> 1 mu 1 1 1")
        self.register_buffer("mugr", mugr)
        # settings
        little_g = rearrange(geometry["little_g"], "s three -> three 1 1 s 1 1")
        self.register_buffer("little_g", little_g)
        bn = rearrange(geometry["bn"], "s -> 1 1 s 1 1")
        self.register_buffer("bn", bn)
        efun = rearrange(geometry["efun"], "s -> 1 1 s 1 1")
        self.register_buffer("efun", efun)
        rfun = rearrange(geometry["rfun"], "s -> 1 1 s 1 1")
        self.register_buffer("rfun", rfun)
        bt_frac = rearrange(geometry["bt_frac"], "s -> 1 1 s 1 1")
        self.register_buffer("bt_frac", bt_frac)
        parseval = rearrange(geometry["parseval"], "y -> 1 1 1 1 y")
        self.register_buffer("parseval", parseval)
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
        self.register_buffer("bessel", bessel)
        self.register_buffer("gamma", gamma)

    def _df_fft(self, df: torch.Tensor, norm: str = "backward"):
        df = df.movedim(0, -1).contiguous()
        df = torch.view_as_complex(df)
        df = torch.fft.fftn(df, dim=(3, 4), norm=norm)
        return torch.fft.ifftshift(df, dim=(3,))

    def _phi_to_spc(self, phi: torch.Tensor, out_shape: Tuple, norm: str = "forward"):
        # drop channels and apply fft
        phi = torch.fft.fftn(phi.squeeze(0), dim=(0, 2), norm=norm)
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
        norm: str = "forward",
    ):
        spc = rearrange(spc, "s x y -> x s y")
        nx, _, ny = original_shape
        spc_nx, _, spc_ny = spc.shape
        # x pad
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
        spc = torch.fft.ifftshift(spc, dim=(0, 2))
        phi = torch.fft.ifftn(spc, dim=(0, 2), norm=norm)
        return phi.unsqueeze(0)  # (c, x, s, y)

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
        ns, nx, ny = df.shape[-3:]
        # phi tensor
        phi = torch.zeros((ns, nx, ny), dtype=df.dtype, device=df.device)
        bufphi = torch.zeros((ns, nx, ny), dtype=df.dtype, device=df.device)
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
        bufphi = bufphi + (1 + 0j) * matz * phi
        # surface average
        bufphi = bufphi.sum(
            (
                2,
                4,
            ),
            keepdim=True,
        )
        # third usmv
        phi = phi + (1 + 0j) * maty * bufphi
        # normalize
        phi = phi * poisson_diag
        return phi.squeeze()

    def forward_single(self, df: torch.Tensor, phi: Optional[torch.Tensor] = None):
        ns, nx, ny = df.shape[3:]
        # df to fourier
        df = self._df_fft(df)  # (par, mu, s, x, y)
        if phi is None:
            phi = self.phi(df)  # (s, x, y)
        else:
            phi_ = self._phi_to_spc(phi, out_shape=(nx, ns, ny))  # (s, x, y)
        pflux, eflux, vflux = self.pev_fluxes(df, phi_)
        # phi repad and back to real
        phi_ = self._spc_to_phi(phi_)
        return phi, (pflux, eflux, vflux)

    def forward(self, df: torch.Tensor, phi: Optional[torch.Tensor] = None):
        # return torch.vmap(self.forward_single)(df, phi)
        self.forward_single(df[0], phi[0])


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
    ) -> Tuple[Dict[str, torch.Tensor],Dict[str, torch.Tensor],Dict[str, torch.Tensor]]:
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
            sleft = (dataset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i])) // bundle_steps - 1
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
            future = executor.submit(fetch_target, dataset, idx_data["file_index"], ts_unrolled)

            inputs_t = inputs.copy()
            for i in range(n_unrolls - 1):
                use_bf16 = use_amp and torch.cuda.is_bf16_supported()
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    conds['timestep'] = tsteps[:, i].to(device)
                    outputs = model(**inputs_t, **conds)
                    if predict_delta:
                        for key in dataset.input_fields:
                            outputs[key] = outputs[key] + inputs[key]

                    for key in dataset.input_fields:
                        inputs_t[key] = outputs[key]

            # Get the result when needed
            unrolled: CycloneSample = future.result()

        gts = {k: getattr(unrolled, k).to(device, non_blocking=True) for k in gts.keys() if k is not None}
        conds = {k: getattr(unrolled, k).to(device, non_blocking=True) for k in conds.keys() if k is not None}
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
