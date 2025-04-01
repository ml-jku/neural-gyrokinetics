from typing import List, Callable, Optional
import warnings
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.distributed import get_rank, is_initialized

from concurrent.futures import ThreadPoolExecutor

from utils import save_model_and_config
from dataset.cyclone import CycloneDataset, CycloneSample


def relative_norm_mse(x, y, dim_to_keep=None, squared=True):
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


def df_fft(df: torch.Tensor, norm: str = "backward"):
    df = df.movedim(0, -1).contiguous()
    df = torch.view_as_complex(df)
    df = torch.fft.fftn(df, dim=(3, 4), norm=norm)
    return torch.fft.ifftshift(df, dim=(3,))


def phi_to_spc(phi: torch.Tensor, out_shape: Tuple, norm: str = "forward"):
    # drop channels and apply fft
    phi = torch.fft.fftn(phi.squeeze(0), dim=(0, 2), norm=norm)
    phi = torch.fft.fftshift(phi, dim=(2, 0))
    # unpad (and positive half of spectra)
    nx, _, ny = out_shape
    phi = phi[..., phi.shape[-1] // 2 :]
    xpad = (phi.shape[0] - nx) // 2 + 1
    return phi[xpad : nx + xpad, :, :ny]


@torch.vmap
def pev_integral_df_phi(
    df: torch.Tensor,
    phi: torch.Tensor,
    geometry: Dict,
    aggregate: bool = True,
    magnitude: bool = False,
):
    """
    Computes particle, heat and momentum fluxes based on the distribution function (df)
    and electrostatic potential (phi).

    Args:
        df (torch.Tensor): 5D density function. Shape: (b, c, vpar, vmu, s, x, y).
        phi (torch.Tensor): 3D electrostatic potential. Shape: (b, 1, x, s, y).
        geometry (Dict): Dictionary containing geometry parameters and settings.
        aggregate (bool, optional): Whether to return the summed fluxes. Default: True.
        magnitude (bool, optional): Whether to use df and phi absolutes. Default: False.
    """
    ns, nx, ny = df.shape[3:]
    # df to fourier, phi to fourier and unpad
    df = df_fft(df)  # (par, mu, s, x, y)
    phi = phi_to_spc(phi, out_shape=(nx, ns, ny))  # (x, s, y)
    # expand geometry constants for broadcasting
    # grids
    krho = rearrange(geometry["krho"], "y -> 1 1 1 1 y")
    kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 x 1")
    ints = rearrange(geometry["ints"], "s -> 1 1 s 1 1")
    intmu = rearrange(geometry["intmu"], "mu -> 1 mu 1 1 1")
    intvp = rearrange(geometry["intvp"], "par -> par 1 1 1 1")
    vpgr = rearrange(geometry["vpgr"], "par -> par 1 1 1 1")
    mugr = rearrange(geometry["mugr"], "mu -> 1 mu 1 1 1")
    # settings
    little_g = rearrange(geometry["little_g"], "s three -> three 1 1 s 1 1")
    bn = rearrange(geometry["bn"], "s -> 1 1 s 1 1")
    efun = rearrange(geometry["efun"], "s -> 1 1 s 1 1")
    rfun = rearrange(geometry["rfun"], "s -> 1 1 s 1 1")
    bt_frac = rearrange(geometry["bt_frac"], "s -> 1 1 s 1 1")
    parseval = rearrange(geometry["parseval"], "y -> 1 1 1 1 y")
    mas, vthrat, signz = geometry["mas"], geometry["vthrat"], geometry["signz"]
    # gyroaveraged phi
    krloc = torch.sqrt(
        krho**2 * little_g[0] + 2 * krho * kxrh * little_g[1] + kxrh**2 * little_g[2]
    )
    bessel = torch.special.bessel_j0(
        mas * vthrat * krloc * torch.sqrt(2.0 * mugr / bn) / signz
    )
    phi_gyro = bessel * rearrange(phi, "x s y -> 1 1 s x y")
    # absolute values of df and phi
    if magnitude:
        df = -1j * torch.abs(df)
        phi_gyro = torch.abs(phi_gyro)
    # grid derivatives
    dum = parseval * ints * (efun * krho) * df
    dum1 = dum * torch.conj(phi_gyro)
    dum2 = dum1 * bn
    d3X = ints * geometry["d2X"]
    d3v = intmu * bn * intvp
    signB = geometry["signB"]
    # flux fields
    pflux = d3X * d3v * torch.imag(dum1)
    eflux = d3X * d3v * (vpgr**2 * torch.imag(dum1) + 2 * mugr * torch.imag(dum2))
    vflux = d3X * d3v * (torch.imag(dum1) * vpgr * rfun * bt_frac * signB)
    # sum total fluxes
    if aggregate:
        pflux = pflux.sum()
        eflux = eflux.sum()
        vflux = vflux.sum()
    return pflux, eflux, vflux


def flux_loss(
    df: torch.Tensor,
    phi: torch.Tensor,
    gt_fluxes: Dict[str, torch.Tensor],
    geometry: Dict,
):
    fluxes = {}
    fluxes["p"], fluxes["e"], fluxes["v"] = pev_integral_df_phi(df, phi, geometry)
    loss = 0.0
    for k in gt_fluxes:
        loss += relative_norm_mse(fluxes[k].unsqueeze(0), gt_fluxes[k].unsqueeze(0))
    return loss


def get_pushforward_fn(
    n_unrolls_schedule: List[int],
    probs_schedule: List[float],
    epoch_schedule: List[float],
    predict_delta: bool,
    dataset: CycloneDataset,
    bundle_steps: int,
    use_amp: bool = False,
    use_potentials: bool = False,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        ts: torch.Tensor,
        itg: torch.Tensor,
        ts_idx: torch.Tensor,
        y: torch.Tensor,
        file_idx: torch.Tensor,
        epoch: int,
        phi: Optional[torch.Tensor] = None,
        y_phi: Optional[torch.Tensor] = None,
        y_flux: Optional[torch.Tensor] = None,
    ) -> List[float]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(epoch_schedule)).sum()
        if not idx:
            if use_potentials:
                return x, phi, ts, y, y_phi, y_flux
            else:
                return x, ts, y

        # sample number of steps
        curr_probs = [p / sum(probs_schedule[:idx]) for p in probs_schedule[:idx]]
        pf_n_unrolls = np.random.choice(n_unrolls_schedule[:idx], p=curr_probs)

        # cap the unroll steps depending on the current max timestep
        n_unrolls = []
        for i, f_idx in enumerate(file_idx.tolist()):
            sleft = (dataset.num_ts(int(f_idx)) - int(ts_idx[i])) // bundle_steps - 1
            n_unrolls.append(min(sleft, pf_n_unrolls))
        n_unrolls = min(n_unrolls)

        if n_unrolls < 2:
            if use_potentials:
                return x, phi, ts, y, y_phi, y_flux
            else:
                return x, ts, y

        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + n_unrolls * ts_step, ts_step))
            for ts in ts_idx.tolist()
        ]
        tsteps = dataset.get_timesteps(file_idx, torch.tensor(ts_idxs))

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(
                file_idx.cpu(),
                ts_unrolled.cpu(),
            )

        executor = ThreadPoolExecutor(max_workers=1)
        with torch.no_grad():
            ts_unrolled = ts_idx + (n_unrolls - 1) * ts_step
            future = executor.submit(fetch_target, dataset, file_idx, ts_unrolled)

            xt = x
            if use_potentials:
                phit = phi
            for i in range(n_unrolls - 1):
                use_bf16 = use_amp and torch.cuda.is_bf16_supported()
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    if use_potentials:
                        x_p, phi_p, _ = model(
                            xt, phit, timestep=tsteps[:, i].to(xt.device), itg=itg
                        )
                    else:
                        x_p = model(xt, timestep=tsteps[:, i].to(xt.device), itg=itg)

                    if predict_delta:
                        x_p = xt + x_p
                    xt = x_p.clone().float()
                    if use_potentials:
                        phit = phi_p.clone().float()
            # Get the result when needed
            unrolled: CycloneSample = future.result()

        # have to clone xt to avoid view mode grad runtime error
        if use_potentials:
            return (
                xt.clone(),
                phit.clone(),
                unrolled.timestep.to(x.device),
                unrolled.y.to(x.device, non_blocking=True),
                unrolled.y_poten.to(x.device, non_blocking=True),
                unrolled.y_flux.to(x.device, non_blocking=True),
            )
        else:
            return (
                xt.clone(),
                unrolled.timestep.to(x.device),
                unrolled.y.to(x.device, non_blocking=True),
            )

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
