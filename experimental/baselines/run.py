from datetime import datetime, timedelta

from functools import partial
import torch
from torch.cuda import max_memory_allocated
from torch.nn.utils import clip_grad_norm_
import warnings
from tqdm import tqdm
from time import perf_counter_ns
from collections import defaultdict
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from dataset import get_data, CycloneSample
from train import relative_norm_mse, pretrain_autoencoder
from utils import (
    load_model_and_config,
    save_model_and_config,
    setup_logging,
)
from eval.plot_utils import plot_potentials

from typing import Dict, Callable, Optional

from einops import rearrange
import torch
from torch import nn
from torch.utils.data import Dataset

from dataset.cyclone import CycloneSample
from models.utils import ContinuousConditionEmbed
from experimental.baselines.models import PhiUnet, QLKNN, FluxMLP, FluxLSTM


def generate_val_plots(ts, phase, phi_rollout=None, y_phi=None):
    plots = {}
    plots[f"Potentials (T={ts[0].item():.2f}, {phase})"] = plot_potentials(
        phi_rollout[0, 0], y_phi[0].cpu()
    )
    return plots


def split_batch_into_phases(phase_change, ts, itg, file_idx, ts_index, phi, y_phi):
    split_idx = torch.searchsorted(ts, phase_change, right=False)
    if split_idx == ts.shape[0]:
        phi_list = [phi]
        y_phi_list = [y_phi]
        ts_list = [ts]
        itg_list = [itg]
        file_idx_list = [file_idx]
        ts_index_list = [ts_index]
        phase_list = ["linear"]
    elif split_idx == 0:
        phi_list = [phi]
        y_phi_list = [y_phi]
        ts_list = [ts]
        itg_list = [itg]
        file_idx_list = [file_idx]
        ts_index_list = [ts_index]
        phase_list = ["saturated"]
    else:
        phi_list = [phi[:split_idx], phi[split_idx:]]
        y_phi_list = [y_phi[:split_idx], y_phi[split_idx:]]
        ts_list = [ts[:split_idx], ts[split_idx:]]
        itg_list = [itg[:split_idx], itg[split_idx:]]
        file_idx_list = [file_idx[:split_idx], file_idx[split_idx:]]
        ts_index_list = [ts_index[:split_idx], ts_index[split_idx:]]
        phase_list = ["linear", "saturated"]
    return (
        ts_list,
        itg_list,
        file_idx_list,
        ts_index_list,
        phase_list,
        phi_list,
        y_phi_list,
    )


def get_rollout_fn(
    n_steps: int,
    bundle_steps: int,
    dataset: Dataset,
    use_amp: bool = False,
    device: str = "cuda",
) -> Callable:
    # correct step size by adding last bundle
    # n_steps_ = n_steps + bundle_steps - 1

    def _rollout(
        model: nn.Module,
        target,
        file_idx: torch.Tensor,
        ts_index_0: torch.Tensor,
        itg: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        timestep_index: Optional[torch.Tensor] = None,
        file_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cap the steps depending on the current max timestep
        rollout_steps = []
        for i, f_idx in enumerate(file_idx.tolist()):
            ts_left = dataset.num_ts(int(f_idx)) - int(ts_index_0[i])
            ts_left = ts_left // bundle_steps - 1
            rollout_steps.append(min(ts_left, n_steps))
        rollout_steps = min(rollout_steps)

        tot_ts = rollout_steps * bundle_steps
        phi_rollout = None
        if target == "phi":
            phit = phi.clone()
            phi_rollout = torch.zeros((phi.shape[0], 1, tot_ts, *phit.shape[2:]))

        # get corresponding timesteps
        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, ts_step))
            for ts in ts_index_0.tolist()
        ]
        fluxes = []
        tsteps = dataset.get_timesteps(file_idx, torch.tensor(ts_idxs))
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i in range(0, rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    if target == "phi":
                        phi_p, flux = model(
                            phit, timestep=tsteps[:, i].to(phi.device), itg=itg
                        )
                        phit = phi_p.clone().float()
                        phi_rollout[:, :, i * bundle_steps : (i + 1) * bundle_steps, ...] = (
                            phi_p.cpu().unsqueeze(2) if phi_p.ndim == 5 else phi_p.cpu()
                        )
                    if target == "flux":
                        flux = model(tsteps[:, i].to(phi.device), itg)
                    if target == "seq_flux":
                        flux_seq = dataset.get_flux_seq(
                            timestep_index.tolist(), file_index.tolist(), window=10
                        )
                        flux_seq = flux_seq.to(device, dtype=itg.dtype)
                        flux = model(flux_seq.unsqueeze(-1), tsteps[:, i].to(phi.device), itg)
                    fluxes.append(flux.cpu())
        if target == "phi":
            phi_rollout = rearrange(phi_rollout, "b c t ... -> t b c ...")
            phi_rollout = phi_rollout[: rollout_steps * bundle_steps, :, ...]

        flux_rollout = rearrange(torch.cat(fluxes, dim=-1), "b t -> t 1 b")

        return phi_rollout, flux_rollout

    return _rollout


def validation_metrics(
    file_idx: torch.Tensor,
    ts_index: torch.Tensor,
    bundle_steps: int,
    dataset,
    metrics_fns: Dict[str, Callable] = None,
    phi_rollout: Optional[torch.Tensor] = None,
    flux_rollout: Optional[torch.Tensor] = None,
    get_normalized: bool = False,
) -> torch.Tensor:
    n_steps = flux_rollout.shape[0]
    assert (
        metrics_fns is not None
    ), "Pleas provide some metrics function for the validation metrics."
    phis = []
    fluxes = []
    for t in range(0, n_steps, bundle_steps):
        sample: CycloneSample = dataset.get_at_time(
            file_idx.long(), (ts_index + t).long(), get_normalized
        )
        phis.append(sample.y_poten)
        fluxes.append(sample.y_flux)
    if bundle_steps == 1:
        phi = torch.stack(phis, dim=0)
        flux = torch.stack(fluxes, dim=0)
    else:
        phi = torch.stack(phis, dim=2)
        flux = torch.stack(fluxes, dim=1)
        phi = rearrange(phi, "b c t ... -> t b c ...")
        flux = rearrange(flux, "b t -> t b")

    metrics = torch.zeros((len(metrics_fns), n_steps))
    for idx, (name, fn) in enumerate(metrics_fns.items()):
        if "phi" in name and phi_rollout is not None:
            value_result = fn(phi_rollout, phi)
        elif "flux" in name and flux_rollout is not None:
            value_result = fn(flux_rollout, flux.unsqueeze(1))
        metrics[idx, ...] = value_result
    return metrics


def get_baseline(cfg, dataset):
    latent_dim = cfg.model.latent_dim
    num_layers = cfg.model.num_layers
    n_cond = cfg.model.swin.timestep_conditioning + cfg.model.swin.itg_conditioning
    target = None

    if cfg.model.name == "mlp":
        model = FluxMLP(latent_dim, n_cond)
        target = "flux"

    if cfg.model.name == "lstm":
        model = FluxLSTM(latent_dim, n_cond, num_layers)
        target = "seq_flux"

    if cfg.model.name == "qlknn":
        n_param_cond = 1  # TODO
        model = QLKNN(n_param_cond)
        target = "avg_flux"

    if cfg.model.name == "phi":
        problem_dim = 1

        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.merging_hidden_ratio
        unmerging_hidden_ratio = cfg.model.swin.unmerging_hidden_ratio
        c_multiplier = cfg.model.swin.c_multiplier
        abs_pe = cfg.model.swin.abs_pe
        patch_skip = cfg.model.swin.patch_skip
        modulation = cfg.model.swin.modulation
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        block_type = cfg.model.swin.block_type

        cond_fn = None
        if n_cond > 0:
            cond_fn = ContinuousConditionEmbed(32, n_cond)

        if cfg.model.bundle_seq_length > 1:
            raise NotImplementedError

        model = PhiUnet(
            dim=latent_dim,
            base_resolution=dataset.phi_resolution,  # TODO
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=problem_dim,
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=0.1,
            abs_pe=abs_pe,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=cond_fn,
            act_fn=act_fn,
            patch_skip=patch_skip,
            modulation=modulation,
            block_type=block_type,
        )

        target = "phi"

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, target


def ddp_setup(rank, world_size):
    init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


def runner(rank, cfg, world_size):
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else "cpu"
    if cfg.use_ddp and world_size > 1:
        ddp_setup(rank, world_size)
        use_ddp = True
    else:
        use_ddp = False

    if not rank:
        data_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
        cfg.logging.run_name = f"{cfg.model.name}_{data_and_time}"
        writer = setup_logging(cfg)
    else:
        writer = None

    # TODO currently only support one resolution for all cyclones
    datasets, dataloaders, augmentations = get_data(cfg)

    if len(datasets) == 3:
        # holdout trajectories and holdout samples for validation
        trainset, valsets = datasets[0], datasets[1:]
        trainloader, valloaders = dataloaders[0], dataloaders[1:]
    else:
        # only holdout trajectories for validation
        trainset, valsets = datasets
        valsets = [valsets]
        trainloader, valloaders = dataloaders
        valloaders = [valloaders]

    trainset.what_to_load = ["phi"]
    valsets[0].what_to_load = ["phi"]

    model, target = get_baseline(cfg, dataset=trainset)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    bundle_seq_length = cfg.model.bundle_seq_length

    opt_state_dict = None
    if cfg.load_ckp is True and cfg.ckpt_path is not None:
        # TODO move config loading to here (now in main.py)
        model, opt_state_dict, _ = load_model_and_config(
            cfg.ckpt_path, model=model, device=device
        )

    if cfg.mode == "train":
        n_epochs = cfg.training.n_epochs
        total_steps = n_epochs * len(trainloader)

        # optimizer config
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        use_amp = cfg.use_amp

        scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        if cfg.training.scheduler is not None:
            scheduler = get_scheduler(
                name=cfg.training.scheduler,
                optimizer=opt,
                num_warmup_steps=total_steps // 6,
                num_training_steps=total_steps,
            )

        if opt_state_dict is not None:
            opt.load_state_dict(opt_state_dict)

        # configure loss
        predict_delta = cfg.training.predict_delta
        loss_val_min = torch.inf

        if cfg.training.pretraining:
            model = pretrain_autoencoder(
                model, cfg, trainloader, valloaders, writer, device
            )  # only valuate on the holdout trajectories, not the holdout samples
            if not hasattr(model, "module") and use_ddp:
                model = DDP(model, device_ids=[rank])

        use_tqdm = cfg.logging.tqdm if not use_ddp else False
        for epoch in range(1, n_epochs + 1):
            train_mse = 0
            train_phi_mse = 0
            train_flux_mse = 0
            model.train()
            info_dict = defaultdict(list)
            t_start_data = perf_counter_ns()

            if use_tqdm or (use_ddp and not rank):
                trainloader = tqdm(trainloader, "Training")
            for i, sample in enumerate(trainloader):
                # reset_peak_memory_stats(device)
                sample: CycloneSample
                if target == "phi":
                    phi = sample.x_poten.to(device, non_blocking=True)
                    y_phi = sample.y_poten.to(device, non_blocking=True)
                y_flux = sample.y_flux.to(device)
                ts = sample.timestep.to(device)
                itg = sample.itg.to(device)

                if augmentations is not None:
                    for aug_fn in augmentations:
                        phi = aug_fn(phi)

                # dataloading timings
                info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

                t_start_fwd = perf_counter_ns()

                with torch.autocast(str(cfg.device), dtype=amp_dtype, enabled=use_amp):
                    phi_loss = torch.tensor(0.0, device=device, dtype=ts.dtype)
                    flux_loss = torch.tensor(0.0, device=device, dtype=ts.dtype)
                    if target == "phi":
                        pred_phi, pred_flux = model(phi, timestep=ts, itg=itg)
                        if predict_delta:
                            pred_phi = phi + pred_phi
                        phi_loss = relative_norm_mse(pred_phi, y_phi)
                    elif target == "flux":
                        pred_flux = model(ts, itg)
                    elif target == "seq_flux":
                        flux_seq = trainset.get_flux_seq(
                            sample.timestep_index.tolist(),
                            sample.file_index.tolist(),
                            window=10,
                        )
                        flux_seq = flux_seq.to(device, dtype=itg.dtype)
                        pred_flux = model(flux_seq.unsqueeze(-1), ts, itg)
                    elif target == "avg_flux":
                        pred_avg_flux = model(ts, itg)
                        pred_flux = None
                        avg_flux = trainset.get_avg_flux(sample.file_index.tolist())
                        avg_flux = avg_flux.to(device, dtype=pred_avg_flux.dtype)
                        flux_loss = relative_norm_mse(
                            pred_avg_flux, avg_flux.unsqueeze(1)
                        )

                    if pred_flux is not None:
                        flux_loss = relative_norm_mse(pred_flux, y_flux.unsqueeze(1))
                    loss = phi_loss + 1e-2 * flux_loss

                # forward timing
                info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
                t_start_bkd = perf_counter_ns()

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if cfg.training.clip_grad:
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if cfg.training.scheduler is not None:
                    scheduler.step()

                train_mse += loss.item()
                train_phi_mse += phi_loss.item()
                train_flux_mse += flux_loss.item()

                info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
                info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
                t_start_data = perf_counter_ns()

            train_mse /= len(trainloader)
            train_phi_mse /= len(trainloader)
            train_losses_dict = {
                "train/relative_norm_mse": train_mse,
                "train/phi_mse": train_phi_mse,
                "train/flux_mse": train_flux_mse,
                "train/lr": (
                    scheduler.get_last_lr()[0]
                    if cfg.training.scheduler
                    else cfg.training.learning_rate
                ),
            }
            info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

            # Validation loop
            n_eval_steps = cfg.validation.n_eval_steps
            val_freq = cfg.validation.validate_every_n_epochs
            tot_eval_steps = n_eval_steps * bundle_seq_length
            denormalize = cfg.validation.denormalize
            log_metric_dict = {}
            val_plots = {}
            if (epoch % val_freq) == 0 or epoch == 1 and not rank:
                # Validation loop
                model.eval()
                if target == "avg_flux":
                    # TODO
                    continue

                for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):

                    rollout_fn = get_rollout_fn(
                        n_steps=n_eval_steps,
                        bundle_steps=bundle_seq_length,
                        dataset=valset,
                        device=str(device),
                        use_amp=use_amp,
                    )

                    valname = (
                        "holdout_trajectories" if val_idx == 0 else "holdout_samples"
                    )
                    # TODO configurable metric list
                    metric_fn_list = {}
                    if target == "phi":
                        metric_fn_list["phi_relative_norm_mse"] = partial(
                            relative_norm_mse, dim_to_keep=0
                        )
                    metric_fn_list["flux_relative_norm_mse"] = partial(
                        relative_norm_mse, dim_to_keep=0
                    )
                    metrics = {
                        "linear": torch.zeros([len(metric_fn_list), tot_eval_steps]),
                        "saturated": torch.zeros([len(metric_fn_list), tot_eval_steps]),
                    }
                    n_timesteps_acc = {
                        "linear": torch.zeros([tot_eval_steps]),
                        "saturated": torch.zeros([tot_eval_steps]),
                    }
                    if use_tqdm or (use_ddp and not rank):
                        valloader = tqdm(
                            valloader,
                            desc=(
                                "Validation holdout trajectories"
                                if val_idx == 0
                                else "Validation holdout samples"
                            ),
                        )
                    with torch.no_grad():
                        for idx, sample in enumerate(valloader):
                            sample: CycloneSample
                            phi = sample.x_poten.to(device, non_blocking=True)
                            y_phi = sample.y_poten.to(device, non_blocking=True)
                            ts = sample.timestep.to(device)
                            itg = sample.itg.to(device)
                            file_idx = sample.file_index.to(device)
                            ts_index = sample.timestep_index.to(device)

                            # TODO: dont hardcode this
                            phase_change = 24
                            (
                                ts_list,
                                itg_list,
                                file_idx_list,
                                ts_index_list,
                                phase_list,
                                phi_list,
                                y_phi_list,
                            ) = split_batch_into_phases(
                                phase_change,
                                ts,
                                itg,
                                file_idx,
                                ts_index,
                                phi=phi,
                                y_phi=y_phi,
                            )

                            # Iterate over the splits
                            for i in range(len(y_phi_list)):
                                phi = phi_list[i]
                                y_phi = y_phi_list[i]
                                ts = ts_list[i]
                                itg = itg_list[i]
                                file_idx = file_idx_list[i]
                                ts_index = ts_index_list[i]
                                phase = phase_list[i]

                                # get the rolled out validation trajectories
                                phi_rollout, flux_rollout = rollout_fn(
                                    model,
                                    target=target,
                                    file_idx=file_idx,
                                    ts_index_0=ts_index,
                                    itg=itg,
                                    phi=phi,
                                    timestep_index=ts_index,
                                    file_index=file_idx,
                                )

                                # TODO: smarter (i.e. use timeindex when we output a dataclass from the dataset)
                                metrics_i = validation_metrics(
                                    file_idx,
                                    ts_index,
                                    bundle_seq_length,
                                    valset,
                                    metrics_fns=metric_fn_list,
                                    phi_rollout=phi_rollout,
                                    flux_rollout=flux_rollout,
                                    get_normalized=not denormalize,
                                )
                                if metrics_i.shape[-1] < tot_eval_steps:
                                    # end of dataset, need to pad the tensor
                                    diff = tot_eval_steps - metrics_i.shape[-1]
                                    metrics_i = torch.cat(
                                        [
                                            metrics_i,
                                            torch.zeros(
                                                [metrics_i.shape[0], diff],
                                                dtype=metrics_i.dtype,
                                            ),
                                        ],
                                        dim=-1,
                                    )
                                    metrics[phase] += metrics_i
                                    validated_steps = (
                                        torch.arange(1, tot_eval_steps + 1)
                                        <= metrics_i.shape[-1]
                                    ).to(dtype=metrics_i.dtype)
                                else:
                                    metrics[phase] += metrics_i
                                    validated_steps = torch.ones([tot_eval_steps])
                                n_timesteps_acc[phase] += validated_steps
                                
                                if target == "phi":
                                    if val_idx == 0:
                                        # holdout trajectories valset
                                        if idx in [0, 20]:
                                            plots = generate_val_plots(
                                                ts=ts,
                                                phase=(
                                                    "Linear Phase"
                                                    if idx == 0
                                                    else "Saturated Phase"
                                                ),
                                                phi_rollout=phi_rollout,
                                                y_phi=y_phi,
                                            )
                                            val_plots.update(plots)
                                    else:
                                        # holdout samples valset
                                        if idx == 0:
                                            plots = generate_val_plots(
                                                ts=ts,
                                                phase="Holdout samples",
                                                phi_rollout=phi_rollout,
                                                y_phi=y_phi,
                                            )
                                            val_plots.update(plots)

                    for key in metrics.keys():
                        metrics[key] = metrics[key] / n_timesteps_acc[key].unsqueeze(0)

                    for idx, metric_name in enumerate(metric_fn_list):
                        for phase, metric in metrics.items():
                            vals = metric[idx, ...]
                            for t in range(tot_eval_steps):
                                log_metric_dict[
                                    f"val_{valname}/{metric_name}_{phase}_x{t + 1}"
                                ] = vals[t]

                    if val_idx == 0:
                        # trajectoy validation
                        n_timesteps_acc_model_saving = n_timesteps_acc

                if cfg.ckpt_path is not None and not rank:
                    if target == "phi":
                        mse_sat = log_metric_dict[
                            "val_holdout_trajectories/phi_relative_norm_mse_saturated_x1"
                        ]
                        mse_lin = log_metric_dict[
                            "val_holdout_trajectories/phi_relative_norm_mse_linear_x1"
                        ]
                    else:
                        mse_sat = log_metric_dict[
                            "val_holdout_trajectories/flux_relative_norm_mse_saturated_x1"
                        ]
                        mse_lin = log_metric_dict[
                            "val_holdout_trajectories/flux_relative_norm_mse_linear_x1"
                        ]
                    sat_ts = n_timesteps_acc_model_saving["saturated"]
                    lin_ts = n_timesteps_acc_model_saving["linear"]
                    val_loss = (mse_sat * sat_ts + mse_lin * lin_ts) / (sat_ts + lin_ts)
                    # Save model if validation loss on trajectories improves
                    loss_val_min = save_model_and_config(
                        model,
                        optimizer=opt,
                        cfg=cfg,
                        epoch=epoch,
                        # TODO decide target metric
                        val_loss=val_loss.mean(),
                        loss_val_min=loss_val_min,
                    )
                else:
                    warnings.warn(
                        "`cfg.ckpt_path` is not set: checkpoints will not be stored"
                    )

            if use_ddp:
                # Add barrier for all ranks and wait for rank 0
                torch.distributed.barrier()
            # log to wandb
            epoch_logs = train_losses_dict | log_metric_dict
            if writer and not rank:
                wandb_logs = epoch_logs | info_dict
                # log epoch details
                if not val_plots:
                    writer.log(wandb_logs)
                else:
                    writer.log(wandb_logs, commit=False)
                    writer.log(val_plots)

            epoch_str = str(epoch).zfill(len(str(int(n_epochs))))
            total_time = (
                info_dict["info/forward_ms"]
                + info_dict["info/backward_ms"]
                + info_dict["info/data_ms"]
            )
            if not rank:
                print(
                    f"Epoch: {epoch_str}, "
                    f"{', '.join([f'{k}: {v:.5f}' for k, v in epoch_logs.items()])}"
                    f", step time: {total_time:.2f}ms"
                )
        if writer:
            writer.finish()

    if cfg.mode == "rollout":
        raise NotImplementedError("TODO")
