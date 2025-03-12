from datetime import datetime, timedelta

from functools import partial
import torch
import gc
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.nn.utils import clip_grad_norm_
import warnings
from tqdm import tqdm
from time import perf_counter_ns
from collections import defaultdict
from diffusers.schedulers import DDPMScheduler
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
from collections import OrderedDict
from copy import deepcopy

from dataset import get_data, CycloneSample
from models import get_model
from train import get_pushforward_fn, relative_norm_mse, pretrain_autoencoder
from eval import get_rollout_fn, validation_metrics, generate_val_plots, get_flux_plot
from eval.gkw_client import request_gkw_sim, dump_rollout
from utils import (
    load_model_and_config,
    save_model_and_config,
    setup_logging,
    split_batch_into_phases,
)


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.995):
    ema_params = OrderedDict(ema_model.named_parameters())
    if hasattr(model, "module"):
        model_params = OrderedDict(model.module.named_parameters())
    else:
        model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


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

    problem_dim = len(trainset.active_keys)
    model = get_model(cfg, dataset=trainset)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])
    bundle_seq_length = cfg.model.bundle_seq_length

    use_gkw = cfg.validation.use_gkw
    if use_gkw:
        gkw_executor = ThreadPoolExecutor(max_workers=1)
        # TODO hardcoded for now
        gkw_dump_path = "/system/user/publicdata/gyrokinetics/dumps/test_gkw_client"
        gkw_futures = {}

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
        # and pushforward
        pf_cfg = cfg.training.pushforward
        pushforward_fn = None
        # if sum(pf_cfg.unrolls) > 0:
        #     pushforward_fn = get_pushforward_fn(
        #         n_unrolls_schedule=pf_cfg.unrolls,
        #         probs_schedule=pf_cfg.probs,
        #         epoch_schedule=pf_cfg.epochs,
        #         predict_delta=predict_delta,
        #         dataset=trainset,
        #         bundle_steps=bundle_seq_length,
        #         use_amp=use_amp,
        #     )

        loss_val_min = torch.inf

        if cfg.training.pretraining:
            model = pretrain_autoencoder(
                model, cfg, trainloader, valloaders, writer, device
            )  # only valuate on the holdout trajectories, not the holdout samples
            if not hasattr(model, "module") and use_ddp:
                model = DDP(model, device_ids=[rank])

        # refiner params
        min_noise_std = 1e-5
        num_refinement_steps = 2

        betas = [
            (min_noise_std ** (k / num_refinement_steps)) ** (1 / 3)
            for k in reversed(range(num_refinement_steps + 1))
        ]

        ddpm = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )

        ema = deepcopy(model).to(device)
        if use_ddp:
            ema = ema.module
        requires_grad(ema, False)
        update_ema(ema, model, decay=0)
        ema.eval()

        # # noise mask
        # import pickle
        # import numpy as np

        # big_std = pickle.load(open("notebooks/big_std.pkl", "rb"))
        # noise_mask = big_std > np.quantile(big_std, 0.70)
        # noise_mask = torch.tensor(noise_mask, device=device, requires_grad=False).unsqueeze(0)

        num_train_timesteps = ddpm.config.num_train_timesteps

        # TODO to use refiner set one more conditioning and channels * 2 in the model

        def refine_step(model_fn, x, y, ts, itg):
            if predict_delta:
                y = y - x
            k = torch.randint(0, num_train_timesteps, (x.shape[0],), device=x.device)
            sigma = ddpm.alphas_cumprod.to(x.device)[k]
            noise_factor = sigma.view(-1, *[1 for _ in range(x.ndim - 1)])
            signal_factor = 1 - noise_factor
            noise = torch.randn_like(y)
            # noise = noise * noise_mask
            y_noised = ddpm.add_noise(y, noise, k)
            if x.ndim > y_noised.ndim:
                y_noised = y_noised.unsqueeze(2)
            x_in = torch.cat([x, y_noised], axis=1)
            target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
            pred = model_fn(x_in, timestep=ts, refinement_step=k, itg=itg)
            return relative_norm_mse(pred, target)

        @torch.no_grad()
        def predict(model_fn, x, **kwargs):
            y_noised = torch.randn_like(x, dtype=x.dtype, device=x.device)
            for k_i in ddpm.timesteps:
                k = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device) + k_i
                x_in = torch.cat([x, y_noised], axis=1)
                pred = model_fn(x_in, refinement_step=k, **kwargs)
                y_noised = ddpm.step(pred, k_i, y_noised).prev_sample
            # NOTE: predict_delta already included in rollout
            return y_noised

        use_tqdm = cfg.logging.tqdm if not use_ddp else False

        for epoch in range(1, n_epochs + 1):
            train_mse = 0
            model.train()
            info_dict = defaultdict(list)
            t_start_data = perf_counter_ns()

            if use_tqdm or (use_ddp and not rank):
                trainloader = tqdm(trainloader, "Training")
            for i, sample in enumerate(trainloader):
                reset_peak_memory_stats(device)
                sample: CycloneSample
                x = sample.x.to(device, non_blocking=True)
                y = sample.y.to(device, non_blocking=True)
                ts = sample.timestep.to(device)
                itg = sample.itg.to(device)

                # TODO should augmentations take place before moving to GPU?
                if augmentations is not None:
                    for aug_fn in augmentations:
                        x = aug_fn(x)

                # dataloading timings
                info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

                if pushforward_fn:
                    start_pf = perf_counter_ns()
                    # accessory information for pf (to retreive unrolled target)
                    file_idx = sample.file_index.to(device)
                    ts_index = sample.timestep_index.to(device)
                    x, ts, y = pushforward_fn(
                        model, x, ts, itg, ts_index, y, file_idx, epoch
                    )
                    info_dict["pf_ms"].append((perf_counter_ns() - start_pf) / 1e6)
                else:
                    info_dict["pf_ms"].append(0.0)

                t_start_fwd = perf_counter_ns()

                with torch.autocast(str(device), dtype=amp_dtype, enabled=use_amp):
                    model_fn = model
                    loss = refine_step(model_fn, x, y, ts, itg)

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

                del x
                del y
                del loss
                gc.collect()
                torch.cuda.empty_cache()

                info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
                info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
                t_start_data = perf_counter_ns()

                update_ema(ema, model)

            train_mse /= len(trainloader)
            train_losses_dict = {
                "train/relative_norm_mse": train_mse,
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
            gkw_kfiles = []
            val_plots = {}
            if (epoch % val_freq) == 0 or epoch == 1:
                # Validation loop
                ema.eval()
                for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):

                    rollout_fn = get_rollout_fn(
                        problem_dim=problem_dim,
                        n_steps=n_eval_steps,
                        bundle_steps=bundle_seq_length,
                        dataset=valset,
                        predict_delta=predict_delta,
                        device=str(device),
                        use_amp=use_amp,
                    )

                    valname = (
                        "holdout_trajectories" if val_idx == 0 else "holdout_samples"
                    )
                    # TODO configurable metric list
                    metric_fn_list = {
                        # NOTE: average across all dimensions except timesteps
                        "relative_norm_mse": partial(relative_norm_mse, dim_to_keep=0),
                    }
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
                            x = sample.x.to(device, non_blocking=True)
                            y = sample.y.to(device, non_blocking=True)
                            ts = sample.timestep.to(device)
                            itg = sample.itg.to(device)
                            file_idx = sample.file_index.to(device)
                            ts_index = sample.timestep_index.to(device)

                            # TODO: dont hardcode this
                            phase_change = 24
                            (
                                x_list,
                                y_list,
                                ts_list,
                                itg_list,
                                file_idx_list,
                                ts_index_list,
                                phase_list,
                                _,
                                _,
                            ) = split_batch_into_phases(
                                phase_change, x, y, ts, itg, file_idx, ts_index
                            )

                            # Iterate over the splits
                            for i in range(len(x_list)):
                                x = x_list[i]
                                y = y_list[i]
                                ts = ts_list[i]
                                itg = itg_list[i]
                                file_idx = file_idx_list[i]
                                ts_index = ts_index_list[i]
                                phase = phase_list[i]

                                pred_fn = partial(predict, ema)
                                # get the rolled out validation trajectories
                                x_rollout = rollout_fn(
                                    pred_fn,
                                    x,
                                    file_idx=file_idx,
                                    ts_index_0=ts_index,
                                    itg=itg,
                                )

                                if denormalize:
                                    # denormalize rollout and target for evaluation / plots
                                    x_rollout = torch.stack(
                                        [
                                            torch.stack(
                                                [
                                                    valset.denormalize(
                                                        x_rollout[t, b], f
                                                    )
                                                    for b, f in enumerate(
                                                        file_idx.tolist()
                                                    )
                                                ]
                                            )
                                            for t in range(x_rollout.shape[0])
                                        ]
                                    )
                                    y = torch.stack(
                                        [
                                            valset.denormalize(y[b], f)
                                            for b, f in enumerate(file_idx.tolist())
                                        ]
                                    )

                                # TODO: smarter (i.e. use timeindex when we output a dataclass from the dataset)
                                metrics_i = validation_metrics(
                                    x_rollout,
                                    file_idx,
                                    ts_index,
                                    bundle_seq_length,
                                    valset,
                                    metric_fn_list,
                                    get_normalized=not denormalize,
                                )
                                if use_gkw:
                                    # onestep predictions (time dimension = 0)
                                    rollout_dict = {
                                        int(t_idx.item()): x_rollout[0, b].cpu().numpy()
                                        for b, t_idx in enumerate(ts_index)
                                    }
                                    # TODO put original file in config?
                                    src_config_path = (
                                        valset.files[0]
                                        .split("/")[-1]
                                        .replace("_ifft", "")
                                        .replace("_separate_zf", "")
                                        .replace(".h5", "")
                                    )
                                    src_config_path = (
                                        f"/restricteddata/ukaea/"
                                        f"gyrokinetics/raw/{src_config_path}"
                                    )
                                    gkw_kfiles.extend(
                                        dump_rollout(
                                            rollout_dict, gkw_dump_path, src_config_path
                                        )
                                    )

                                if metrics_i.shape[-1] < tot_eval_steps:
                                    # reached end of dataset at some point so we need to pad the tensor
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

                                if val_idx == 0:
                                    # holdout trajectories valset
                                    if idx in [0, 20]:
                                        plots = generate_val_plots(
                                            x_rollout=x_rollout,
                                            y=y,
                                            ts=ts,
                                            phase=(
                                                "Linear Phase"
                                                if idx == 0
                                                else "Saturated Phase"
                                            ),
                                        )
                                        val_plots.update(plots)
                                else:
                                    # holdout samples valset
                                    if idx == 0:
                                        plots = generate_val_plots(
                                            x_rollout=x_rollout,
                                            y=y,
                                            ts=ts,
                                            phase="Holdout samples",
                                        )
                                        val_plots.update(plots)
                    if dist.is_initialized():
                        for key in metrics.keys():
                            cur_metric = metrics[key].to(device)
                            cur_ts = n_timesteps_acc[key].to(device)
                            gathered = [
                                torch.zeros_like(
                                    cur_metric,
                                    dtype=cur_metric.dtype,
                                    device=cur_metric.device,
                                )
                                for _ in range(world_size)
                            ]
                            gathered_ts = [
                                torch.zeros_like(
                                    cur_metric,
                                    dtype=cur_metric.dtype,
                                    device=cur_metric.device,
                                )
                                for _ in range(world_size)
                            ]
                            dist.all_gather(gathered, cur_metric)
                            dist.all_gather(gathered_ts, cur_ts)
                            metrics[key] = gathered
                            n_timesteps_acc[key] = gathered_ts

                        # TODO: fix all_gather_object for val_plots

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

                if use_gkw:
                    # run gkw on accumulated rollout
                    gkw_futures[epoch] = gkw_executor.submit(
                        request_gkw_sim, gkw_dump_path, gkw_kfiles, cfg.logging.run_id
                    )

                if cfg.ckpt_path is not None and not rank:
                    mse_sat = log_metric_dict[
                        "val_holdout_trajectories/relative_norm_mse_saturated_x1"
                    ]
                    mse_lin = log_metric_dict[
                        "val_holdout_trajectories/relative_norm_mse_linear_x1"
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

            # check gkw future once a epoch
            if use_gkw:
                fluxes, potentials = {}, {}
                done_futures = []
                for future_epoch, fut in gkw_futures.items():
                    if fut.done():
                        done_futures.append(future_epoch)
                        fluxes[future_epoch], potentials[future_epoch] = fut.result()
                        flux_plot, flux_error = get_flux_plot(
                            fluxes[future_epoch], dataset=valset
                        )
                        gkw_logs = {
                            "Flux": flux_plot,
                            "val_holdout_trajectories/flux_mse": flux_error,
                        }
                        if writer and not rank:
                            # NOTE not possible to log at previous step...
                            writer.log(gkw_logs)
                # close finished futures
                for k in done_futures:
                    del gkw_futures[k]
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
                + info_dict["info/pf_ms"]
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
