from datetime import datetime

from functools import partial
import torch
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.nn.utils import clip_grad_norm_
import warnings
from tqdm import tqdm
from time import perf_counter_ns
from collections import defaultdict
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from dataset import get_data, CycloneSample
from models import get_model
from train import get_pushforward_trick, relative_norm_mse, pretrain_autoencoder
from eval import (
    get_rollout,
    validation_metrics,
    generate_val_plots,
    to_fourier,
)
from utils import load_model_and_config, save_model_and_config, setup_logging


def ddp_setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


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

    model = get_model(cfg, dataset=trainset)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])
    active_keys = cfg.dataset.active_keys

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

        scaler = torch.amp.GradScaler(device=device, enabled=cfg.use_amp)
        use_bf16 = cfg.use_amp and torch.cuda.is_bf16_supported()
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
        if sum(pf_cfg.unrolls) > 0:
            pushforward_fn = get_pushforward_trick(
                pf_cfg.unrolls,
                pf_cfg.probs,
                schedule=pf_cfg.epochs,
                predict_delta=predict_delta,
                dataset=trainset,
                bundle_steps=cfg.model.bundle_seq_length,
                use_amp=cfg.use_amp,
            )

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
            model.train()
            info_dict = defaultdict(list)
            t_start_data = perf_counter_ns()

            if use_tqdm or (use_ddp and not rank):
                trainloader = tqdm(trainloader, "Training")
            for i, sample in enumerate(trainloader):
                # if i > 0:
                #     break
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
                with torch.autocast(
                    str(cfg.device),
                    dtype=torch.float16 if not use_bf16 else torch.bfloat16,
                    enabled=cfg.use_amp,
                ):
                    # TODO: currently only supporting integer conditioning, therefore ceiling the actual float timestep
                    pred_x = model(x, timestep=ts, itg=itg)
                    if predict_delta:
                        pred_x = x + pred_x
                    loss = relative_norm_mse(pred_x, y)

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

                info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
                info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
                t_start_data = perf_counter_ns()

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
            log_metric_dict = {}
            val_plots = {}
            if (
                (epoch % cfg.validation.validate_every_n_epochs) == 0 or epoch == 1
            ) and not rank:
                # Validation loop
                model.eval()
                for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):
                    valname = (
                        "holdout_trajectories" if val_idx == 0 else "holdout_samples"
                    )
                    # if val_idx == 0:
                    #     continue
                    # TODO configurable metric list
                    metric_fn_list = {
                        "relative_norm_mse": partial(
                            relative_norm_mse, dim_to_keep=0
                        ),  # to average across all dimensions except timesteps
                        # TODO: add more useful metrics
                    }
                    n_eval_steps = cfg.validation.n_eval_steps
                    metrics = torch.zeros(
                        [
                            len(metric_fn_list),
                            n_eval_steps * cfg.model.bundle_seq_length,
                        ]
                    )  # shape [n_metrics, n_timesteps]
                    n_timesteps_count = torch.zeros(
                        [n_eval_steps * cfg.model.bundle_seq_length]
                    )
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

                            # get the rolled out validation trajectories
                            x_rollout = get_rollout(
                                problem_dim=len(active_keys),
                                n_steps=n_eval_steps,
                                bundle_steps=cfg.model.bundle_seq_length,
                                dataset=valset,
                                predict_delta=cfg.training.predict_delta,
                                device=str(device),
                                use_amp=cfg.use_amp,
                            )(model, x, file_idx=file_idx, ts_index_0=ts_index, itg=itg)

                            # back to fourier for plotting
                            if cfg.dataset.spatial_ifft and cfg.dataset.in_memory:
                                # TODO move somewhere else
                                x_rollout, y = to_fourier(x_rollout, y)

                            # TODO: make smarter (i.e. use timeindex when we output a dataclass from the dataset)
                            # metrics tensor will have shape [number_of_metrics, n_timesteps]
                            metrics_i = validation_metrics(
                                x_rollout,
                                file_idx,
                                ts_index,
                                cfg.model.bundle_seq_length,
                                valset,
                                metric_fn_list,
                            )

                            if (
                                metrics_i.shape[-1]
                                < n_eval_steps * cfg.model.bundle_seq_length
                            ):
                                # reached end of dataset at some point so we need to pad the tensor
                                diff = (
                                    n_eval_steps * cfg.model.bundle_seq_length
                                    - metrics_i.shape[-1]
                                )
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
                                metrics += metrics_i
                                validated_steps = (
                                    torch.arange(
                                        1,
                                        n_eval_steps * cfg.model.bundle_seq_length + 1,
                                    )
                                    <= metrics_i.shape[-1]
                                )
                                validated_steps = validated_steps.to(
                                    dtype=metrics_i.dtype
                                )
                            else:
                                metrics += metrics_i
                                validated_steps = torch.ones(
                                    [n_eval_steps * cfg.model.bundle_seq_length]
                                )
                            n_timesteps_count += (
                                validated_steps  # shape: [n_eval_steps]
                            )

                            if val_idx == 0:
                                # holdout trajectories valset
                                if idx in [0, 5]:
                                    generate_val_plots(
                                        x_rollout=x_rollout,
                                        y=y,
                                        ts=ts,
                                        phase=(
                                            "Linear Phase"
                                            if idx == 0
                                            else "Saturated Phase"
                                        ),
                                        val_plots=val_plots,
                                    )
                            else:
                                # holdout samples valset
                                if idx == 0:
                                    generate_val_plots(
                                        x_rollout=x_rollout,
                                        y=y,
                                        ts=ts,
                                        phase="Holdout samples",
                                        val_plots=val_plots,
                                    )

                    metrics = metrics / n_timesteps_count.unsqueeze(0)

                    for idx, metric_name in enumerate(metric_fn_list):
                        vals = metrics[idx, ...]
                        for t in range(n_eval_steps * cfg.model.bundle_seq_length):
                            log_metric_dict[
                                f"val_{valname}/" + metric_name + f"x{t + 1}"
                            ] = vals[t]

                if cfg.ckpt_path is not None and not rank:
                    # Save model if validation loss improves
                    save_model_and_config(
                        model,
                        optimizer=opt,
                        cfg=cfg,
                        epoch=epoch,
                        # TODO decide target metric
                        val_loss=log_metric_dict[
                            "val_holdout_trajectories/relative_norm_msex1"
                        ],  # current metric only on hold out trajectories (not holdout samples)
                        loss_val_min=loss_val_min,
                    )
                else:
                    warnings.warn(
                        "`cfg.ckpt_path` is not set: checkpoints will not be stored"
                    )

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
                    f", epoch time: {total_time:.2f}ms"
                )
        if writer:
            writer.finish()

    if cfg.mode == "rollout":
        raise NotImplementedError("TODO")
