import os
from tqdm import tqdm
from time import perf_counter_ns
from datetime import timedelta

import torch
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from collections import defaultdict
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils._pytree import tree_map

from neugk.dataset import get_data, CycloneSample
from neugk.models import get_model
from neugk.train import get_pushforward_fn, LossWrapper, GradientBalancer
from neugk.eval.evaluate import evaluate
from neugk.utils import (
    load_model_and_config,
    setup_logging,
    edit_tag,
    get_linear_burn_in_fn,
    remainig_progress,
    exclude_from_weight_decay,
)


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


def runner(rank, cfg, world_size):
    if cfg.ddp.enable and cfg.ddp.n_nodes > 1 and world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = rank
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu"
    if cfg.ddp.enable and world_size > 1:
        ddp_setup(rank, world_size)
        use_ddp = True
    else:
        use_ddp = False

    if not rank:
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
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    bundle_seq_length = cfg.model.bundle_seq_length
    n_epochs = cfg.training.n_epochs
    total_steps = n_epochs * len(trainloader)

    if cfg.training.exclude_from_weight_decay is not None:
        params = exclude_from_weight_decay(
            model,
            cfg.training.exclude_from_weight_decay,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        params = model.parameters()

    # optimizer config
    opt = torch.optim.Adam(
        params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = cfg.amp.enable
    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
    use_bf16 = use_amp and cfg.amp.bfloat and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if cfg.training.scheduler is not None:
        scheduler = get_scheduler(
            name=cfg.training.scheduler,
            optimizer=opt,
            num_warmup_steps=total_steps // 6,
            num_training_steps=total_steps,
        )

    loss_scheduler_dict = {}
    weights = dict(cfg.model.loss_weights) | dict(cfg.model.extra_loss_weights)
    for key in weights.keys():
        if cfg.model.loss_scheduler[key]:
            scheduler_params = getattr(cfg.model.loss_scheduler, key)
            loss_scheduler_dict[key] = get_linear_burn_in_fn(
                scheduler_params.start,
                end=scheduler_params.end,
                start_fraction=scheduler_params.start_fraction,
                end_fraction=scheduler_params.end_fraction,
            )
    # configure loss
    predict_delta = cfg.training.predict_delta
    loss_wrap = LossWrapper(
        weights=weights,
        schedulers=loss_scheduler_dict,
        denormalize_fn=trainset.denormalize,
        separate_zf=cfg.dataset.separate_zf,
        real_potens=cfg.dataset.real_potens,
    )
    grad_balancer = GradientBalancer(
        opt,
        mode=cfg.training.gradnorm_balancer,
        scaler=scaler,
        clip_grad=cfg.training.clip_grad,
        n_tasks=len(weights),
    )
    # and pushforward
    pf_cfg = cfg.training.pushforward
    pushforward_fn = None
    if sum(pf_cfg.unrolls) > 0:
        pushforward_fn = get_pushforward_fn(
            n_unrolls_schedule=pf_cfg.unrolls,
            probs_schedule=pf_cfg.probs,
            epoch_schedule=pf_cfg.epochs,
            predict_delta=predict_delta,
            dataset=trainset,
            bundle_steps=bundle_seq_length,
            use_amp=use_amp,
            use_bf16=use_bf16,
            device=device,
        )

    input_fields = set(cfg.dataset.input_fields)
    output_fields = list(
        (set(cfg.model.loss_weights.keys())).union(
            [k.split("_")[0] for k in cfg.model.extra_loss_weights.keys()]
        )
    )
    conditioning = cfg.model.conditioning
    idx_keys = ["file_index", "timestep_index"]
    use_tqdm = cfg.logging.tqdm if not use_ddp else False

    if cfg.load_ckpt:
        # choosing best.pt since ckpt.pt does not contain scheduler sd
        ckpt_path = os.path.join(cfg.output_path, "best.pth")
        model, ckpt_dict = load_model_and_config(
            ckpt_path, model, device, for_ddp=use_ddp
        )
        opt.load_state_dict(ckpt_dict["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
        start_epoch = ckpt_dict["epoch"]
        loss_val_min = ckpt_dict["loss"]
        cur_update_step = start_epoch * len(trainloader)
    else:
        loss_val_min = torch.inf
        cur_update_step = 0.0
        start_epoch = 0

    for epoch in range(start_epoch + 1, n_epochs + 1):
        loss_logs = {k: 0 for k in loss_wrap.active_losses}
        loss_logs["relative_norm"] = 0.0
        model.train()
        loss_wrap.train().to(device)
        info_dict = defaultdict(list)
        t_start_data = perf_counter_ns()

        if use_tqdm or (use_ddp and not rank):
            trainloader = tqdm(trainloader, "Training")

        ############################# train loop start #############################

        for _, sample in enumerate(trainloader):
            try:
                reset_peak_memory_stats(device)
            except:
                pass  # only works with cuda device
            sample: CycloneSample
            inputs = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in input_fields
                if getattr(sample, k) is not None
            }
            gts = {
                k: getattr(sample, f"y_{k}").to(device, non_blocking=True)
                for k in output_fields
                if getattr(sample, f"y_{k}") is not None
            }
            conds = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in conditioning
                if getattr(sample, k) is not None
            }
            idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}
            geometry = tree_map(lambda g: g.to(device), sample.geometry)

            if augmentations is not None:
                for aug_fn in augmentations:
                    inputs = {k: aug_fn(v) for k, v in inputs.items()}

            # dataloading timings
            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

            if pushforward_fn:
                start_pf = perf_counter_ns()
                # accessory information for pf (to retreive unrolled target)
                inputs, gts, conds = pushforward_fn(
                    model,
                    inputs,
                    gts,
                    conds,
                    idx_data,
                    epoch,
                )
                info_dict["pf_ms"].append((perf_counter_ns() - start_pf) / 1e6)
            else:
                info_dict["pf_ms"].append(0.0)

            t_start_fwd = perf_counter_ns()
            with torch.autocast(str(device), dtype=amp_dtype, enabled=use_amp):
                # model prediction
                preds = model(**inputs, **conds)
                # predict residuals
                if predict_delta:
                    for key in cfg.dataset.input_fields:
                        preds[key] = preds[key] + inputs[key]

                # compute losses
                progress_remaining = remainig_progress(cur_update_step, total_steps)
                loss, losses = loss_wrap(
                    preds,
                    gts,
                    idx_data,
                    geometry=geometry,
                    progress_remaining=progress_remaining,
                    separate_zf=(
                        cfg.dataset.separate_zf if cfg.model.extra_zf_loss else False
                    ),
                )

            # forward timing
            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

            # backward pass (+optional gradnorm for multitask)
            model = grad_balancer(model, loss, list(losses.values()))
            # lr scheduler step
            if cfg.training.scheduler is not None:
                scheduler.step()

            cur_update_step += 1.0
            for k in loss_wrap.active_losses:
                if k not in loss_logs:
                    # if schedulers start from zero
                    loss_logs[k] = losses[k]
                else:
                    loss_logs[k] += losses[k].item()
            loss_logs["relative_norm"] += loss.item()
            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
            t_start_data = perf_counter_ns()

        ############################## train loop end ##############################

        for k in loss_logs:
            loss_logs[k] /= len(trainloader)
        loss_logs["relative_norm"] /= len(trainloader)
        # logging loss tags (for wandb)
        loss_logs = edit_tag(loss_logs, prefix="train", postfix="mse")
        train_losses_dict = {
            "train/lr": (
                scheduler.get_last_lr()[0]
                if cfg.training.scheduler
                else cfg.training.learning_rate
            ),
        }
        for key in loss_scheduler_dict.keys():
            train_losses_dict[f"train/{key}_schedule"] = loss_scheduler_dict[key](
                progress_remaining
            )
        train_losses_dict = train_losses_dict | loss_logs
        info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

        ############################# evaluation start #############################

        log_metric_dict, val_plots, loss_val_min = evaluate(
            rank,
            world_size,
            model,
            loss_wrap,
            valsets,
            valloaders,
            opt,
            scheduler,
            epoch,
            cfg,
            device,
            loss_val_min,
        )

        ############################## evaluation end ##############################

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

    if use_ddp:
        dist.destroy_process_group()
