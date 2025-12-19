from typing import Optional, Tuple, Dict

from datetime import timedelta
from time import perf_counter_ns
from collections import defaultdict
import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils._pytree import tree_map
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dataset import get_data, CycloneAESample
from neugk.pinc.autoencoders import get_autoencoder
from neugk.gyroswin.losses import LossWrapper, GradientBalancer

from neugk.pinc.autoencoders.ae_eval import evaluate
from neugk.pinc.autoencoders.ae_utils import (
    aggregate_dataset_stats,
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    load_autoencoder,
    train_step_autoencoder,
    train_step_peft,
)
from neugk.pinc.peft_utils import setup_peft_stage
from neugk.utils import (
    setup_logging,
    get_linear_burn_in_fn,
    remainig_progress,
    memory_cleanup,
    exclude_from_weight_decay,
)


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


def pinc_ae_runner(rank, cfg, world_size):
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

    # dataset
    datasets, dataloaders, augmentations = get_data(cfg, rank=rank)
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

    model_cfg = cfg.autoencoder
    use_amp = cfg.amp.enable
    n_epochs = cfg.training.n_epochs
    total_steps = n_epochs * len(trainloader)

    loss_scheduler_dict = {}
    weights = dict(model_cfg.loss_weights) | dict(model_cfg.extra_loss_weights)
    for key in weights.keys():
        start = weights[key]
        if (
            start > 0.0
            and key in model_cfg.loss_scheduler
            and model_cfg.loss_scheduler[key]
        ):
            scheduler_params = getattr(model_cfg.loss_scheduler, key)
            loss_scheduler_dict[key] = get_linear_burn_in_fn(
                scheduler_params.start,
                end=scheduler_params.end,
                start_fraction=scheduler_params.start_fraction,
                end_fraction=scheduler_params.end_fraction,
            )

    # load dataset statistics for integral loss normalization
    dataset_stats = {}
    if hasattr(trainset, "files") and trainset.files:
        # aggregate statistics across ALL training files
        dataset_stats = aggregate_dataset_stats(trainset.files)
    elif not rank:
        print(
            "Warning: Could not load dataset statistics - dataset files not available"
        )

    # configure loss
    loss_wrap = LossWrapper(
        weights=weights,
        schedulers=loss_scheduler_dict,
        denormalize_fn=trainset.denormalize,
        separate_zf=cfg.dataset.separate_zf,
        real_potens=cfg.dataset.real_potens,
        integral_loss_type=getattr(cfg.training, "integral_loss_type", "mse"),
        spectral_loss_type=getattr(cfg.training, "spectral_loss_type", "l1"),
        dataset_stats=dataset_stats,
        ds=getattr(cfg.training, "ds", None),
        ema_normalization_loss=getattr(cfg.training, "ema_normalization_loss", None),
        ema_beta=getattr(cfg.training, "ema_beta", 0.99),
        eval_loss_type=getattr(cfg.training, "eval_loss_type", "mse"),
        eval_integral_loss_type=getattr(cfg.training, "eval_integral_loss_type", "mse"),
        eval_spectral_loss_type=getattr(cfg.training, "eval_spectral_loss_type", "l1"),
    )

    models = get_autoencoder(cfg, dataset=trainset, stage=cfg.stage, rank=rank)

    # PEFT checkpointing
    is_peft_checkpoint = False
    ae_ckpt_dict = {}

    if cfg.stage == "autoencoder":
        ae = models
        ae = ae.to(device)
        if use_ddp:
            ae = DDP(ae, device_ids=[rank])  # , find_unused_parameters=True)
        # optimizer config
        if hasattr(cfg.training, "optimizer") and cfg.training.optimizer == "muon":

            def separate_parameters(model, exclude_from_wd=[]):
                """Separate parameters between Muon (>=2D tensors) vs AdamW (1D tensors, embeddings,...)
                Also separate AdamW parameters into those with and without weight decay.
                """
                muon_params = []
                adamw_decay_params = []
                adamw_no_decay_params = []

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue

                    # Check if >=2D parameter in the main body (not embedding/head layers)
                    if param.ndim >= 2 and not any(
                        exclude in name.lower()
                        for exclude in ["embed", "head", "pos_embed", "cls_token"]
                    ):
                        muon_params.append(param)
                    else:
                        # AdamW - check if it should be excluded from weight decay
                        if exclude_from_wd != []:
                            should_exclude = any(
                                exclude_name in name.lower()
                                for exclude_name in exclude_from_wd
                            )
                            if should_exclude:
                                adamw_no_decay_params.append(param)
                            else:
                                adamw_decay_params.append(param)
                        else:
                            adamw_decay_params.append(param)

                return muon_params, adamw_decay_params, adamw_no_decay_params

            exclude_params = getattr(cfg.training, "exclude_from_wd", [])
            muon_params, adamw_decay_params, adamw_no_decay_params = (
                separate_parameters(ae, exclude_params)
            )

            if rank == 0:
                total_adamw = len(adamw_decay_params) + len(adamw_no_decay_params)
                print(f"Optimizer: Muon + AdamW")
                print(
                    f"  Muon: {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} params"
                )
                print(
                    f"  AdamW: {total_adamw} tensors, {sum(p.numel() for p in adamw_decay_params + adamw_no_decay_params):,} params"
                )
                print(f"    - with weight decay: {len(adamw_decay_params)} tensors")
                print(
                    f"    - without weight decay: {len(adamw_no_decay_params)} tensors"
                )

            param_groups = [
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": cfg.training.learning_rate * 100,
                    "weight_decay": cfg.training.weight_decay,
                    "momentum": 0.95,
                },
                {
                    "params": adamw_decay_params,
                    "use_muon": False,
                    "lr": cfg.training.learning_rate,
                    "betas": (0.9, 0.95),
                    "weight_decay": cfg.training.weight_decay,
                },
            ]

            # Only add the no-decay group if there are parameters to exclude
            if adamw_no_decay_params:
                param_groups.append(
                    {
                        "params": adamw_no_decay_params,
                        "use_muon": False,
                        "lr": cfg.training.learning_rate,
                        "betas": (0.9, 0.95),
                        "weight_decay": 0.0,
                    }
                )

            if use_ddp:
                opt = MuonWithAuxAdam(param_groups)
            else:
                opt = SingleDeviceMuonWithAuxAdam(param_groups)
            # set lr for scheduler compatibility
            opt.defaults = {"lr": cfg.training.learning_rate}
        else:
            # Adam optimizer (+ weight decay exclude)
            exclude_params = getattr(cfg.training, "exclude_from_wd", [])
            if exclude_params != []:
                param_groups = exclude_from_weight_decay(
                    ae, exclude_params, cfg.training.weight_decay
                )
            else:
                param_groups = [
                    {
                        "params": ae.parameters(),
                        "weight_decay": cfg.training.weight_decay,
                    }
                ]

            opt = torch.optim.Adam(
                param_groups,
                lr=cfg.training.learning_rate,
            )

    if cfg.stage == "peft":
        if cfg.ae_checkpoint is None:
            raise ValueError(
                "PEFT stage requires a pretrained autoencoder checkpoint (set ae_checkpoint)"
            )

        # get ae first
        ae = models
        ae = ae.to(device)

        checkpoint_path = cfg.ae_checkpoint
        if os.path.isdir(checkpoint_path):
            if getattr(cfg.training, "use_latest_checkpoint", False):
                checkpoint_path = os.path.join(checkpoint_path, "ckp.pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(cfg.ae_checkpoint, "best.pth")
            else:
                checkpoint_path = os.path.join(checkpoint_path, "best.pth")
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = os.path.join(cfg.ae_checkpoint, "ckp.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"No checkpoint files found in {cfg.ae_checkpoint}"
                )

        # load checkpoint once + check if it is a PEFT checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        loaded_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        is_peft_checkpoint = loaded_ckpt.get("stage") == "peft"

        if is_peft_checkpoint:
            ae, ae_ckpt_dict = load_autoencoder(
                checkpoint_path, model=ae, device=device, load_peft=True
            )

            # get training state for resuming
            start_epoch = ae_ckpt_dict["epoch"]
            loss_val_min = ae_ckpt_dict["loss"]
            cur_update_step = start_epoch * len(trainloader)

            print(
                f"Resumed PEFT training from epoch {start_epoch}, loss: {loss_val_min:.6f}"
            )
            del loaded_ckpt

        else:
            print("Starting PEFT training from base autoencoder checkpoint")
            del loaded_ckpt

            ae, ae_ckpt_dict = load_autoencoder(
                checkpoint_path, model=ae, device=device
            )
            print(
                f"Loaded base autoencoder from epoch {ae_ckpt_dict['epoch']}, loss: {ae_ckpt_dict['loss']:.6f}"
            )

            peft_config = getattr(cfg.autoencoder, "peft", {})
            peft_model, peft_info = setup_peft_stage(ae, cfg, dataloader=trainloader)
            ae = peft_model

            start_epoch = 0
            loss_val_min = torch.inf
            cur_update_step = 0.0

        if use_ddp:
            ae = DDP(ae, device_ids=[rank])

        # config optimizer for PEFT parameters
        if hasattr(cfg.training, "optimizer") and cfg.training.optimizer == "muon":

            def separate_lora_parameters(model):
                muon_params = []
                adamw_params = []

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue

                    if any(key in name for key in ["lora_A", "lora_B", "eva_"]):
                        if param.ndim >= 2:
                            muon_params.append(param)
                        else:
                            adamw_params.append(param)
                    else:
                        adamw_params.append(param)

                return muon_params, adamw_params

            muon_params, adamw_params = separate_lora_parameters(ae)

            if rank == 0:
                print(f"PEFT Optimizer: Muon + AdamW")
                print(
                    f"  Muon: {len(muon_params)} tensors, {sum(p.numel() for p in muon_params):,} params"
                )
                print(
                    f"  AdamW: {len(adamw_params)} tensors, {sum(p.numel() for p in adamw_params):,} params"
                )

            param_groups = [
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": cfg.training.learning_rate * 100,  # Muon uses higher LR
                    "weight_decay": cfg.training.weight_decay,
                    "momentum": 0.95,
                },
                {
                    "params": adamw_params,
                    "use_muon": False,
                    "lr": cfg.training.learning_rate,
                    "betas": (0.9, 0.95),
                    "weight_decay": cfg.training.weight_decay,
                },
            ]

            if use_ddp:
                opt = MuonWithAuxAdam(param_groups)
            else:
                opt = SingleDeviceMuonWithAuxAdam(param_groups)
            opt.defaults = {"lr": cfg.training.learning_rate}

        else:
            # default Adam optimizer
            trainable_params = [p for p in ae.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError("No trainable parameters found in PEFT model")

            opt = torch.optim.Adam(
                trainable_params,
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )

        # Store for later use
        model = ae

    if cfg.training.scheduler is not None:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler,
            optimizer=opt,
            num_warmup_steps=(
                total_steps // 6
                if n_epochs > 150
                else max(total_steps // 10, 10 * len(trainloader))
            ),
            num_training_steps=total_steps,
            scheduler_specific_kwargs={
                "min_lr": (
                    cfg.training.min_lr if hasattr(cfg.training, "min_lr") else None
                ),
            },
        )

    loss_val_min = torch.inf
    cur_update_step = 0.0
    start_epoch = 0
    if cfg.ae_checkpoint is not None and cfg.stage != "peft":
        # choosing best.pt since ckpt.pt does not contain scheduler sd
        ae, ae_ckpt_dict = load_autoencoder(cfg.ae_checkpoint, model=ae, device=device)
        if cfg.stage == "autoencoder":
            opt.load_state_dict(ae_ckpt_dict["optimizer_state_dict"])
            lr_scheduler.load_state_dict(ae_ckpt_dict["scheduler_state_dict"])
            start_epoch = ae_ckpt_dict["epoch"]
            loss_val_min = ae_ckpt_dict["loss"]
            cur_update_step = start_epoch * len(trainloader)
    elif cfg.stage == "peft":
        if is_peft_checkpoint:
            try:
                opt.load_state_dict(ae_ckpt_dict["optimizer_state_dict"])
                if cfg.training.scheduler is not None:
                    lr_scheduler.load_state_dict(ae_ckpt_dict["scheduler_state_dict"])
                print(f"Loaded optimizer and scheduler state from PEFT checkpoint")
            except KeyError as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")
                print("Starting with fresh optimizer state")
        pass
    else:
        assert cfg.stage == "autoencoder", "Set a pretrained autoencoder."

    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
    grad_balancer = GradientBalancer(
        opt,
        mode=cfg.training.gradnorm_balancer,
        scaler=scaler,
        clip_grad=cfg.training.clip_grad,
        n_tasks=len(loss_wrap.active_losses),  # Use active losses, not all weights
    )

    if not rank:
        print(f"Gradient Balancer Setup:")
        print(f"  Mode: {cfg.training.gradnorm_balancer}")
        print(f"  Total loss weights: {len(weights)}")
        print(
            f"  Active losses: {len(loss_wrap.active_losses)} - {loss_wrap.active_losses}"
        )
        (
            print(
                f"  Zero-weighted losses: {[k for k in weights if weights[k] == 0.0]}"
            )
            if len(weights) != len(loss_wrap.active_losses)
            else ""
        )

    if cfg.stage == "autoencoder" or cfg.stage == "peft":
        # rename to avoid unbound locals
        model = ae

    use_tqdm = cfg.logging.tqdm if not use_ddp else False

    for epoch in range(start_epoch + 1, n_epochs + 1):
        if use_tqdm or (use_ddp and not rank):
            trainloader = tqdm(trainloader, "Training")

        ############################### train loop start ###############################

        loss_logs = {"total": []}
        if len(loss_wrap.active_losses) > 1:
            loss_logs.update({k: [] for k in loss_wrap.active_losses})
        progress_remaining = 1.0
        loss_wrap.train().to(device)
        info_dict = defaultdict(list)
        t_start_data = perf_counter_ns()

        input_fields = set(cfg.dataset.input_fields)
        idx_keys = ["file_index", "timestep_index"]

        use_amp = cfg.amp.enable
        use_bf16 = use_amp and cfg.amp.bfloat and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        for sample in trainloader:
            try:
                reset_peak_memory_stats(device)
            except:
                pass  # only works with cuda device
            sample: CycloneAESample
            xs = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in input_fields
                if getattr(sample, k) is not None
            }
            condition = sample.conditioning.to(device)
            idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}
            geometry = tree_map(lambda g: g.to(device), sample.geometry)

            if augmentations is not None:
                for aug_fn in augmentations:
                    xs = {k: aug_fn(v) for k, v in xs.items()}

            # dataloading timings
            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

            t_start_fwd = perf_counter_ns()

            with torch.autocast(str(device), dtype=amp_dtype, enabled=use_amp):
                if cfg.stage == "autoencoder":
                    loss, losses = train_step_autoencoder(
                        cfg,
                        model=model,
                        xs=xs,
                        condition=condition,
                        idx_data=idx_data,
                        geometry=geometry,
                        loss_wrap=loss_wrap,
                        progress_remaining=progress_remaining,
                    )
                elif cfg.stage == "peft":
                    loss, losses = train_step_peft(
                        cfg,
                        model=model,
                        xs=xs,
                        condition=condition,
                        idx_data=idx_data,
                        geometry=geometry,
                        loss_wrap=loss_wrap,
                        progress_remaining=progress_remaining,
                    )

            # forward timing
            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

            # backward pass (+optional gradnorm for multitask)
            # filter losses for gradient balancer: exclude monitoring losses and zero-weighted losses
            grad_losses = []
            grad_loss_names = []
            for k, v in losses.items():
                # Skip monitoring losses that are computed with torch.no_grad()
                if k in ["total_mse", "phi_int_mse", "flux_int_mse"] or k.endswith(
                    "_mse"
                ):
                    continue
                # skip losses with zero weight (not in active losses)
                if k not in loss_wrap.active_losses:
                    continue
                if hasattr(v, "requires_grad") and v.requires_grad:
                    # print(f'Including loss "{k}" for gradient balancing with value {v.item():.6f}')
                    grad_losses.append(v)
                    grad_loss_names.append(k)

            model = grad_balancer(model, loss, grad_losses)

            # lr scheduler step
            if cfg.training.scheduler is not None:
                lr_scheduler.step()

            progress_remaining = remainig_progress(cur_update_step, total_steps)
            cur_update_step += 1.0

            # Add all losses to loss_logs
            for k, v in losses.items():
                if k not in loss_logs:
                    loss_logs[k] = []
                loss_logs[k].append(v.item())
            loss_logs["total"].append(loss.item())

            # cleanup every 100 steps to prevent memory buildup
            if (cur_update_step % 100) == 0:
                del xs, condition, idx_data, geometry, loss, losses
                # memory_cleanup(device)
                memory_cleanup(device, aggressive=True)

            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
            t_start_data = perf_counter_ns()

        ################################ train loop end ################################
        memory_cleanup(device, aggressive=True)

        # logging
        mse_logs = {k: v for k, v in loss_logs.items() if "total_mse" in k}
        training_logs = {k: v for k, v in loss_logs.items() if "total_mse" not in k}

        # tag training objective with actual loss type
        loss_type = getattr(cfg.training, "loss_type", "mse")
        # change 'df' to 'df_{loss_type}' and 'total' as well (legacy)
        training_logs = {
            k.replace("df", f"df_{loss_type}") if "df" in k else k: v
            for k, v in training_logs.items()
        }
        training_logs = {
            k.replace("total", f"total_{loss_type}") if "total" in k else k: v
            for k, v in training_logs.items()
        }
        # add train/ to training logs
        training_logs = {f"train/{k}": v for k, v in training_logs.items()}

        # tag MSE monitoring metrics (used if mse is not the training loss)
        mse_logs_tagged = {f"train/{k}": v for k, v in mse_logs.items()}

        # combine all logs
        all_logs = {**training_logs, **mse_logs_tagged}
        all_logs = {k: sum(v) / max(len(v), 1) for k, v in all_logs.items()}
        train_losses_dict = {
            "train/lr": (
                lr_scheduler.get_last_lr()[0]
                if cfg.training.scheduler
                else cfg.training.learning_rate
            ),
        }
        for key in loss_scheduler_dict.keys():
            train_losses_dict[f"train/{key}_schedule"] = loss_scheduler_dict[key](
                progress_remaining
            )
        train_losses_dict = train_losses_dict | all_logs
        info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

        ############################### evaluation start ###############################

        log_metric_dict, val_plots, loss_val_min = evaluate(
            stage=cfg.stage,
            rank=rank,
            world_size=world_size,
            model=model,
            loss_wrap=loss_wrap,
            valsets=valsets,
            valloaders=valloaders,
            opt=opt,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            cfg=cfg,
            device=device,
            loss_val_min=loss_val_min,
        )

        ################################ evaluation end ################################
        memory_cleanup(device, aggressive=True)

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

    if use_ddp:
        dist.destroy_process_group()
