from datetime import datetime, timedelta

import gc
import torch
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import perf_counter_ns
from collections import defaultdict
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor

from dataset import get_data, CycloneSample
from models import get_model
from train import get_pushforward_fn, relative_norm_mse, pretrain_autoencoder
from eval.evaluate import evaluate
from utils import (
    load_model_and_config,
    setup_logging,
    edit_tag
)


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


def runner(rank, cfg, train_method, world_size):
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

    model = get_model(cfg, dataset=trainset, train_method=train_method)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    bundle_seq_length = cfg.model.bundle_seq_length
    use_gkw = cfg.validation.use_gkw
    gkw_args = {}
    if use_gkw:
        if cfg.dataset.normalization_scope == "sample":
            raise UserWarning(
                "Cannot denormalize with `normalization_scope='sample'`. "
                "Cannot compute correct GKW results."
            )
        gkw_executor = ThreadPoolExecutor(max_workers=1)
        # TODO hardcoded for now
        gkw_dump_path = "/system/user/publicdata/gyrokinetics/dumps/test_gkw_client"
        gkw_futures = {}
        gkw_args['executor'] = gkw_executor
        gkw_args['dump_path'] = gkw_dump_path
        gkw_args['futures'] = gkw_futures

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
        if sum(pf_cfg.unrolls) > 0:
            pushforward_fn = get_pushforward_fn(
                n_unrolls_schedule=pf_cfg.unrolls,
                probs_schedule=pf_cfg.probs,
                epoch_schedule=pf_cfg.epochs,
                predict_delta=predict_delta,
                dataset=trainset,
                bundle_steps=bundle_seq_length,
                use_amp=use_amp,
                device=device,
            )

        if cfg.training.pretraining:
            model = pretrain_autoencoder(
                model, cfg, trainloader, valloaders, writer, device
            )  # only valuate on the holdout trajectories, not the holdout samples
            if not hasattr(model, "module") and use_ddp:
                model = DDP(model, device_ids=[rank])

        input_fields = cfg.dataset.input_fields
        outputs = cfg.model.losses
        conditioning = cfg.model.conditioning
        idx_keys = ["file_index", "timestep_index"]
        loss_weight_dict = {k: v for k, v in zip(cfg.model.losses, cfg.model.loss_weights)}
        use_tqdm = cfg.logging.tqdm if not use_ddp else False
        loss_val_min = torch.inf
        for epoch in range(1, n_epochs + 1):
            loss_logs = {k: 0 for k in cfg.model.losses}
            loss_logs['relative_norm'] = 0.
            model.train()
            info_dict = defaultdict(list)
            t_start_data = perf_counter_ns()

            if use_tqdm or (use_ddp and not rank):
                trainloader = tqdm(trainloader, "Training")

            for i, sample in enumerate(trainloader):
                reset_peak_memory_stats(device)
                sample: CycloneSample
                inputs = { k: getattr(sample, k).to(device, non_blocking=True) for k in input_fields if k is not None }
                gts = {k: getattr(sample, f"y_{k}").to(device, non_blocking=True) for k in outputs if k is not None}
                conds = { k: getattr(sample, k).to(device, non_blocking=True) for k in conditioning if k is not None }
                idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

                # TODO should augmentations take place before moving to GPU?
                if augmentations is not None:
                    for aug_fn in augmentations:
                        inputs = {k: aug_fn(v) for k, v in inputs.items()}

                # dataloading timings
                info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

                if pushforward_fn:
                    start_pf = perf_counter_ns()
                    # accessory information for pf (to retreive unrolled target)
                    inputs, gts, conds = pushforward_fn(
                        model, inputs, gts, conds,
                        idx_data, epoch,
                    )
                    info_dict["pf_ms"].append((perf_counter_ns() - start_pf) / 1e6)
                else:
                    info_dict["pf_ms"].append(0.0)

                t_start_fwd = perf_counter_ns()
                with torch.autocast(str(device), dtype=amp_dtype, enabled=use_amp):
                    outputs = model(**inputs, **conds)
                    if predict_delta:
                        for key in cfg.dataset.input_fields:
                            outputs[key] = outputs[key] + inputs[key]

                    losses = {}
                    loss = 0.
                    for key in cfg.model.losses:
                        losses[key] = relative_norm_mse(outputs[key], gts[key])
                        loss += loss_weight_dict[key] * losses[key]

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

                for k in cfg.model.losses:
                    loss_logs[k] += losses[k].item()
                loss_logs['relative_norm'] += loss.item()

                del inputs
                del gts
                del losses
                del loss
                gc.collect()
                torch.cuda.empty_cache()

                info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
                info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
                t_start_data = perf_counter_ns()

            for k in cfg.model.losses:
                loss_logs[k] /= len(trainloader)
            loss_logs['relative_norm'] /= len(trainloader)
            loss_logs = edit_tag(loss_logs, prefix="train", postfix="mse")
            train_losses_dict = {
                "train/lr": (
                    scheduler.get_last_lr()[0]
                    if cfg.training.scheduler
                    else cfg.training.learning_rate
                ),
            }
            train_losses_dict = train_losses_dict | loss_logs
            info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

            log_metric_dict, val_plots, loss_val_min = evaluate(rank, world_size, model, valsets,
                     valloaders, opt, epoch, cfg, device, gkw_args, loss_val_min)

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
        if use_gkw:
            # TODO wait some more if some gkw process is still hanging
            pass

        if writer:
            writer.finish()

    if cfg.mode == "rollout":
        raise NotImplementedError("TODO")
