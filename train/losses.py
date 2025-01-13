from typing import List, Callable

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.distributed import get_rank, get_world_size, is_initialized

from concurrent.futures import ThreadPoolExecutor

from dataset.cyclone import CycloneDataset


def relative_norm_mse(x, y, dim_to_keep=None):
    if dim_to_keep is None:
        y = y.flatten(1)
        diff = x.flatten(1) - y
        diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        diff_norms, y_norms = diff_norms**2, y_norms**2
        # sum over timesteps and mean over examples in batch
        return torch.mean(diff_norms / y_norms)
    else:
        # TODO: Check if this is necessary
        y = y.flatten(2)
        diff = x.flatten(2) - y
        diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        diff_norms, y_norms = diff_norms**2, y_norms**2
        dims = [i for i in range(len(y_norms.shape))][dim_to_keep + 1 :]
        return torch.mean(diff_norms / y_norms, dim=dims)


def get_pushforward_trick(
    unrolls: List[int],
    probs: List[float],
    schedule: List[float],
    predict_delta: bool,
    dataset: CycloneDataset,
    bundle_steps: int,
    use_amp: bool = False,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        ts: torch.Tensor,
        ts_idx: torch.Tensor,
        y: torch.Tensor,
        file_idx: torch.Tensor,
        epoch: int,
        device: str
    ) -> List[float]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(schedule)).sum()
        # sample number of steps
        curr_probs = [p / sum(probs[:idx]) for p in probs[:idx]]
        unroll_steps = np.random.choice(unrolls[:idx], p=curr_probs)

        # cap the unroll steps depending on the current max timestep
        unroll_steps = min(
            [
                min(
                    (dataset.num_ts(f_idx) - int(ts_idx[i])) // bundle_steps - 1,
                    unroll_steps,
                )
                for i, f_idx in enumerate(file_idx.tolist())
            ]
        )

        if unroll_steps < 2:
            return x, ts, y

        # get timesteps for unrolling
        ts_idxs = [
            [
                i
                for i in range(
                    int(ts_idx_start),
                    int(ts_idx_start) + unroll_steps * bundle_steps,
                    bundle_steps
                )
            ]
            for ts_idx_start in ts_idx.tolist()
        ]
        tsteps = dataset.get_timesteps_only(file_idx, torch.tensor(ts_idxs))

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(file_idx.cpu(), ts_unrolled.cpu())

        executor = ThreadPoolExecutor(max_workers=1)
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        with torch.no_grad():
            ts_unrolled = ts_idx + (unroll_steps - 1) * bundle_steps
            future = executor.submit(fetch_target, dataset, file_idx, ts_unrolled)

            xt = x
            for i in range(unroll_steps - 1):
                # TODO: currenlty only integer conditioning. Remove that line if floats are possible
                with torch.autocast("cuda", dtype=torch.float16 if not use_bf16 else torch.bfloat16, enabled=use_amp):
                    x_p = model(xt, timestep=tsteps[:, i].to(xt.device))
                    if predict_delta:
                        x_p = xt + x_p
                    xt = x_p.clone().float()
            # Get the result when needed
            unrolled = future.result()

        # have to clone xt to avoid view mode grad runtime error
        return (
            xt.clone(),
            unrolled.timestep.to(x.device),
            unrolled.y.to(x.device, non_blocking=True),
        )

    return _loss_fn


def pretrain_autoencoder(model, cfg, trainloader, valloader, writer, device):

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
    for epoch in range(1, cfg.training.pretraining_kwargs.n_epochs + 1):
        train_mse = 0
        if use_tqdm or (use_ddp and not rank):
            trainloader = tqdm(trainloader, "AE pretraining")
        for sample in trainloader:
            x = sample.x.to(device)
            ts = sample.timestep.to(device)

            with torch.autocast(cfg.device, dtype=torch.float16 if not use_bf16 else torch.bfloat16, enabled=cfg.use_amp):
                if cfg.training.pretraining_kwargs.target_modules == "all":
                    pred_x = model(x, timestep=torch.ceil(ts))
                else:
                    if hasattr(model, "module"):
                        z, pad_ax = model.module.patch_encode(x)
                    else:
                        z, pad_ax = model.patch_encode(x)

                    if cfg.training.pretraining_kwargs.add_noise:
                        z = z + torch.normal(0, 1e-3, size=(z.shape), device=z.device)

                    if hasattr(model, "module"):
                        pred_x = model.module.patch_decode(z, pad_ax)
                    else:
                        pred_x = model.patch_decode(z, pad_ax)
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
            val_mse = 0
            if cfg.logging.tqdm:
                valloader = tqdm(valloader, "AE evaluation")
            for sample in valloader:
                x = sample.x.to(device)
                if hasattr(model, "module"):
                    pred_x = model.module.patch_decode(*model.module.patch_encode(x))
                else:
                    pred_x = model.patch_decode(*model.patch_encode(x))
                if cfg.training.predict_delta:
                    pred_x = x + pred_x
                loss = relative_norm_mse(pred_x, x)
                val_mse += loss.item()
            val_mse = val_mse / len(valloader)
            val_log = f", val/relative_norm_mse: {val_mse:.4f}"
            if is_main_proc and writer:
                writer.log({
                    "pretrain/val_relative_norm_mse": val_mse,
                })

        epoch_str = str(epoch).zfill(
            len(str(int(cfg.training.pretraining_kwargs.n_epochs)))
        )
        print(
            f"AE epoch: {epoch_str}, train/relative_norm_mse: {train_mse:.4f}{val_log}"
        )
        if is_main_proc and writer:
            writer.log({
                "pretrain/train_relative_norm_mse": train_mse,
                "pretrain/train_lr": (
                    scheduler.get_last_lr()[0]
                    if cfg.training.pretraining_kwargs.scheduler is not None
                    else cfg.training.pretraining_kwargs.lr
                ),
            })

    if cfg.training.pretraining_kwargs.freeze_after:
        # freeze patching
        print("Freezing AE weights...")
        if hasattr(model, "module"):
            model = model.module
        model.patch_embed.requires_grad_(False)
        model.unpatch.requires_grad_(False)

    print("Pretraining done!\n\n")

    return model
