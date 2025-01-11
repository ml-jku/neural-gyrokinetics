from typing import List, Callable

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_

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
    schecule: List[float],
    predict_delta: bool,
    dataset: CycloneDataset,
    bundle_steps: int,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        ts: torch.Tensor,
        ts_idx: torch.Tensor,
        y: torch.Tensor,
        file_idx: torch.Tensor,
        epoch: int,
    ) -> List[float]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(schecule)).sum()
        # sample number of steps
        curr_probs = [p / sum(probs[:idx]) for p in probs[:idx]]
        unroll_steps = np.random.choice(unrolls[:idx], p=curr_probs)

        # cap the unroll steps depending on the current max timestep
        unroll_steps = min(
            [
                min(
                    dataset.num_ts(f_idx) - int(ts_idx[i]) - bundle_steps + 1,
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
                    int(ts_idx_start) + (unroll_steps - 1) * bundle_steps,
                )
            ]
            for ts_idx_start in ts_idx.tolist()
        ]
        tsteps = dataset.get_timesteps_only(file_idx, torch.tensor(ts_idxs))
        # ts_idxs = [
        #     [i for i in range(int(ts_idx_start), int(ts_idx_start) + n_steps*bundle_steps, bundle_steps)]
        #     for ts_idx_start in ts_index_0.tolist()
        # ]
        # tsteps = dataset.get_timesteps_only(file_idx, torch.tensor(ts_idxs))

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(file_idx.cpu(), ts_unrolled.cpu())

        executor = ThreadPoolExecutor(max_workers=1)

        with torch.no_grad():
            ts_unrolled = ts_idx + (unroll_steps - 1) * bundle_steps
            future = executor.submit(fetch_target, dataset, file_idx, ts_unrolled)

            xt = x
            for i in range(unroll_steps - 1):
                # TODO: currenlty only integer conditioning. Remove that line if floats are possible
                x_p = model(xt, timestep=tsteps[:, i]).to(xt.device)
                if predict_delta:
                    x_p = xt + x_p
                xt = x_p.clone()
            # Get the result when needed
            unrolled = future.result()

        # have to clone xt to avoid view mode grad runtime error
        return (
            xt.clone(),
            unrolled.timestep.to(x.device),
            unrolled.y.to(x.device, non_blocking=True),
        )

    return _loss_fn


def pretrain_autoencoder(model, cfg, trainloader, valloader, writer):

    if cfg.training.pretraining_kwargs.target_modules == "all":
        target_modules = model.parameters()
    else:
        target_modules = []
        for n, p in model.named_parameters():
            for t in cfg.training.pretraining_kwargs.target_modules:
                if t in n:
                    target_modules.append(p)

    scaler = torch.amp.GradScaler(device=cfg.device, enabled=cfg.use_amp)
    use_bf16 = cfg.use_amp and torch.cuda.is_bf16_supported()
    n_epochs = cfg.training.pretraining_kwargs.n_epochs

    opt = torch.optim.Adam(
        target_modules,
        lr=cfg.training.pretraining_kwargs.lr,
        weight_decay=cfg.training.pretraining_kwargs.weight_decay,
    )

    if cfg.training.pretraining_kwargs.scheduler is not None:
        total_steps = n_epochs * len(trainloader)
        scheduler = get_scheduler(
            name=cfg.training.pretraining_kwargs.scheduler,
            optimizer=opt,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

    for epoch in range(1, cfg.training.pretraining_kwargs.n_epochs + 1):
        train_mse = 0
        if cfg.logging.tqdm:
            trainloader = tqdm(trainloader, "AE pretraining")
        for sample in trainloader:
            x = sample.x.to(cfg.device)

            with torch.autocast(
                cfg.device,
                dtype=torch.float16 if not use_bf16 else torch.bfloat16,
                enabled=cfg.use_amp,
            ):
                z, pad_ax = model.patch_encode(x)
                # TODO
                z = z + torch.normal(0, 1e-3, size=(z.shape), device=z.device)
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
        if (epoch % 10) == 0 or epoch == 1:
            val_mse = 0
            if cfg.logging.tqdm:
                valloader = tqdm(valloader, "AE evaluation")
            for sample in valloader:
                x = sample.x.to(cfg.device)
                pred_x = model.patch_decode(*model.patch_encode(x))
                if cfg.training.predict_delta:
                    pred_x = x + pred_x
                loss = relative_norm_mse(pred_x, x)
                val_mse += loss.item()
            val_mse = val_mse / len(valloader)
            val_log = f", val/relative_norm_mse: {val_mse:.4f}"
            if writer:
                writer.log({"pretrain/val_relative_norm_mse": val_mse})

        epoch_str = str(epoch).zfill(
            len(str(int(cfg.training.pretraining_kwargs.n_epochs)))
        )
        print(
            f"AE epoch: {epoch_str}, train/relative_norm_mse: {train_mse:.4f}{val_log}"
        )
        if writer:
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
        model.patch_embed.requires_grad_(False)
        model.unpatch.requires_grad_(False)

    print("Pretraining done!\n\n")

    return model
