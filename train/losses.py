from typing import List, Callable

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

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
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        ts: torch.Tensor,
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
                min(dataset.num_ts(f_idx) - int(ts[i]), unroll_steps)
                for i, f_idx in enumerate(file_idx.tolist())
            ]
        )

        if unroll_steps < 2:
            return x, ts, y

        with torch.no_grad():
            xt = x
            for i in range(unroll_steps - 1):
                x_p = model(x, timestep=(ts + i))

                if predict_delta:
                    x_p = x + x_p
            ts_unrolled = ts + unroll_steps - 1

            # TODO check if fetching correct target!
            # get unrolled y lazily (too large to load otherwise)
            _, _, y_unrolled, _ = dataset.get_at_time(file_idx, ts_unrolled.cpu())

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(file_idx, ts_unrolled.cpu())

        executor = ThreadPoolExecutor(max_workers=1)

        with torch.no_grad():
            ts_unrolled = ts + unroll_steps - 1
            future = executor.submit(fetch_target, dataset, file_idx, ts_unrolled)

            xt = x
            for i in range(unroll_steps - 1):
                x_p = model(x, timestep=(ts + i))

                if predict_delta:
                    x_p = x + x_p

            # Get the result when needed
            _, _, y_unrolled, _ = future.result()

        # have to clone xt to avoid view mode grad runtime error
        return xt.clone(), ts_unrolled, y_unrolled.to(x.device)

    return _loss_fn


def pretrain_autoencoder(model, cfg, trainloader, valloader):
    AE_n_epochs = 20  # TODO

    AE_opt = torch.optim.Adam(
        list(model.patch_embed.parameters()) + list(model.unpatch.parameters()),
        lr=5e-4,
        weight_decay=cfg.training.weight_decay,
    )

    for epoch in range(1, AE_n_epochs + 1):
        train_mse = 0
        if cfg.logging.tqdm:
            trainloader = tqdm(trainloader, "AE pretraining")
        for sample in trainloader:
            x = sample[0].to(cfg.device)
            # TODO
            x = x + torch.normal(0, 0.25, size=(x.shape), device=x.device)
            pred_x = model.autoencoder(x)
            if cfg.training.predict_delta:
                pred_x = x + pred_x
            loss = relative_norm_mse(pred_x, x)
            AE_opt.zero_grad()
            loss.backward()
            AE_opt.step()
            train_mse += loss.item()
        train_mse = train_mse / len(trainloader)

        val_log = ""
        if (epoch % 10) == 0 or epoch == 1:
            val_mse = 0
            if cfg.logging.tqdm:
                valloader = tqdm(valloader, "AE evaluation")
            for sample in valloader:
                x = sample[0].to(cfg.device)
                pred_x = model.autoencoder(x)
                if cfg.training.predict_delta:
                    pred_x = x + pred_x
                loss = relative_norm_mse(pred_x, x)
                val_mse += loss.item()
            val_mse = val_mse / len(valloader)
            val_log = f", val/relative_norm_mse: {val_mse:.4f}"

        epoch_str = str(epoch).zfill(len(str(int(AE_n_epochs))))
        print(
            f"AE epoch: {epoch_str}, train/relative_norm_mse: {train_mse:.4f}{val_log}"
        )

    # freeze patching
    model.patch_embed.requires_grad_ = False
    model.unpatch.requires_grad_ = False

    print("Pretraining done!\n\n")

    return model
