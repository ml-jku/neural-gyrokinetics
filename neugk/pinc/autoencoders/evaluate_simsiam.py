"""Evaluation utilities for SimSiam-style training."""

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from neugk.dataset.cyclone_diff import CycloneSimSiamSample


@torch.no_grad()
def collect_xy(
    rank: int,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    desc: str = "linear probe",
):
    """Collect model latents and target fluxes for linear probing."""
    model.eval()
    latents, fluxes = [], []

    # setup iterator
    use_tqdm = (not dist.is_initialized() or rank == 0) and desc
    iterator = tqdm(dataloader, desc=desc) if use_tqdm else dataloader

    for sample in iterator:
        sample: CycloneSimSiamSample
        xs = sample.df.to(device, non_blocking=True)
        condition = sample.conditioning.to(device, non_blocking=True)
        flux = sample.flux.to(device, non_blocking=True)

        # forward pass for latents
        z = model(xs, condition=condition, decoder=False)["z"]
        zpool = z.view(z.shape[0], -1, z.shape[-1]).mean(1)
        latents.append(zpool.cpu())
        fluxes.append(flux.view(flux.shape[0], -1).cpu())

    return torch.cat(latents, 0), torch.cat(fluxes, 0)


def evaluate_linear_probe(
    rank: int,
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloaders: list,
    epoch: int,
    cfg: DictConfig,
    device: torch.device,
    loss_val_min: float,
):
    """Solve linear probe and evaluate on validation sets."""
    log_metric_dict, val_freq = {}, cfg.validation.validate_every_n_epochs
    if epoch % val_freq != 0 and epoch != 1:
        return log_metric_dict, loss_val_min

    # compute probe weights on trainset
    x_train, y_train = collect_xy(rank, trainloader, model, device)
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    x_centered, y_centered = x_train - x_mean, y_train - y_mean
    w = torch.linalg.lstsq(x_centered, y_centered, driver="gels").solution

    # log train performance
    train_mse = torch.mean(((x_centered @ w) - y_centered) ** 2)
    log_metric_dict["val_traj/probe_train_mse"] = train_mse.item()

    # evaluate on holdouts
    for val_idx, valloader in enumerate(valloaders):
        valname = "val_traj" if val_idx == 0 else "val_samples"
        x_val, y_val = collect_xy(rank, valloader, model, device, desc=None)
        x_val_centered, y_val_centered = x_val - x_mean, y_val - y_mean
        val_mse = torch.mean(((x_val_centered @ w) - y_val_centered) ** 2)
        log_metric_dict[f"{valname}/probe_val_mse"] = val_mse.item()

    return log_metric_dict, loss_val_min
