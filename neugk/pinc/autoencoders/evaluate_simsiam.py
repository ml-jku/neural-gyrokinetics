import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from neugk.dataset.cyclone_diff_simsiam import CycloneSimSiamSample
from neugk.utils import save_model_and_config


@torch.no_grad()
def collect_xy(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, desc: str = "Validation holdout trajectories"):
    model.eval()
    latents, fluxes = [], []
    
    if (not dist.is_initialized() or dist.get_rank() == 0) and desc is not None:
        iterator = tqdm(dataloader, desc=desc)
    else:
        iterator = dataloader

    for sample in iterator:
        sample: CycloneSimSiamSample
        xs = sample.df.to(device, non_blocking=True)
        condition = sample.conditioning.to(device, non_blocking=True)
        flux = sample.flux.to(device, non_blocking=True)
        
        (z, _), _, _ = model(xs, condition=condition)
        
        zpool = z.view(z.shape[0], -1, z.shape[-1]).mean(1)
        latents.append(zpool.cpu())
        fluxes.append(flux.view(flux.shape[0], -1).cpu())

    return torch.cat(latents, 0), torch.cat(fluxes, 0)


def evaluate_linear_probe(
    rank: int,
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloaders: list,
    opt: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    epoch: int,
    cfg: DictConfig,
    device: torch.device,
    loss_val_min: float,
): 
    log_metric_dict, val_freq = {}, cfg.validation.validate_every_n_epochs
    if (epoch % val_freq) != 0 and epoch != 1:
        return log_metric_dict, loss_val_min

    X_train, Y_train = collect_xy(trainloader, model, device)
    X_mean, Y_mean = X_train.mean(0), Y_train.mean(0)
    X_centered, Y_centered = X_train - X_mean, Y_train - Y_mean

    W = torch.linalg.lstsq(X_centered, Y_centered, driver="gels").solution

    train_mse = torch.mean(((X_centered @ W) - Y_centered) ** 2)
    log_metric_dict["probe/train_mse"] = train_mse.item()

    for val_idx, valloader in enumerate(valloaders):
        valname = "val_traj" if val_idx == 0 else "val_samples"
        X_val, Y_val = collect_xy(valloader, model, device, desc=None)
        
        X_val_centered, Y_val_centered = X_val - X_mean, Y_val - Y_mean

        val_mse = torch.mean(((X_val_centered @ W) - Y_val_centered) ** 2)
        log_metric_dict[f"probe/{valname}_mse"] = val_mse.item()

    if not rank:
        val_loss = log_metric_dict["probe/val_traj_mse"]
        loss_val_min = save_model_and_config(
            model,
            optimizer=opt,
            scheduler=lr_scheduler,
            cfg=cfg,
            epoch=epoch,
            # TODO decide target metric
            val_loss=val_loss,
            loss_val_min=loss_val_min,
        )

    return log_metric_dict, loss_val_min