"""Evaluation utilities for SimSiam-style training."""

from typing import Optional, List, Dict, Tuple, Any
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from neugk.dataset.cyclone_diff import CycloneSimSiamSample
from neugk.pinc.autoencoders.evaluate import AutoencoderEvaluator


class SimSiamEvaluator(AutoencoderEvaluator):
    """Evaluator for SimSiam models with linear probing."""

    @torch.no_grad()
    def collect_xy(
        self,
        rank: int,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        desc: Optional[str] = "linear probe",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect model latents and target fluxes for linear probing."""
        model.eval()
        latents: List[torch.Tensor] = []
        fluxes: List[torch.Tensor] = []

        # setup iterator
        use_tqdm = (not dist.is_initialized() or rank == 0) and desc
        iterator = tqdm(dataloader, desc=desc) if use_tqdm else dataloader

        for sample in iterator:
            sample: CycloneSimSiamSample
            xs = sample.df.to(device, non_blocking=True)
            condition = sample.conditioning.to(device, non_blocking=True)
            flux = sample.flux.to(device, non_blocking=True)

            # forward pass for latents
            res = model(xs, condition=condition, decoder=False)
            z = res["z"]
            zpool = z.view(z.shape[0], -1, z.shape[-1]).mean(1)
            latents.append(zpool.cpu())
            fluxes.append(flux.view(flux.shape[0], -1).cpu())

        return torch.cat(latents, 0), torch.cat(fluxes, 0)

    def __call__(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        device: torch.device,
        loss_val_min: float,
        trainloader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs: Any
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """Solve linear probe and evaluate on validation sets."""
        # First perform standard autoencoder evaluation if requested
        log_metric_dict: Dict[str, float] = {}
        val_plots: Dict[str, Any] = {}
        
        if getattr(model, "use_simae_decoder", True):
            log_metric_dict, val_plots, loss_val_min = super().__call__(
                rank, world_size, model, opt, scheduler, epoch, device, loss_val_min, **kwargs
            )
        
        if not self._is_eval_epoch(epoch):
            return log_metric_dict, val_plots, loss_val_min

        # Linear Probing
        if trainloader is None:
            return log_metric_dict, val_plots, loss_val_min

        # compute probe weights on trainset
        x_train, y_train = self.collect_xy(rank, trainloader, model, device)
        x_mean, y_mean = x_train.mean(0), y_train.mean(0)
        x_centered, y_centered = x_train - x_mean, y_train - y_mean
        
        # Linear regression solution
        w = torch.linalg.lstsq(x_centered, y_centered, driver="gels").solution

        # log train performance
        train_mse = torch.mean(((x_centered @ w) - y_centered) ** 2)
        log_metric_dict["val_traj/probe_train_mse"] = train_mse.item()

        # evaluate on holdouts
        for val_idx, valloader in enumerate(self.valloaders):
            valname = "val_traj" if val_idx == 0 else "val_samples"
            x_val, y_val = self.collect_xy(rank, valloader, model, device, desc=None)
            x_val_centered, y_val_centered = x_val - x_mean, y_val - y_mean
            val_mse = torch.mean(((x_val_centered @ w) - y_val_centered) ** 2)
            log_metric_dict[f"{valname}/probe_val_mse"] = val_mse.item()

        return log_metric_dict, val_plots, loss_val_min
