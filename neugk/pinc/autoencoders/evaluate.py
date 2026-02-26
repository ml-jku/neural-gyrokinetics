from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

from neugk.dataset import CycloneAESample
from neugk.eval import BaseEvaluator, validation_metrics
from neugk.plot_utils import generate_val_plots


class AutoencoderEvaluator(BaseEvaluator):
    """Evaluator for autoencoder models."""

    def _prepare_sample(
        self, sample: CycloneAESample, device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        xs = {
            k: getattr(sample, k).to(device, non_blocking=True)
            for k in self.cfg.dataset.input_fields
            if getattr(sample, k) is not None
        }
        tgts = {
            k: getattr(sample, k).to(device, non_blocking=True)
            for k in ["df", "phi", "flux", "avg_flux"]
            if getattr(sample, k) is not None
        }
        condition = (
            sample.conditioning.to(device)
            if sample.conditioning is not None
            else None
        )
        idx_data = {
            k: getattr(sample, k).to(device) 
            for k in ["file_index", "timestep_index"]
        }
        return xs, tgts, condition, idx_data

    @torch.no_grad()
    def __call__(
        self,
        rank: int,
        world_size: int,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        device: torch.device,
        loss_val_min: float,
        **kwargs
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """Evaluate autoencoder model on validation datasets"""
        if not self._is_eval_epoch(epoch):
            return {}, {}, loss_val_min

        model.eval()
        if self.loss_wrap:
            self.loss_wrap.eval().cpu()

        log_metric_dict: Dict[str, float] = {}
        val_plots: Dict[str, Any] = {}
        eval_integrals = self.cfg.validation.eval_integrals

        for val_idx, (valset, valloader) in enumerate(zip(self.valsets, self.valloaders)):
            valname = "val_traj" if val_idx == 0 else "val_samples"
            metrics = {key: torch.tensor(0.0) for key in self.loss_wrap.all_losses}
            n_timesteps_acc = torch.tensor(0.0)

            valloader = self.get_iterator(valloader, val_idx, rank)

            for idx, sample in enumerate(valloader):
                sample: CycloneAESample
                xs, tgts, condition, idx_data = self._prepare_sample(sample, device)

                # forward pass
                preds = model(xs["df"], condition=condition)

                # denormalize
                preds = self._denormalize_batch(preds, idx_data, valset.denormalize)
                tgts = self._denormalize_batch(tgts, idx_data, valset.denormalize)

                # cpu transfer for metric calculation
                tgts = {k: v.cpu() for k, v in tgts.items()}
                preds = {k: v.cpu() for k, v in preds.items()}

                # handle zonal flow
                if getattr(self.cfg.dataset, "separate_zf", False):
                    for d in [preds, tgts]:
                        for k in d:
                            d[k] = self._recombine_zf(d[k])

                # compute validation metrics
                metrics_i, integrated_i = validation_metrics(
                    tgts=tgts,
                    preds=preds,
                    geometry=sample.geometry,
                    loss_wrap=self.loss_wrap,
                    eval_integrals=eval_integrals,
                )

                # store integrals for plotting
                preds["phi_int"] = integrated_i.get("phi")
                preds["flux_int"] = integrated_i.get("eflux")

                # accumulate
                metrics, n_timesteps_acc = self._accumulate_metrics(
                    metrics, metrics_i, n_timesteps_acc
                )

                # generate plots
                if not val_plots:
                    batch_idx = torch.randint(0, len(idx_data["timestep_index"]), (1,)).item()
                    preds_plots = {
                        "df": preds.get("df")[batch_idx],
                        "phi": preds.get("phi_int")[batch_idx] if preds.get("phi_int") is not None else None,
                        "flux": preds.get("flux_int") # flux_int is already averaged or a scalar per sample
                    }
                    plot_tgts = {k: tgts[k][batch_idx] for k in tgts}

                    val_plots.update(
                        generate_val_plots(
                            rollout=preds_plots,
                            gt=plot_tgts,
                            ts=sample.timestep,
                            phase="random draw" if val_idx == 0 else "holdout samples",
                        )
                    )

            # sync and finalize
            metrics, n_timesteps_acc = self._sync_metrics(
                metrics, n_timesteps_acc, device, world_size
            )
            log_metric_dict = self._finalize_logs(
                log_metric_dict, metrics, n_timesteps_acc, valname
            )

        # save checkpoint
        val_loss = metrics.get("df", torch.tensor(0.0)).mean().item()
        loss_val_min = self._save_checkpoint(
            rank, model, opt, scheduler, epoch, val_loss, loss_val_min
        )

        return log_metric_dict, val_plots, loss_val_min


@torch.no_grad()
def evaluate(*args: Any, **kwargs: Any) -> None:
    """Legacy wrapper for evaluate function."""
    pass
