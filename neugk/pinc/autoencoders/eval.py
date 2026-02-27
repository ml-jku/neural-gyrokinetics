from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from neugk.dataset.cyclone_diff import CycloneAESample
from neugk.evaluate import BaseEvaluator, validation_metrics
from neugk.plot_utils import generate_val_plots


class AutoencoderEvaluator(BaseEvaluator):
    """Evaluator for autoencoder models with optional linear probing."""

    def _prepare_sample(self, sample: CycloneAESample, device: torch.device) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
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
            sample.conditioning.to(device) if sample.conditioning is not None else None
        )
        idx_data = {
            k: getattr(sample, k).to(device) for k in ["file_index", "timestep_index"]
        }
        return xs, tgts, condition, idx_data

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
            sample: CycloneAESample
            xs = sample.df.to(device, non_blocking=True)
            condition = (
                sample.conditioning.to(device, non_blocking=True)
                if sample.conditioning is not None
                else None
            )
            flux = sample.flux.to(device, non_blocking=True)

            # forward pass for latents
            if hasattr(model, "encode"):
                z, _ = model.encode(xs, condition=condition)
            else:
                # DDP wrapper
                z, _ = model.module.encode(xs, condition=condition)

            # global average pool spatially
            zpool = z.view(z.shape[0], -1, z.shape[-1]).mean(1)
            latents.append(zpool.cpu())
            fluxes.append(flux.view(flux.shape[0], -1).cpu())

        return torch.cat(latents, 0), torch.cat(fluxes, 0)

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
        trainloader: Optional[torch.utils.data.DataLoader] = None,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """Evaluate autoencoder model on validation datasets"""
        log_metric_dict: Dict[str, float] = {}
        val_plots: Dict[str, Any] = {}

        if not self._is_eval_epoch(epoch):
            return {}, {}, loss_val_min

        # standard autoencoder reconstruction evaluation
        if getattr(model, "use_simae_decoder", True):
            model.eval()
            if self.loss_wrap:
                self.loss_wrap.eval().cpu()

            eval_integrals = self.cfg.validation.eval_integrals

            for val_idx, (valset, valloader) in enumerate(
                zip(self.valsets, self.valloaders)
            ):
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
                    preds = self._denormalize_batch(
                        preds, idx_data, valset.denormalize, dataset=valset
                    )
                    tgts = self._denormalize_batch(
                        tgts, idx_data, valset.denormalize, dataset=valset
                    )

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
                        batch_idx = torch.randint(
                            0, len(idx_data["timestep_index"]), (1,)
                        ).item()
                        preds_plots = {
                            "df": preds.get("df")[batch_idx],
                            "phi": (
                                preds.get("phi_int")[batch_idx]
                                if preds.get("phi_int") is not None
                                else None
                            ),
                            "flux": preds.get(
                                "flux_int"
                            ),  # flux_int is already averaged or a scalar per sample
                        }
                        plot_tgts = {k: tgts[k][batch_idx] for k in tgts}

                        val_plots.update(
                            generate_val_plots(
                                rollout=preds_plots,
                                gt=plot_tgts,
                                ts=sample.timestep,
                                phase=(
                                    "random draw" if val_idx == 0 else "holdout samples"
                                ),
                            )
                        )

                # sync and finalize
                metrics, n_timesteps_acc = self._sync_metrics(
                    metrics, n_timesteps_acc, device, world_size
                )
                log_metric_dict = self._finalize_logs(
                    log_metric_dict, metrics, n_timesteps_acc, valname
                )

        # linear probing evaluation
        if trainloader is not None and getattr(self.cfg.validation, "probing", True):
            # compute probe weights on trainset
            x_train, y_train = self.collect_xy(rank, trainloader, model, device)
            # column of ones for bias
            x_train_b = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)
            # solve linear system: w @ w = y
            res = torch.linalg.lstsq(x_train_b, y_train, driver="gels")
            w = res.solution
            # report train RMSE on physical scale
            y_train_pred = x_train_b @ w
            train_rmse = torch.sqrt(torch.mean((y_train_pred - y_train) ** 2))
            log_metric_dict["val_traj/probe_train_rmse"] = train_rmse.item()
            # evaluate on validation sets
            for val_idx, valloader in enumerate(self.valloaders):
                valname = "val_traj" if val_idx == 0 else "val_samples"
                x_val, y_val = self.collect_xy(
                    rank, valloader, model, device, desc=None
                )
                x_val_b = torch.cat([x_val, torch.ones(x_val.shape[0], 1)], dim=1)
                y_val_pred = x_val_b @ w
                val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val) ** 2))
                log_metric_dict[f"{valname}/probe_val_rmse"] = val_rmse.item()
        # save checkpoint
        sel_metric = (
            f"val_traj/{self.cfg.validation.get('model_selection_metric', 'df')}"
        )
        val_loss = log_metric_dict.get(sel_metric, 0.0)
        loss_val_min = self._save_checkpoint(
            rank, model, opt, scheduler, epoch, val_loss, loss_val_min
        )

        return log_metric_dict, val_plots, loss_val_min
