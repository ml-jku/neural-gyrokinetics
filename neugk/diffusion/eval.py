"""Evaluation utilities for diffusion models."""

import re
from typing import Callable, Optional, Dict, Tuple, Any
from collections import defaultdict
from functools import partial

import torch
import numpy as np
import torch.distributed as dist

from neugk.evaluate import BaseEvaluator, validation_metrics
from neugk.plot_utils import generate_val_plots, avg_flux_confidence


class DiffusionEvaluator(BaseEvaluator):
    """Evaluator for diffusion models."""

    @torch.no_grad()
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
        sample_fn: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """Run evaluation on multiple validation sets and log metrics."""
        if not self._is_eval_epoch(epoch):
            return {}, {}, loss_val_min

        assert sample_fn is not None, "Diffusion evaluation requires a sample_fn"

        # validate loop parameters
        eval_integrals = self.cfg.validation.eval_integrals
        idx_keys = ["file_index", "timestep_index"]
        log_metric_dict: Dict[str, float] = {}
        val_plots: Dict[str, Any] = {}

        if self.loss_wrap:
            self.loss_wrap.eval().cpu()

        for val_idx, (valset, valloader) in enumerate(
            zip(self.valsets, self.valloaders)
        ):
            valname = "val_traj" if val_idx == 0 else "val_samples"
            metrics = {key: torch.tensor(0.0) for key in self.loss_wrap.all_losses}
            n_timesteps_acc = torch.tensor(0.0)

            valloader = self.get_iterator(valloader, val_idx, rank)

            # collect stats
            all_preds_flux = []
            all_tgts_flux = []

            for idx, sample in enumerate(valloader):
                # prepare targets
                tgts = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in ["df", "phi", "flux", "avg_flux"]
                    if getattr(sample, k) is not None
                }
                condition = sample.conditioning.to(device)
                idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

                # compute predictions
                preds = sample_fn(condition)

                # denormalize
                preds = self._denormalize_batch(
                    preds,
                    idx_data=idx_data,
                    denormalize_fn=partial(valset.denormalize, condition=condition),
                    dataset=valset,
                )
                tgts = self._denormalize_batch(
                    tgts,
                    idx_data=idx_data,
                    denormalize_fn=valset.denormalize,
                    dataset=valset,
                )

                tgts = {k: v.cpu() for k, v in tgts.items()}
                preds = {k: v.cpu() for k, v in preds.items()}

                # combine zonal flow
                if self.cfg.dataset.separate_zf:
                    for d in [preds, tgts]:
                        for k in d:
                            d[k] = self._recombine_zf(d[k])

                # validation metrics
                metrics_i, integrated_i = validation_metrics(
                    tgts=tgts,
                    preds=preds,
                    geometry=sample.geometry,
                    loss_wrap=self.loss_wrap,
                    eval_integrals=eval_integrals,
                )

                # store integrals
                preds["phi_int"] = integrated_i["phi"]
                preds["flux_int"] = integrated_i["eflux"]

                # flux analysis
                if preds["flux_int"] is not None and "avg_flux" in tgts:
                    all_preds_flux.append(
                        {
                            "val": preds["flux_int"].cpu(),
                            "file_index": idx_data["file_index"].cpu(),
                        }
                    )
                    all_tgts_flux.append(
                        {
                            "val": tgts["avg_flux"].cpu(),
                            "file_index": idx_data["file_index"].cpu(),
                        }
                    )

                # accumulate
                metrics, n_timesteps_acc = self._accumulate_metrics(
                    metrics, metrics_i, n_timesteps_acc
                )

                # generate plots
                if len(val_plots) == 0:
                    batch_idx = torch.randint(
                        0, len(idx_data["file_index"]), (1,)
                    ).item()
                    preds_plots = {
                        "df": preds["df"][batch_idx] if "df" in preds else None,
                        "phi": (
                            preds["phi_int"][batch_idx]
                            if preds["phi_int"] is not None
                            else None
                        ),
                        "flux": preds["flux_int"],
                    }
                    tgts_plot = {k: tgts[k][batch_idx] for k in tgts}

                    val_plots.update(
                        generate_val_plots(
                            rollout=preds_plots,
                            gt=tgts_plot,
                            ts=sample.timestep,
                            phase="Random draw" if val_idx == 0 else "Holdout samples",
                        )
                    )

            # sync and finalize
            metrics, n_timesteps_acc = self._sync_metrics(
                metrics, n_timesteps_acc, device, world_size
            )
            log_metric_dict = self._finalize_logs(
                log_metric_dict, metrics, n_timesteps_acc, valname
            )

            # flux per trajectory averaging
            if len(all_preds_flux) > 0 and (not dist.is_initialized() or rank == 0):
                fpreds = torch.cat([b["val"].flatten() for b in all_preds_flux])
                ftgts = torch.cat([b["val"].flatten() for b in all_tgts_flux])
                fidxs = torch.cat([b["file_index"].flatten() for b in all_preds_flux])

                grouped_preds = defaultdict(list)
                grouped_tgts = {}

                for p, t, i in zip(fpreds, ftgts, fidxs):
                    match = re.search(r"iteration_\d+", valset.files[i.item()])
                    if match:
                        idx = match.group()
                        grouped_preds[idx].append(p.item())
                        grouped_tgts[idx] = t.item()

                if grouped_preds:
                    traj_ids = sorted(list(grouped_preds.keys()))
                    pred_means = np.array([np.mean(grouped_preds[i]) for i in traj_ids])
                    pred_stds = np.array([np.std(grouped_preds[i]) for i in traj_ids])
                    tgt_vals = np.array([grouped_tgts[i] for i in traj_ids])

                    avg_flux_rmse = np.sqrt(((pred_means - tgt_vals) ** 2).mean())
                    log_metric_dict[f"{valname}/avg_flux_rmse"] = avg_flux_rmse.item()

                    val_plots["avg_flux_UQ"] = avg_flux_confidence(
                        pred_means, pred_stds, tgt_vals, traj_ids
                    )

        # store checkpoints
        val_loss = log_metric_dict.get("val_traj/avg_flux_rmse", 0.0)
        loss_val_min = self._save_checkpoint(
            rank, model, opt, scheduler, epoch, val_loss, loss_val_min
        )

        return log_metric_dict, val_plots, loss_val_min
