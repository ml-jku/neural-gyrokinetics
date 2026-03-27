from typing import Dict, Optional, Tuple, Any, List
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

from neugk.dataset.cyclone_diff import CycloneAESample
from neugk.evaluate import BaseEvaluator, validation_metrics
from neugk.plot_utils import generate_val_plots
from neugk.utils import recombine_zf


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

    def _gather_probe_targets(
        self,
        sample: CycloneAESample,
        dataset,
        probe_targets: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Extract probe target values from per-file dataset metadata.

        Returns a (B, n_targets) tensor.
        """
        file_indices = sample.file_index  # (B,)
        timestep_indices = sample.timestep_index  # (B,)
        batch_targets = []
        for fi, ti in zip(file_indices.tolist(), timestep_indices.tolist()):
            meta = dataset.metadata[fi]
            vals = []
            for tgt_name in probe_targets:
                v = meta[tgt_name]
                # time-indexed arrays (e.g. fluxes) vs per-file scalars (e.g. itg)
                if hasattr(v, '__len__') and len(v) > 1:
                    v = v[ti]
                stats = dataset.stats.get(tgt_name, {})
                mean = 0.0
                std = 1.0
                if len(stats):
                    mean = stats["full"]["mean"]
                    std = stats["full"]["std"]
                else:
                    warnings.warn(f"No stats found for probe target '{tgt_name}', skipping normalization.")
                if tgt_name in ["fluxspec", "kyspec"]:
                    v = np.log1p(v)  # log-transform spectra for stability
                v = (v - mean) / std  # normalize
                v_t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
                vals.append(v_t)
            batch_targets.append(torch.cat(vals))
        return torch.stack(batch_targets, dim=0)

    @torch.no_grad()
    def collect_xy(
        self,
        rank: int,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        dataset=None,
        desc: Optional[str] = "linear probe",
        probe_targets: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect model latents and probe targets for linear probing."""
        model.eval()
        latents: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

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

            # forward pass for latents
            if hasattr(model, "encode"):
                z, _ = model.encode(xs, condition=condition)
            else:
                # DDP wrapper
                z, _ = model.module.encode(xs, condition=condition)

            # global average pool spatially
            zpool = z.view(z.shape[0], -1, z.shape[-1]).mean(1)
            latents.append(zpool.cpu())

            # gather probe targets from dataset metadata
            y = self._gather_probe_targets(sample, dataset, probe_targets)
            targets.append(y)

        return torch.cat(latents, 0), torch.cat(targets, 0)

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
        evaluate_recon: bool = False,
        probe_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """Evaluate autoencoder model on validation datasets"""
        log_metric_dict: Dict[str, float] = {}
        val_plots: Dict[str, Any] = {}

        if not self._is_eval_epoch(epoch):
            return {}, {}, loss_val_min

        # standard autoencoder reconstruction evaluation
        if evaluate_recon:
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

                    # combine zonal flow
                    if self.cfg.dataset.separate_zf:
                        if "df" in preds:
                            preds["df"] = recombine_zf(preds["df"], dim=1)
                        if "df" in tgts:
                            tgts["df"] = recombine_zf(tgts["df"], dim=1)

                    # compute validation metrics
                    geometry = valset.get_batch_geometry(idx_data["file_index"])
                    metrics_i, integrated_i = validation_metrics(
                        tgts={k: v.cpu() for k, v in tgts.items()},
                        preds={k: v.cpu() for k, v in preds.items()},
                        geometry=geometry,
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
                        # Prepare single-sample plot dicts
                        preds_plots = {}
                        plot_tgts = {}

                        # Handle potentially batched fields (df, phi, flux)
                        for k in ["df", "phi", "flux"]:
                            if k in preds and preds[k] is not None:
                                preds_plots[k] = preds[k][batch_idx]
                            elif f"{k}_int" in preds and preds[f"{k}_int"] is not None:
                                # integrated fields might be (B, ...) or already (S, X, Y)
                                v = preds[f"{k}_int"]
                                preds_plots[k] = v[batch_idx] if v.ndim > 3 else v

                            if k in tgts and tgts[k] is not None:
                                plot_tgts[k] = tgts[k][batch_idx]

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
        if trainloader is not None and probe_cfg is not None:
            trainset = kwargs.get("trainset")
            probe_targets: List[str] = probe_cfg.get("targets", ["flux"])

            def _make_encode_fn(dataset):
                def encode_fn(sample, device):
                    sample: CycloneAESample
                    xs = sample.df.to(device, non_blocking=True)
                    condition = (
                        sample.conditioning.to(device, non_blocking=True)
                        if sample.conditioning is not None
                        else None
                    )
                    # forward pass for latents
                    if hasattr(model, "encode"):
                        z, _ = model.encode(xs, condition=condition)
                    else:
                        z, _ = model.module.encode(xs, condition=condition)
                    y = self._gather_probe_targets(sample, dataset, probe_targets)
                    return z, y
                return encode_fn

            self.run_probing_evaluation(
                rank=rank,
                trainloader=trainloader,
                extraction_fn=_make_encode_fn(trainset),
                device=device,
                epoch=epoch,
                log_metric_dict=log_metric_dict,
                val_plots=val_plots,
                probe_cfg=probe_cfg,
                probe_targets=probe_targets,
                val_extraction_fns=[_make_encode_fn(vs) for vs in self.valsets],
                dataset_for_stats=trainset,
            )
                
        # save checkpoint
        loss_val_min = self._save_checkpoint(
            rank, model, opt, scheduler, epoch, log_metric_dict, loss_val_min
        )

        return log_metric_dict, val_plots, loss_val_min
