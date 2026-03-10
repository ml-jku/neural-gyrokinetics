"""Evaluation pipeline for the GySwin model."""

from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from einops import rearrange
from collections import defaultdict

from neugk.evaluate import BaseEvaluator, validation_metrics
from neugk.utils import recombine_zf
from neugk.plot_utils import generate_val_plots


class GyroSwinEvaluator(BaseEvaluator):
    """Evaluator for GyroSwin models supporting autoregressive rollouts."""

    def get_target_rollout(
        self,
        output_fields: List[str],
        dataset: Dataset,
        idx_data: Dict[str, torch.Tensor],
        n_eval_steps: int,
        bundle_seq_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Fetch ground truth trajectory targets for rollout comparison."""
        tgts = defaultdict(list)
        # fetch sequence
        for t in range(0, n_eval_steps, bundle_seq_length):
            sample = dataset.get_at_time(
                idx_data["file_index"].long(),
                (idx_data["timestep_index"] + t).long(),
                get_normalized=False,
                num_workers=4,
            )
            for key in output_fields:
                tgts[key].append(getattr(sample, f"y_{key}"))

        # stack fields
        for key, val in tgts.items():
            if bundle_seq_length == 1:
                tgts[key] = torch.stack(val, 0)
            elif key == "flux":
                tgts[key] = rearrange(torch.stack(val, 1), "b t -> t b")
            elif key == "phi":
                tgts[key] = rearrange(torch.stack(val, 1), "b t ... -> t b ...")
            else:
                tgts[key] = rearrange(torch.stack(val, 2), "b c t ... -> t b c ...")
        return dict(tgts)

    def _rollout(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        idx_data: Dict[str, torch.Tensor],
        conds: Dict[str, torch.Tensor],
        valset: Dataset,
        n_steps: int,
        bundle_steps: int,
        predict_delta: bool,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Perform autoregressive model rollouts."""
        # compute actual steps
        rollout_steps = min(
            [
                (valset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i]))
                // bundle_steps
                - 1
                for i, f_idx in enumerate(idx_data["file_index"].tolist())
            ]
            + [n_steps]
        )

        # initialization
        tot_ts = rollout_steps * bundle_steps
        inputs_t = inputs.copy()
        preds = defaultdict(list)

        # compute timesteps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, bundle_steps))
            for ts in idx_data["timestep_index"].tolist()
        ]
        tsteps = valset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))
        fluxes = []

        # rollout loop
        with torch.no_grad():
            for i in range(rollout_steps):
                conds["timestep"] = tsteps[:, i].to(device)
                pred = model(**inputs_t, **conds)

                if "flux" in pred:
                    fluxes.append(pred.pop("flux").cpu())

                if predict_delta:
                    for k in pred:
                        pred[k] = pred[k] + inputs_t[k]

                # update next input
                for k in inputs_t:
                    if k in pred:
                        inputs_t[k] = pred[k].clone().float()

                # cache predictions
                for k, v in pred.items():
                    preds[k].append(
                        v.cpu().unsqueeze(2) if v.ndim in [4, 5, 7] else v.cpu()
                    )

        # finalize predictions
        for k, v in preds.items():
            if "position" in inputs_t:
                preds[k] = torch.cat([p.unsqueeze(2) for p in v], 2)
            else:
                preds[k] = torch.cat(v, 2)
            preds[k] = rearrange(preds[k], "b c t ... -> t b c ...")[
                : rollout_steps * bundle_steps
            ]

        if fluxes:
            preds["flux"] = rearrange(torch.stack(fluxes, dim=-1), "b t -> t b")

        return {k: p.to(dtype=torch.float32) for k, p in preds.items()}

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
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        """Run rollout evaluation on all validation sets and compute physics metrics."""
        if not self._is_eval_epoch(epoch):
            return {}, {}, loss_val_min

        # constants setup
        n_eval_steps = self.cfg.validation.n_eval_steps
        bundle_seq_length = self.cfg.model.bundle_seq_length
        tot_eval_steps = n_eval_steps * bundle_seq_length

        predict_delta = self.cfg.training.predict_delta
        eval_integrals = self.cfg.validation.eval_integrals

        # field mapping
        input_fields = list(self.cfg.dataset.input_fields)
        if self.cfg.model.name in ["pointnet", "transolver", "transformer"]:
            input_fields.append("position")

        output_fields = [
            k
            for k, w in self.cfg.model.loss_weights.items()
            if w > 0.0 or self.cfg.model.loss_scheduler[k]
        ]
        if set(output_fields) != {"df", "phi", "flux"}:
            eval_integrals = False
        if eval_integrals:
            output_fields = ["df", "phi", "flux"]

        # prepare modules
        model.eval()
        if self.loss_wrap:
            self.loss_wrap.eval().cpu()

        log_metric_dict: Dict[str, float] = {}
        val_plots: Dict[str, Any] = {}

        # validation loop
        for val_idx, (valset, valloader) in enumerate(
            zip(self.valsets, self.valloaders)
        ):
            valname = "val_traj" if val_idx == 0 else "val_samples"
            metrics = {
                k: torch.zeros([tot_eval_steps]) for k in self.loss_wrap.all_losses
            }
            n_timesteps_acc = torch.zeros([tot_eval_steps])

            valloader = self.get_iterator(valloader, val_idx, rank)

            # batch loop
            for idx, sample in enumerate(valloader):
                inputs = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in input_fields
                    if getattr(sample, k) is not None
                }
                conds = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in self.cfg.model.conditioning
                    if getattr(sample, k) is not None
                }
                idx_data = {
                    k: getattr(sample, k).to(device)
                    for k in ["file_index", "timestep_index"]
                }

                # run rollout
                tgts = self.get_target_rollout(
                    output_fields, valset, idx_data, n_eval_steps, bundle_seq_length
                )
                rollout = self._rollout(
                    model,
                    inputs,
                    idx_data,
                    conds,
                    valset,
                    n_eval_steps,
                    bundle_seq_length,
                    predict_delta,
                    device,
                )

                # denormalize
                rollout = self._denormalize_rollout(
                    rollout, idx_data, valset.denormalize, dataset=valset
                )
                tgts = {k: v.cpu() for k, v in tgts.items()}
                rollout = {k: v.cpu() for k, v in rollout.items()}

                # handle zonal flow
                if self.cfg.dataset.separate_zf:
                    rollout["df"] = recombine_zf(rollout["df"], dim=2)
                    tgts["df"] = recombine_zf(tgts["df"], dim=2)

                # compute validation metrics
                metrics_i, integrated_i = validation_metrics(
                    tgts=tgts,
                    preds=rollout,
                    geometry=sample.geometry,
                    loss_wrap=self.loss_wrap,
                    eval_integrals=eval_integrals,
                )

                if integrated_i and integrated_i[0] is not None:
                    rollout["phi_int"] = torch.stack([i["phi"] for i in integrated_i])

                # accumulate
                metrics, n_timesteps_acc = self._accumulate_metrics(
                    metrics, metrics_i, n_timesteps_acc
                )

                # update plots
                if "position" not in inputs and not val_plots:
                    batch_idx = torch.randint(
                        0, len(idx_data["timestep_index"]), (1,)
                    ).item()
                    rollout_plot = {k: v[:, batch_idx] for k, v in rollout.items()}
                    plot_gt = {k: v[0][batch_idx] for k, v in tgts.items()}
                    val_plots.update(
                        generate_val_plots(
                            rollout_plot,
                            plot_gt,
                            conds["timestep"],
                            "random draw" if val_idx == 0 else "holdout samples",
                        )
                    )

            # sync and finalize
            metrics, n_timesteps_acc = self._sync_metrics(
                metrics, n_timesteps_acc, device, world_size
            )
            log_metric_dict = self._finalize_logs(
                log_metric_dict, metrics, n_timesteps_acc, valname
            )

        # persist checkpoints
        loss_val_min = self._save_checkpoint(
            rank, model, opt, scheduler, epoch, log_metric_dict, loss_val_min
        )

        return log_metric_dict, val_plots, loss_val_min
