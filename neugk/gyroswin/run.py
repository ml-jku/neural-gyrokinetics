"""Workflow execution for the GyroSwin model."""

import os

from collections import defaultdict
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.utils._pytree import tree_map
from time import perf_counter_ns

from neugk.utils import (
    remainig_progress,
    exclude_from_weight_decay,
    load_model_and_config,
)
from neugk.dataset import CycloneSample
from neugk.losses import LossWrapper, GradientBalancer
from neugk.runner import BaseRunner

from neugk.gyroswin.models import get_model
from neugk.losses import get_pushforward_fn
from neugk.gyroswin.eval.eval import GyroSwinEvaluator


class GyroSwinRunner(BaseRunner):
    """Workflow runner for GyroSwin models supporting multitasking and pushforward losses."""

    def setup_components(self):
        """Initialize GyroSwin model, optimizer, loss functions, and gradient balancer."""
        # model setup
        self.model = get_model(self.cfg, dataset=self.trainset).to(self.device)
        if self.use_ddp:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], find_unused_parameters=True
            )

        # load checkpoints
        if self.cfg.load_ckpt:
            ckpt_path = os.path.join(self.cfg.output_path, "ckp.pth")
            self.model, ckpt_dict = load_model_and_config(
                ckpt_path, self.model, self.device, for_ddp=self.use_ddp
            )
            # handle parameter freezing
            if self.cfg.training.params_to_include:
                for n, p in self.model.named_parameters():
                    for key in self.cfg.training.params_to_include:
                        if key not in n:
                            p.requires_grad = False
            self.start_epoch = ckpt_dict.get("epoch", 0)
            self.cur_update_step = self.start_epoch * len(self.trainloader)

            # optimizer state loading happens after opt init
            self._ckpt_dict = ckpt_dict  # temp store

        # setup parameters for optimizer
        if self.cfg.training.exclude_from_wd is not None:
            params = exclude_from_weight_decay(
                self.model,
                self.cfg.training.exclude_from_wd,
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            params = self.model.parameters()

        # optimizer setup
        self.opt = torch.optim.Adam(
            params,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            betas=(0.9, 0.95),
        )

        # restore optimizer state
        if hasattr(self, "_ckpt_dict"):
            try:
                self.opt.load_state_dict(self._ckpt_dict["optimizer_state_dict"])
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")
            del self._ckpt_dict

        # losses setup
        weights = self.setup_common_losses(self.cfg.model)
        self.loss_wrap = LossWrapper(
            weights=weights,
            schedulers=self.loss_scheduler_dict,
            denormalize_fn=self.trainset.denormalize,
            separate_zf=self.cfg.dataset.separate_zf,
            real_potens=self.cfg.dataset.real_potens,
        )

        # grad balancer setup
        n_tasks = len(self.loss_wrap.active_losses)
        self.grad_balancer = GradientBalancer(
            self.opt,
            mode=self.cfg.training.gradnorm_balancer,
            scaler=self.scaler,
            clip_grad=self.cfg.training.clip_grad,
            n_tasks=n_tasks,
        )

        # pushforward logic setup
        pf_cfg = self.cfg.training.pushforward
        self.pushforward_fn = None
        if sum(pf_cfg.unrolls) > 0:
            self.pushforward_fn = get_pushforward_fn(
                n_unrolls_schedule=pf_cfg.unrolls,
                probs_schedule=pf_cfg.probs,
                epoch_schedule=pf_cfg.epochs,
                predict_delta=self.cfg.training.predict_delta,
                dataset=self.trainset,
                bundle_steps=self.cfg.model.bundle_seq_length,
                use_amp=self.cfg.amp.enable,
                use_bf16=self.use_bf16,
                device=self.device,
            )

        # fields configuration
        self.input_fields = set(self.cfg.dataset.input_fields)
        if self.cfg.model.name in ["pointnet", "transolver", "transformer"]:
            self.input_fields.add("position")
        self.output_fields = list(
            (set(self.cfg.model.loss_weights.keys())).union(
                [k.split("_")[0] for k in self.cfg.model.extra_loss_weights.keys()]
            )
        )
        self.compute_integrals = (
            True if set(self.output_fields) != set(["flux", "phi", "df"]) else False
        )
        self.idx_keys = ["file_index", "timestep_index"]

        # setup evaluator
        self.evaluator = GyroSwinEvaluator(
            cfg=self.cfg,
            valsets=self.valsets,
            valloaders=self.valloaders,
            loss_wrap=self.loss_wrap,
        )

    def train_epoch(self, epoch):
        """Run one training epoch with multitasking and pushforward updates."""
        loss_logs = defaultdict(float)
        info_dict = defaultdict(list)
        t_start_data = perf_counter_ns()

        for _, sample in enumerate(self.pbar):
            try:
                reset_peak_memory_stats(self.device)
            except Exception:
                pass

            sample: CycloneSample
            # gather inputs
            inputs = {
                k: getattr(sample, k).to(self.device, non_blocking=True)
                for k in self.input_fields
                if getattr(sample, k) is not None
            }
            gts = {
                k: getattr(sample, f"y_{k}").to(self.device, non_blocking=True)
                for k in self.output_fields
                if getattr(sample, f"y_{k}") is not None
            }
            conds = {
                k: getattr(sample, k).to(self.device, non_blocking=True)
                for k in self.cfg.model.conditioning
                if getattr(sample, k) is not None
            }
            idx_data = {k: getattr(sample, k).to(self.device) for k in self.idx_keys}
            geometry = tree_map(lambda g: g.to(self.device), sample.geometry)

            # augmentations
            if self.augmentations:
                for aug_fn in self.augmentations:
                    inputs = {k: aug_fn(v) for k, v in inputs.items()}

            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

            # pushforward unroll
            if self.pushforward_fn:
                start_pf = perf_counter_ns()
                inputs, gts, conds = self.pushforward_fn(
                    self.model, inputs, gts, conds, idx_data, epoch
                )
                info_dict["pf_ms"].append((perf_counter_ns() - start_pf) / 1e6)
            else:
                info_dict["pf_ms"].append(0.0)

            # forward pass and loss
            t_start_fwd = perf_counter_ns()
            with torch.autocast(
                str(self.device), dtype=self.amp_dtype, enabled=self.use_amp
            ):
                preds = self.model(**inputs, **conds)
                if self.cfg.training.predict_delta:
                    for key in self.cfg.dataset.input_fields:
                        preds[key] = preds[key] + inputs[key]

                loss, losses = self.loss_wrap(
                    preds,
                    gts,
                    idx_data,
                    geometry=geometry,
                    progress_remaining=remainig_progress(
                        self.cur_update_step, self.total_steps
                    ),
                    separate_zf=(
                        self.cfg.dataset.separate_zf
                        if self.cfg.model.extra_zf_loss
                        else False
                    ),
                    compute_integrals=self.compute_integrals,
                )

            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

            # optimize
            self.model = self.grad_balancer(self.model, loss, list(losses.values()))
            if self.scheduler:
                self.scheduler.step()

            self.cur_update_step += 1.0

            # record logs
            for k, v in losses.items():
                loss_logs[k] += v.item()
            loss_logs["relative_norm"] += loss.item()

            del inputs, gts, losses, loss

            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(self.device) / 1024**2)
            t_start_data = perf_counter_ns()

        # finalize epoch statistics
        n_batches = len(self.trainloader)
        loss_logs = {k: v / n_batches for k, v in loss_logs.items()}
        return loss_logs, info_dict

    def _log_epoch(self, epoch, epoch_logs, info_dict, val_plots):
        # gyroswin specific renames
        gyro_logs = {
            (k if any(x in k for x in ["lr", "schedule", "val"]) else f"{k}_mse"): v
            for k, v in epoch_logs.items()
        }
        super()._log_epoch(epoch, gyro_logs, info_dict, val_plots)

    def evaluate(self, epoch):
        """Call evaluation pipeline for current epoch."""
        return self.evaluator(
            self.rank,
            self.world_size,
            self.model,
            self.opt,
            self.scheduler,
            epoch,
            self.device,
            self.loss_val_min,
        )
