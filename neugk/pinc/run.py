import os
from tqdm import tqdm

from collections import defaultdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.utils._pytree import tree_map
from time import perf_counter_ns

from neugk.dataset import CycloneAESample
from neugk.utils import remainig_progress, exclude_from_weight_decay, memory_cleanup
from neugk.runner import BaseRunner

from neugk.pinc.autoencoders import get_autoencoder
from neugk.pinc.losses import PINCLossWrapper, PINCGradientBalancer
from neugk.pinc.autoencoders.evaluate import evaluate as pinc_evaluate
from neugk.pinc.autoencoders.evaluate_simsiam import evaluate_linear_probe
from neugk.pinc.autoencoders.ae_utils import (
    aggregate_dataset_stats,
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    load_autoencoder,
    train_step_autoencoder,
    train_step_peft,
    train_step_simsiam,
)
from neugk.pinc.peft_utils import setup_peft_stage


class PINCRunner(BaseRunner):
    def setup_components(self):
        assert self.cfg.stage is not None, "Stage is not set."

        model_key = "autoencoder" if hasattr(self.cfg, "autoencoder") else "model"
        model_cfg = getattr(self.cfg, model_key)

        # losses
        weights = self.setup_common_losses(model_cfg)
        # dataset_stats = (
        #     aggregate_dataset_stats(self.trainset.files)
        #     if hasattr(self.trainset, "files")
        #     else {}
        # )
        dataset_stats = {}

        # pinc specific losses
        self.loss_wrap = PINCLossWrapper(
            weights=weights,
            schedulers=self.loss_scheduler_dict,
            denormalize_fn=self.trainset.denormalize,
            separate_zf=self.cfg.dataset.separate_zf,
            real_potens=self.cfg.dataset.real_potens,
            integral_loss_type=getattr(self.cfg.training, "integral_loss_type", "mse"),
            spectral_loss_type=getattr(self.cfg.training, "spectral_loss_type", "l1"),
            dataset_stats=dataset_stats,
            ds=getattr(self.cfg.training, "ds", None),
            ema_normalization_loss=getattr(
                self.cfg.training, "ema_normalization_loss", None
            ),
            ema_beta=getattr(self.cfg.training, "ema_beta", 0.99),
            eval_loss_type=getattr(self.cfg.training, "eval_loss_type", "mse"),
            eval_integral_loss_type=getattr(
                self.cfg.training, "eval_integral_loss_type", "mse"
            ),
            eval_spectral_loss_type=getattr(
                self.cfg.training, "eval_spectral_loss_type", "l1"
            ),
        )

        self.model = get_autoencoder(
            self.cfg, dataset=self.trainset, rank=self.rank
        ).to(self.device)

        # peft setup vs standard setup
        self._load_checkpoints()

        self.simae = len(set(self.loss_wrap.active_losses).difference({"simsiam"})) > 0
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                # NOTE unused only if no decode
                find_unused_parameters=(self.cfg.stage == "simsiam"),
            )

        is_muon = (
            hasattr(self.cfg.training, "optimizer")
            and self.cfg.training.optimizer == "muon"
        )

        if is_muon:
            param_groups = self._split_muon_param_groups(self.model)
            self.opt = (
                MuonWithAuxAdam(param_groups)
                if self.use_ddp
                else SingleDeviceMuonWithAuxAdam(param_groups)
            )
            self.opt.defaults = {"lr": self.cfg.training.learning_rate}
        elif self.cfg.training.gradnorm_balancer == "pseudo":
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.opt = torch.optim.SGD(params, lr=self.cfg.training.learning_rate)
        else:
            # Standard Adam
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise ValueError("No trainable params")
            exclude = getattr(self.cfg.training, "exclude_from_wd", [])
            if exclude:
                groups = exclude_from_weight_decay(
                    self.model, exclude, self.cfg.training.weight_decay
                )
                self.opt = torch.optim.Adam(groups, lr=self.cfg.training.learning_rate)
            else:
                self.opt = torch.optim.Adam(
                    params,
                    lr=self.cfg.training.learning_rate,
                    weight_decay=self.cfg.training.weight_decay,
                )

        # gradient balancer
        self.grad_balancer = PINCGradientBalancer(
            self.opt,
            mode=self.cfg.training.gradnorm_balancer,
            scaler=self.scaler,
            clip_grad=self.cfg.training.clip_grad,
            n_tasks=len(self.loss_wrap.active_losses),
        )

    def _load_checkpoints(self):
        # TODO workaround to load. as of now loading does not work
        ckpt_path = os.path.join(
            self.cfg.output_path,
            "..",
            (
                "ckp.pth"
                if getattr(self.cfg.training, "use_latest_checkpoint", False)
                else "best.pth"
            ),
        )

        self.ae_ckpt_dict = {}

        if self.cfg.stage == "peft":
            if not ckpt_path and not os.path.exists(ckpt_path):
                raise ValueError("PEFT requires ae_checkpoint")

            print(f"Loading checkpoint: {ckpt_path}")
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            is_peft_ckpt = loaded_ckpt.get("stage") == "peft"

            if is_peft_ckpt:
                self.model, self.ae_ckpt_dict = load_autoencoder(
                    ckpt_path, model=self.model, device=self.device, load_peft=True
                )
                self.start_epoch = self.ae_ckpt_dict["epoch"]
                self.loss_val_min = self.ae_ckpt_dict["loss"]
                print(f"Resumed PEFT from epoch {self.start_epoch}")
            else:
                self.model, self.ae_ckpt_dict = load_autoencoder(
                    ckpt_path, model=self.model, device=self.device
                )
                print(f"Loaded base AE from epoch {self.ae_ckpt_dict['epoch']}")
                self.model, _ = setup_peft_stage(
                    self.model, self.cfg, dataloader=self.trainloader
                )
                self.start_epoch = 0
        elif ckpt_path and os.path.exists(ckpt_path):
            # resume standard AE
            self.model, self.ae_ckpt_dict = load_autoencoder(
                ckpt_path, model=self.model, device=self.device
            )
            self.start_epoch = self.ae_ckpt_dict["epoch"]
            self.loss_val_min = self.ae_ckpt_dict["loss"]

        self.cur_update_step = self.start_epoch * len(self.trainloader)

    def _split_muon_param_groups(self, model):
        muon, adam_decay, adam_no_decay = [], [], []
        exclude_wd = getattr(self.cfg.training, "exclude_from_wd", [])

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.cfg.stage == "peft":
                # PEFT specific separation
                if (
                    any(k in name for k in ["lora_A", "lora_B", "eva_"])
                    and param.ndim >= 2
                ):
                    muon.append(param)
                else:
                    adam_decay.append(param)
            else:
                # Standard AE separation
                if param.ndim >= 2 and not any(
                    x in name.lower()
                    for x in ["embed", "head", "pos_embed", "cls_token"]
                ):
                    muon.append(param)
                else:
                    if any(x in name.lower() for x in exclude_wd):
                        adam_no_decay.append(param)
                    else:
                        adam_decay.append(param)

        groups = [
            {
                "params": muon,
                "use_muon": True,
                "lr": self.cfg.training.learning_rate * 100,
                "weight_decay": self.cfg.training.weight_decay,
                "momentum": 0.95,
            },
            {
                "params": adam_decay,
                "use_muon": False,
                "lr": self.cfg.training.learning_rate,
                "betas": (0.9, 0.95),
                "weight_decay": self.cfg.training.weight_decay,
            },
        ]
        if adam_no_decay:
            groups.append(
                {
                    "params": adam_no_decay,
                    "use_muon": False,
                    "lr": self.cfg.training.learning_rate,
                    "betas": (0.9, 0.95),
                    "weight_decay": 0.0,
                }
            )
        return groups

    def __call__(self):
        # load states now that everything is initialized
        if self.ae_ckpt_dict:
            try:
                if "optimizer_state_dict" in self.ae_ckpt_dict:
                    self.opt.load_state_dict(self.ae_ckpt_dict["optimizer_state_dict"])
                if "scheduler_state_dict" in self.ae_ckpt_dict and self.scheduler:
                    # self.scheduler.load_state_dict(
                    #     self.ae_ckpt_dict["scheduler_state_dict"]
                    # )
                    pass
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")

        # input fields caching
        self.input_fields = set(self.cfg.dataset.input_fields)
        self.idx_keys = ["file_index", "timestep_index"]

        for epoch in range(self.start_epoch + 1, self.cfg.training.n_epochs + 1):
            if self.cfg.logging.tqdm and (not self.use_ddp or not self.rank):
                self.pbar = tqdm(self.trainloader, "Training")
            else:
                self.pbar = self.trainloader

            # training step
            self.model.train()
            self.loss_wrap.train().to(self.device)
            loss_logs, info_dict = self.train_epoch(epoch)

            loss_type = getattr(self.cfg.training, "loss_type", "mse")
            formatted_logs = {}
            for k, v in loss_logs.items():
                new_k = k
                if "total_mse" not in k:
                    if "df" in k:
                        new_k = new_k.replace("df", f"df_{loss_type}")
                    if "total" in k:
                        new_k = new_k.replace("total", f"total_{loss_type}")
                    new_k = f"train/{new_k}"
                else:
                    new_k = f"train/{k}"
                formatted_logs[new_k] = v

            progress = remainig_progress(self.cur_update_step, self.total_steps)
            train_losses_dict = {
                "train/lr": (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler
                    else self.cfg.training.learning_rate
                ),
            }
            for k, sched in self.loss_scheduler_dict.items():
                train_losses_dict[f"train/{k}_schedule"] = sched(progress)

            train_losses_dict.update(formatted_logs)
            info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

            log_metric_dict, val_plots = self.evaluate(epoch)
            self._log_epoch(
                epoch, train_losses_dict | log_metric_dict, info_dict, val_plots
            )

            if (self.cur_update_step % 100) == 0:
                memory_cleanup(self.device, aggressive=True)

        if self.writer:
            self.writer.finish()
        if self.use_ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        loss_logs = defaultdict(list)
        info_dict = defaultdict(list)
        t_start_data = perf_counter_ns()

        for sample in self.pbar:
            try:
                reset_peak_memory_stats(self.device)
            except:
                pass

            sample: CycloneAESample
            xs = {
                k: getattr(sample, k).to(self.device, non_blocking=True)
                for k in self.input_fields
                if getattr(sample, k) is not None
            }
            condition = None
            if sample.conditioning is not None:
                condition = sample.conditioning.to(self.device)
            idx_data = {k: getattr(sample, k).to(self.device) for k in self.idx_keys}
            geometry = tree_map(lambda g: g.to(self.device), sample.geometry)
            if self.augmentations:
                for aug_fn in self.augmentations:
                    xs = {k: aug_fn(v, idx_data["file_index"]) for k, v in xs.items()}
                    if self.cfg.dataset.augment.mask_modes.active:
                        # separate input and target
                        xs["df"], xs["df_tgt"] = xs["df"]

            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)
            t_start_fwd = perf_counter_ns()

            with torch.autocast(
                str(self.device), dtype=self.amp_dtype, enabled=self.use_amp
            ):
                # dispatch to correct step function
                if self.cfg.stage == "autoencoder":
                    step_fn = train_step_autoencoder
                if self.cfg.stage == "peft":
                    step_fn = train_step_peft
                if self.cfg.stage == "simsiam":
                    step_fn = train_step_simsiam
                    xs["df_aug"] = getattr(sample, "df_aug").to(self.device)

                loss, losses = step_fn(
                    self.cfg,
                    model=self.model,
                    xs=xs,
                    condition=condition,
                    idx_data=idx_data,
                    geometry=geometry,
                    loss_wrap=self.loss_wrap,
                    progress_remaining=remainig_progress(
                        self.cur_update_step, self.total_steps
                    ),
                )

            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

            # grad balancer filtering
            grad_losses = [
                v
                for k, v in losses.items()
                if k in self.loss_wrap.active_losses
                and k not in ["total_mse", "phi_int_mse", "flux_int_mse", "df_delta"]
                and not k.endswith("_mse")
                and v.requires_grad
            ]

            self.model = self.grad_balancer(self.model, loss, grad_losses)
            if self.scheduler:
                self.scheduler.step()

            self.cur_update_step += 1.0

            # log accumulation (list based in original)
            loss_logs["total"].append(loss.item())
            for k, v in losses.items():
                loss_logs[k].append(v.item())

            if (self.cur_update_step % 100) == 0:
                del xs, condition, idx_data, geometry, loss, losses
                memory_cleanup(self.device, aggressive=True)

            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(self.device) / 1024**2)
            t_start_data = perf_counter_ns()

        # average logs
        loss_logs = {k: sum(v) / max(len(v), 1) for k, v in loss_logs.items()}
        return loss_logs, info_dict

    def evaluate(self, epoch):
        if getattr(self.model, "use_simae_decoder", True):
            log_metric_dict, val_plots, self.loss_val_min = pinc_evaluate(
                rank=self.rank,
                world_size=self.world_size,
                model=self.model,
                loss_wrap=self.loss_wrap,
                valsets=self.valsets,
                valloaders=self.valloaders,
                opt=self.opt,
                lr_scheduler=self.scheduler,
                epoch=epoch,
                cfg=self.cfg,
                device=self.device,
                loss_val_min=self.loss_val_min,
            )
        if self.cfg.stage == "simsiam":
            probe_log_metric_dict, _ = evaluate_linear_probe(
                rank=self.rank,
                model=self.model,
                trainloader=self.trainloader,
                valloaders=self.valloaders,
                epoch=epoch,
                cfg=self.cfg,
                device=self.device,
                loss_val_min=self.loss_val_min,
            )
            log_metric_dict = log_metric_dict | probe_log_metric_dict
        return log_metric_dict, val_plots
