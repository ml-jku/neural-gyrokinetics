import os
from functools import partial
from collections import defaultdict
from time import perf_counter_ns

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.utils._pytree import tree_map

from neugk.utils import remainig_progress, exclude_from_weight_decay
from neugk.runner import BaseRunner
from neugk.pinc.autoencoders import get_autoencoder
from neugk.pinc.losses import PINCLossWrapper, PINCGradientBalancer
from neugk.pinc.autoencoders.eval import AutoencoderEvaluator
from neugk.pinc.autoencoders.ae_utils import (
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    load_autoencoder,
    train_step_autoencoder,
    train_step_peft,
    train_step_simsiam,
)
from neugk.pinc.peft_utils import setup_peft_stage


class PINCRunner(BaseRunner):
    """PINCRunner class."""

    def setup_components(self):
        assert self.cfg.stage is not None, "stage is not set"

        model_key = "autoencoder" if hasattr(self.cfg, "autoencoder") else "model"
        model_cfg = getattr(self.cfg, model_key)

        weights = self.setup_common_losses(model_cfg)
        # dataset_stats = (
        #     aggregate_dataset_stats(self.trainset.files)
        #     if hasattr(self.trainset, "files")
        #     else {}
        # )
        dataset_stats = {}
        augmentations = [
            k
            for k in getattr(self.cfg.dataset, "augment", {}).keys()
            if getattr(self.cfg.dataset.augment, k).active
        ]

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
            augmentations=augmentations,
            dataset=self.trainset,
        )

        self.model = get_autoencoder(
            self.cfg, dataset=self.trainset, rank=self.rank
        ).to(self.device)

        self._load_checkpoints()

        self.simae = len(set(self.loss_wrap.active_losses).difference({"simsiam"})) > 0
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                # NOTE unused only if no decode
                find_unused_parameters=(self.cfg.stage == "simsiam"),
            )

        is_muon = getattr(self.cfg.training, "optimizer", "") == "muon"

        if is_muon:
            param_groups = self._split_muon_param_groups(self.model)
            opt_cls = MuonWithAuxAdam if self.use_ddp else SingleDeviceMuonWithAuxAdam
            self.opt = opt_cls(param_groups)
            self.opt.defaults = {"lr": self.cfg.training.learning_rate}
        elif self.cfg.training.gradnorm_balancer == "pseudo":
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.opt = torch.optim.SGD(params, lr=self.cfg.training.learning_rate)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise ValueError("no trainable params")
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

        self.grad_balancer = PINCGradientBalancer(
            self.opt,
            mode=self.cfg.training.gradnorm_balancer,
            scaler=self.scaler,
            clip_grad=self.cfg.training.clip_grad,
            n_tasks=len(self.loss_wrap.active_losses),
        )

        # setup evaluator
        self.evaluator = AutoencoderEvaluator(
            cfg=self.cfg,
            valsets=self.valsets,
            valloaders=self.valloaders,
            loss_wrap=self.loss_wrap,
        )
        self.input_fields = set(self.cfg.dataset.input_fields)
        self.idx_keys = ["file_index", "timestep_index"]

    def _load_checkpoints(self):
        # build checkpoint path
        use_latest = getattr(self.cfg.training, "use_latest_checkpoint", False)
        ckpt_name = "ckp.pth" if use_latest else "best.pth"
        ckpt_path = os.path.join(self.cfg.output_path, "..", ckpt_name)
        self.ae_ckpt_dict = {}

        if self.cfg.stage == "peft":
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise ValueError("peft requires ae_checkpoint")

            print(f"loading checkpoint: {ckpt_path}")
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if loaded_ckpt.get("stage") == "peft":
                self.model, self.ae_ckpt_dict = load_autoencoder(
                    ckpt_path, model=self.model, device=self.device, load_peft=True
                )
                self.start_epoch = self.ae_ckpt_dict["epoch"]
                self.loss_val_min = self.ae_ckpt_dict["loss"]
                print(f"resumed peft from epoch {self.start_epoch}")
            else:
                self.model, self.ae_ckpt_dict = load_autoencoder(
                    ckpt_path, model=self.model, device=self.device
                )
                print(f"loaded base ae from epoch {self.ae_ckpt_dict['epoch']}")
                self.model, _ = setup_peft_stage(
                    self.model, self.cfg, dataloader=self.trainloader
                )
                self.start_epoch = 0
        elif ckpt_path and os.path.exists(ckpt_path):
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
                if (
                    any(k in name for k in ["lora_A", "lora_B", "eva_"])
                    and param.ndim >= 2
                ):
                    muon.append(param)
                else:
                    adam_decay.append(param)
            else:
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

    def __call__(self, skip_eval: bool = False):
        if self.ae_ckpt_dict:
            try:
                if "optimizer_state_dict" in self.ae_ckpt_dict:
                    self.opt.load_state_dict(self.ae_ckpt_dict["optimizer_state_dict"])
            except Exception as e:
                print(f"warning: could not load optimizer state: {e}")

        super().__call__(skip_eval=skip_eval)

    def _log_epoch(self, epoch, epoch_logs, info_dict, val_plots):
        loss_type = getattr(self.cfg.training, "loss_type", "mse")
        # rename keys for PINC specific logging
        pinc_logs = {
            (
                k.replace("df", f"df_{loss_type}").replace(
                    "total", f"total_{loss_type}"
                )
                if "total_mse" not in k
                else k
            ): v
            for k, v in epoch_logs.items()
        }
        super()._log_epoch(epoch, pinc_logs, info_dict, val_plots)

    def train_epoch(self, epoch):
        loss_logs = defaultdict(list)
        info_dict = defaultdict(list)
        t_start_data = perf_counter_ns()

        step_fn = {
            "autoencoder": train_step_autoencoder,
            "peft": train_step_peft,
            "simsiam": train_step_simsiam,
        }[self.cfg.stage]

        for sample in self.pbar:
            try:
                reset_peak_memory_stats(self.device)
            except Exception:
                pass

            xs = {
                k: getattr(sample, k).to(self.device, non_blocking=True)
                for k in self.input_fields
                if getattr(sample, k) is not None
            }
            condition = (
                sample.conditioning.to(self.device)
                if sample.conditioning is not None
                else None
            )
            idx_data = {k: getattr(sample, k).to(self.device) for k in self.idx_keys}
            geometry = tree_map(lambda g: g.to(self.device), sample.geometry)

            if self.augmentations:
                for aug_fn in self.augmentations:
                    xs = {k: aug_fn(v, idx_data["file_index"]) for k, v in xs.items()}
                    if self.cfg.dataset.augment.mask_modes.active:
                        # separate input and target
                        xs["df"], xs["df_tgt"], xs["mask"], mask_strategy = xs["df"]

            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)
            t_start_fwd = perf_counter_ns()

            with torch.autocast(
                str(self.device), dtype=self.amp_dtype, enabled=self.use_amp
            ):
                # dispatch to correct step function
                if self.cfg.stage == "autoencoder":
                    if self.cfg.dataset.augment.mask_modes.active:
                        step_fn = partial(
                            train_step_autoencoder,
                            denormalize_fn=self.trainset.denormalize,
                        )
                    else:
                        step_fn = train_step_autoencoder
                if self.cfg.stage == "peft":
                    step_fn = train_step_peft
                if self.cfg.stage == "simsiam":
                    xs["df_aug"] = getattr(sample, "df_aug").to(self.device)
                loss, losses = step_fn(
                    self.cfg,
                    self.model,
                    xs,
                    condition,
                    idx_data,
                    geometry,
                    self.loss_wrap,
                    remainig_progress(self.cur_update_step, self.total_steps),
                )

            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

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
            loss_logs["total"].append(loss.item())
            for k, v in losses.items():
                loss_logs[k].append(v.item())

            # if self.cur_update_step % 100 == 0:
            #     del xs, condition, idx_data, geometry, loss, losses
            #     memory_cleanup(self.device, aggressive=True)

            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(self.device) / 1024**2)
            t_start_data = perf_counter_ns()

        return {k: sum(v) / max(len(v), 1) for k, v in loss_logs.items()}, info_dict

    def evaluate(self, epoch):
        return self.evaluator(
            rank=self.rank,
            world_size=self.world_size,
            model=self.model,
            opt=self.opt,
            scheduler=self.scheduler,
            epoch=epoch,
            device=self.device,
            loss_val_min=self.loss_val_min,
            trainloader=self.trainloader,
            evaluate_recon=True,  # TODO adapt
            evaluate_probing=False,
        )
