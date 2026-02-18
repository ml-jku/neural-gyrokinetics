import os
from abc import abstractmethod
from tqdm import tqdm

import torch
import torch.distributed as dist
from transformers.optimization import get_scheduler

from neugk.utils import (
    ddp_setup,
    setup_logging,
    get_linear_burn_in_fn,
    remainig_progress,
)
from neugk.dataset import get_data


class BaseRunner:
    def __init__(self, rank, cfg, world_size):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size

        # ddp setup
        if cfg.ddp.enable and cfg.ddp.n_nodes > 1 and world_size > 1:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self.local_rank = rank

        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else "cpu"
        )

        if cfg.ddp.enable and world_size > 1:
            ddp_setup(rank, world_size)
            self.use_ddp = True
        else:
            self.use_ddp = False

        self.writer = setup_logging(cfg) if not rank else None

        # common state
        self.model = None
        self.opt = None
        self.scheduler = None
        self.loss_wrap = None
        self.grad_balancer = None
        self.scaler = None
        self.trainloader = None
        self.valsets = None
        self.valloaders = None
        self.start_epoch = 0
        self.loss_val_min = torch.inf
        self.cur_update_step = 0.0
        self.loss_scheduler_dict = {}

    def setup_data(self):
        datasets, dataloaders, self.augmentations = get_data(self.cfg, rank=self.rank)
        if len(datasets) == 3:
            self.trainset, self.valsets = datasets[0], datasets[1:]
            self.trainloader, self.valloaders = dataloaders[0], dataloaders[1:]
        else:
            self.trainset, self.valsets = datasets
            self.valsets = [self.valsets]
            self.trainloader, self.valloaders = dataloaders
            self.valloaders = [self.valloaders]

        self.total_steps = self.cfg.training.n_epochs * len(self.trainloader)

    def setup_common_losses(self, weights_cfg):
        weights = dict(weights_cfg.loss_weights) | dict(weights_cfg.extra_loss_weights)
        for key in weights.keys():
            if (
                hasattr(weights_cfg, "loss_scheduler")
                and weights_cfg.loss_scheduler is not None
                and key in weights_cfg.loss_scheduler
                and weights_cfg.loss_scheduler[key]
            ):
                sp = getattr(weights_cfg.loss_scheduler, key)
                self.loss_scheduler_dict[key] = get_linear_burn_in_fn(
                    sp.start,
                    end=sp.end,
                    start_fraction=sp.start_fraction,
                    end_fraction=sp.end_fraction,
                )
        return weights

    def setup_scheduler(self):
        if self.cfg.training.scheduler is not None:
            kwargs = {}
            if hasattr(self.cfg.training, "min_lr"):
                kwargs["scheduler_specific_kwargs"] = {
                    "min_lr": self.cfg.training.min_lr
                }

            self.scheduler = get_scheduler(
                name=self.cfg.training.scheduler,
                optimizer=self.opt,
                num_warmup_steps=(
                    self.total_steps // 6
                    if self.cfg.training.n_epochs > 150
                    else max(self.total_steps // 10, 10 * len(self.trainloader))
                ),
                num_training_steps=self.total_steps,
                **kwargs,
            )

    def _log_epoch(self, epoch, epoch_logs, info_dict, val_plots):
        if self.writer and not self.rank:
            wandb_logs = epoch_logs | info_dict
            if not val_plots:
                self.writer.log(wandb_logs)
            else:
                self.writer.log(wandb_logs, commit=False)
                self.writer.log(val_plots)

        if not self.rank:
            total_time = sum(
                v
                for k, v in info_dict.items()
                if "ms" in k and isinstance(v, (int, float))
            )
            epoch_str = str(epoch).zfill(len(str(int(self.cfg.training.n_epochs))))
            logged = ", ".join([f"{k}: {v:.5f}" for k, v in epoch_logs.items()])
            print(f"Epoch: {epoch_str}, {logged}, step time: {total_time:.2f}ms")

    @abstractmethod
    def train_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def setup_components(self):
        raise NotImplementedError

    def __call__(self):
        self.setup_data()
        self.setup_components()
        self.setup_scheduler()

        # amp setup
        self.use_amp = self.cfg.amp.enable
        self.use_bf16 = (
            self.use_amp and self.cfg.amp.bfloat and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.use_amp)

        use_tqdm = self.cfg.logging.tqdm if not self.use_ddp else False

        # main loop
        for epoch in range(self.start_epoch + 1, self.cfg.training.n_epochs + 1):
            if use_tqdm or (self.use_ddp and not self.rank):
                self.pbar = tqdm(self.trainloader, "Training")
            else:
                self.pbar = self.trainloader

            # training step
            self.model.train()
            if self.loss_wrap is not None:
                self.loss_wrap.train().to(self.device)
            loss_logs, info_dict = self.train_epoch(epoch)

            # logging
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

            train_losses_dict.update(loss_logs)
            info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

            # evaluate
            # log_metric_dict, val_plots = self.evaluate(epoch)
            log_metric_dict = {}
            val_plots = {}

            # log
            epoch_logs = train_losses_dict | log_metric_dict
            self._log_epoch(epoch, epoch_logs, info_dict, val_plots)

            # TODO need?
            # memory_cleanup(self.device, aggressive=True)

        if self.writer:
            self.writer.finish()
        if self.use_ddp:
            dist.destroy_process_group()
