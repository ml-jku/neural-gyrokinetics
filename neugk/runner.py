"""Standard training and evaluation loop runners."""

import os
from abc import abstractmethod
from tqdm import tqdm
import atexit
import signal
import gc

import torch
import torch.distributed as dist
from neugk.utils import (
    edit_tag,
    ddp_setup,
    setup_logging,
    get_linear_burn_in_fn,
    get_cyclical_annealing_fn,
    remainig_progress,
    set_seed,
    get_scheduler,
    cleanup,
    handle_signal,
    memory_cleanup,
)
from neugk.dataset import get_data


class BaseRunner:
    """Base class for implementing workflow-specific training runners."""

    def __init__(self, rank, cfg, world_size):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.use_deepspeed = getattr(cfg, "deepspeed", {}).get("enable", False)
        set_seed(cfg.seed)

        # ddp setup
        if cfg.ddp.enable and cfg.ddp.n_nodes > 1 and world_size > 1:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        elif self.use_deepspeed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", rank))
        else:
            self.local_rank = rank

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        if self.use_deepspeed:
            assert (
                not cfg.ddp.enable
            ), "Cannot enable both DDP and DeepSpeed. Set ddp.enable=false."
            import deepspeed

            deepspeed.init_distributed()
            self.use_ddp = False
        elif cfg.ddp.enable and world_size > 1:
            ddp_setup(rank, world_size)
            self.use_ddp = True
        else:
            self.use_ddp = False

        # register what happens on SIGTERM (e.g. from slurm)
        atexit.register(cleanup)
        signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup(), exit(1)))

        self.writer = setup_logging(cfg) if not rank else None

        # common state
        self.start_epoch = 0
        self.loss_val_min = torch.inf
        self.cur_update_step = 0.0
        self.loss_scheduler_dict = {}
        self.scheduler = None

        # amp setup
        self.use_amp = self.cfg.amp.enable
        self.use_bf16 = (
            self.use_amp and self.cfg.amp.bfloat and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        if self.use_deepspeed:
            self.scaler = None
        else:
            self.scaler = torch.amp.GradScaler(
                device=self.device, enabled=self.use_amp and not self.use_bf16
            )

        self.setup_data()
        if not rank:
            with open("/proc/self/status") as _f:
                for _l in _f:
                    if "VmRSS" in _l:
                        print(
                            f"[init] RSS after setup_data: {int(_l.split()[1])/1024/1024:.1f} GB"
                        )
                        break
        self.setup_components()
        if not rank:
            with open("/proc/self/status") as _f:
                for _l in _f:
                    if "VmRSS" in _l:
                        print(
                            f"[init] RSS after setup_components: {int(_l.split()[1])/1024/1024:.1f} GB"
                        )
                        break
        self.setup_scheduler()

    def setup_data(self):
        """Initialize datasets and dataloaders."""
        datasets, dataloaders, self.augmentations = get_data(
            self.cfg, rank=self.local_rank
        )
        if len(datasets) == 3:
            self.trainset, self.valsets = datasets[0], datasets[1:]
            self.trainloader, self.valloaders = dataloaders[0], dataloaders[1:]
        else:
            self.trainset, self.valsets = datasets
            self.valsets = [self.valsets]
            self.trainloader, self.valloaders = dataloaders
            self.valloaders = [self.valloaders]

    def setup_common_losses(self, weights_cfg):
        """Configure loss weights and their respective schedulers."""
        weights = dict(weights_cfg.loss_weights) | dict(weights_cfg.extra_loss_weights)
        for key in weights.keys():
            if (
                hasattr(weights_cfg, "loss_scheduler")
                and weights_cfg.loss_scheduler is not None
                and key in weights_cfg.loss_scheduler
                and weights_cfg.loss_scheduler[key]
            ):
                sp = getattr(weights_cfg.loss_scheduler, key)
                sched_type = getattr(sp, "type", "linear")
                if sched_type == "cyclical":
                    self.loss_scheduler_dict[key] = get_cyclical_annealing_fn(
                        sp.start,
                        end=sp.end,
                        start_fraction=sp.start_fraction,
                        end_fraction=sp.end_fraction,
                        n_cycles=getattr(sp, "n_cycles", 4),
                        ratio=getattr(sp, "ratio", 0.5),
                    )
                else:
                    self.loss_scheduler_dict[key] = get_linear_burn_in_fn(
                        sp.start,
                        end=sp.end,
                        start_fraction=sp.start_fraction,
                        end_fraction=sp.end_fraction,
                    )
        if self.cfg.dataset.augment.mask_modes.active:
            weights["df_delta"] = self.cfg.dataset.augment.mask_modes.df_delta_weight
        if self.cfg.dataset.augment.vicreg_variance.active:
            weights["vicreg_variance"] = self.cfg.dataset.augment.vicreg_variance.weight
        if self.cfg.dataset.augment.logdet.active:
            weights["logdet"] = self.cfg.dataset.augment.logdet.weight
        return weights

    def setup_scheduler(self):
        """Initialize learning rate scheduler."""
        # LR/loss-schedule span covers only the remaining epochs, so a warm-start
        # finetune gets a full warmup+decay over its own epoch budget rather than
        # the absolute n_epochs (which would land mid-warmup for a late start).
        remaining_epochs = self.cfg.training.n_epochs - self.start_epoch
        self.total_steps = remaining_epochs * len(self.trainloader)
        if self.cfg.training.scheduler is not None:
            kwargs = {}
            # scheduler specific parameters
            if hasattr(self.cfg.training, "min_lr"):
                kwargs["min_lr"] = self.cfg.training.min_lr
            # if hasattr(self.cfg.training, "final_learning_rate"):
            # for OneCycle, final_div_factor = initial_lr / final_lr
            # but let's just pass it through
            # does not work for all transformers versions!!!
            # kwargs["final_div_factor"] = (
            #     self.cfg.training.learning_rate / self.cfg.training.final_learning_rate
            # )

            # warm up steps (not used by OneCycle but needed by others)
            is_long_run = self.cfg.training.n_epochs > 150
            if is_long_run:
                n_warmup = self.total_steps // 6
            else:
                n_warmup = max(self.total_steps // 10, 10 * len(self.trainloader))

            self.scheduler = get_scheduler(
                name=self.cfg.training.scheduler,
                optimizer=self.opt,
                num_warmup_steps=n_warmup,
                num_training_steps=self.total_steps,
                scheduler_specific_kwargs=kwargs,
            )

    def _build_deepspeed_config(self):
        """Translate Hydra config into a DeepSpeed JSON config dict."""
        ds = self.cfg.deepspeed
        ds_config = {
            "train_micro_batch_size_per_gpu": self.cfg.training.batch_size,
            "gradient_clipping": (
                self.cfg.training.clip_to if self.cfg.training.clip_grad else 0.0
            ),
            "zero_optimization": {
                "stage": ds.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if ds.offload_optimizer else "none",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu" if ds.offload_param else "none",
                    "pin_memory": True,
                },
                "allgather_bucket_size": int(float(ds.allgather_bucket_size)),
                "reduce_bucket_size": int(float(ds.reduce_bucket_size)),
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
        }
        if ds.offload_activations:
            ds_config["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "number_checkpoints": None,
                "contiguous_memory_optimization": False,
            }
        if self.use_bf16:
            ds_config["bf16"] = {"enabled": True}
        elif self.use_amp:
            ds_config["fp16"] = {"enabled": True, "initial_scale_power": 16}
        return ds_config

    def _log_epoch(self, epoch, epoch_logs, info_dict, val_plots):
        """Log training and validation statistics."""
        if self.writer and not self.rank:
            wandb_logs = epoch_logs | info_dict
            if not val_plots:
                self.writer.log(wandb_logs)
            else:
                self.writer.log(wandb_logs, commit=False)
                self.writer.log(val_plots)

        # console output
        if not self.rank:
            total_time = sum(
                v
                for k, v in info_dict.items()
                if "ms" in k and isinstance(v, (int, float))
            )
            epoch_str = str(epoch).zfill(len(str(int(self.cfg.training.n_epochs))))
            logged = ", ".join(
                [
                    f"{k}: {v:.5f}"
                    for k, v in epoch_logs.items()
                    if isinstance(v, (int, float))
                ]
            )
            print(f"Epoch: {epoch_str}, {logged}, step time: {total_time:.2f}ms")

    @abstractmethod
    def train_epoch(self, epoch):
        """Execute one training epoch."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, epoch):
        """Execute one evaluation pass."""
        raise NotImplementedError

    @abstractmethod
    def setup_components(self):
        """Initialize model, optimizer, and other workflow-specific components."""
        raise NotImplementedError

    def __call__(self, skip_eval: bool = False):
        """Main training loop execution.

        Returns:
            list[dict]: Per-epoch log dicts (train losses + val metrics).
                        Each dict also contains ``"val_plots"`` when
                        evaluation produced figures (e.g. ``avg_flux_UQ``).
        """
        use_tqdm = self.cfg.logging.tqdm if not self.use_ddp else False
        all_logs = []

        # main loop
        for epoch in range(self.start_epoch + 1, self.cfg.training.n_epochs + 1):
            if use_tqdm or (self.use_ddp and not self.rank):
                self.pbar = tqdm(self.trainloader, "Training")
            else:
                self.pbar = self.trainloader

            # training step
            self.model.train()
            if getattr(self, "loss_wrap", None) is not None:
                self.loss_wrap.train().to(self.device)
            loss_logs, info_dict = self.train_epoch(epoch)

            # logging
            progress = remainig_progress(self.cur_update_step, self.total_steps)
            train_logs = {
                "lr": (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler
                    else self.cfg.training.learning_rate
                ),
                **{
                    f"{k}_schedule": sched(progress)
                    for k, sched in self.loss_scheduler_dict.items()
                },
                **loss_logs,
            }
            train_losses_dict = edit_tag(train_logs, prefix="train")
            info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

            # evaluate
            memory_cleanup(self.device, aggressive=True)
            if not self.rank:
                with open("/proc/self/status") as _f:
                    for _line in _f:
                        if "VmRSS" in _line:
                            rss_gb = int(_line.split()[1]) / 1024 / 1024
                            break
                gpu_alloc = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(
                    f"[rank 0] epoch {epoch} post-cleanup: "
                    f"RSS={rss_gb:.1f} GB, "
                    f"GPU alloc={gpu_alloc:.1f} GB, "
                    f"GPU reserved={gpu_reserved:.1f} GB, "
                    f"Slurm headroom={115 - rss_gb:.1f} GB"
                )
            log_metric_dict, val_plots = {}, {}
            if not skip_eval:
                log_metric_dict, val_plots, self.loss_val_min = self.evaluate(epoch)

            # finalize logs
            epoch_logs = (
                {"epoch": epoch} | train_losses_dict | log_metric_dict | info_dict
            )
            if val_plots:
                epoch_logs["val_plots"] = val_plots
            all_logs.append(epoch_logs)
            self._log_epoch(epoch, epoch_logs, info_dict, val_plots)

        if self.writer:
            self.writer.finish()
        if self.use_ddp:
            dist.destroy_process_group()

        return all_logs
