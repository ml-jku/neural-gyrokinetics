from typing import Dict

import os
from collections import defaultdict
from time import perf_counter_ns

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.utils._pytree import tree_map
from diffusers import DDPMScheduler
from torch.distributions import StudentT, Laplace
from torch.utils._pytree import tree_map
from torchdiffeq import odeint

from neugk.diffusion.models import get_diffusion_model, DummyAE
from neugk.diffusion.evaluate import evaluate as diff_evaluate
from neugk.dataset import CycloneAESample
from neugk.utils import exclude_from_weight_decay, memory_cleanup
from neugk.runner import BaseRunner
from neugk.pinc.autoencoders.ae_utils import load_autoencoder


class DDPMRunner(BaseRunner):
    def setup_components(self):
        ckpt_path = self.cfg.ae_checkpoint
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise ValueError(f"AE not found at {ckpt_path}.")

        if "latent" in self.cfg.model.model_type:
            self.autoencoder, _, _ = load_autoencoder(ckpt_path, device=self.device)
            # TODO(gg) clean up latent dataset creation
            self.trainset.precompute_latents(
                self.rank,
                dataloader=self.trainloader,
                autoencoder=self.autoencoder,
                device=self.device,
            )

            for i, valloader in enumerate(self.valloaders):
                self.valsets[i].precompute_latents(
                    self.rank,
                    dataloader=valloader,
                    autoencoder=self.autoencoder,
                    device=self.device,
                    latent_stats=self.trainset.latent_stats,
                )
        else:
            # dummy autoencoder for pixel diffusion
            self.autoencoder = DummyAE()

        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        self.model = get_diffusion_model(self.cfg, self.autoencoder).to(self.device)

        self.latents_buffer = {}

        diff_cfg = self.cfg.model.scheduler

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diff_cfg.num_train_timesteps,
            beta_start=diff_cfg.beta_start,
            beta_end=diff_cfg.beta_end,
            beta_schedule=diff_cfg.beta_schedule,
            prediction_type=getattr(diff_cfg, "prediction_type", "epsilon"),
        )

        # TODO(diff) checkpointing
        # self._load_checkpoints()

        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            print(
                "Warning: No trainable params found in diffusion model (placeholder active?)"
            )
        else:
            exclude = getattr(self.cfg.training, "exclude_from_wd", [])
            if exclude:
                groups = exclude_from_weight_decay(
                    self.model, exclude, self.cfg.training.weight_decay
                )
                self.opt = torch.optim.AdamW(groups, lr=self.cfg.training.learning_rate)
            else:
                self.opt = torch.optim.AdamW(
                    params,
                    lr=self.cfg.training.learning_rate,
                    weight_decay=self.cfg.training.weight_decay,
                )

        self.input_fields = set(self.cfg.dataset.input_fields)
        self.idx_keys = ["file_index", "timestep_index"]

    def _load_checkpoints(self):
        # Load the diffusion model checkpoint (not the VAE one)
        ckpt_path = os.path.join(self.cfg.output_path, "ckp.pth")
        self.diff_ckpt_dict = {}

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading Diffusion Checkpoint: {ckpt_path}")
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            # TODO(diff): Ensure state dict keys match your diffusion model structure
            self.model.load_state_dict(loaded_ckpt["model_state_dict"])
            self.diff_ckpt_dict = loaded_ckpt

            self.start_epoch = self.diff_ckpt_dict.get("epoch", 0)
            self.loss_val_min = self.diff_ckpt_dict.get("loss", float("inf"))
            print(f"Resumed Diffusion from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
            self.loss_val_min = float("inf")

        self.cur_update_step = self.start_epoch * len(self.trainloader)

    def forward_step_diffusion(self, sample: Dict[str, torch.Tensor], condition):
        latents = sample["df"]
        bs = latents.shape[0]
        noise = torch.randn_like(latents, device=self.device)
        n_timesteps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, n_timesteps, (bs,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_output = self.model(noisy_latents, tstep=timesteps, condition=condition)
        # target selection
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.flatten(1).mean(1)
        # min snr loss weighting
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_prod_t = alphas_cumprod[timesteps]
        snr = alpha_prod_t / (1 - alpha_prod_t)
        snr_gamma = 5.0
        if pred_type == "v_prediction":
            weights = torch.clamp(snr, max=snr_gamma) / (snr + 1)
        elif pred_type == "epsilon":
            weights = torch.clamp(snr, max=snr_gamma) / snr

        return (loss * weights).mean()

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
            condition = sample.conditioning.to(self.device)
            idx_data = {k: getattr(sample, k).to(self.device) for k in self.idx_keys}
            geometry = tree_map(lambda g: g.to(self.device), sample.geometry)

            if self.augmentations:
                for aug_fn in self.augmentations:
                    xs = {k: aug_fn(v) for k, v in xs.items()}

            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)
            t_start_fwd = perf_counter_ns()

            with torch.autocast(str(self.device), self.amp_dtype, enabled=self.use_amp):
                loss = self.forward_step_diffusion(xs, condition)

            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

            self.opt.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_grad)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_grad)
                self.opt.step()

            if self.scheduler:
                self.scheduler.step()

            self.cur_update_step += 1.0

            loss_logs["loss"].append(loss.item())

            if (self.cur_update_step % 100) == 0:
                del xs, condition, idx_data, geometry, loss
                memory_cleanup(self.device, aggressive=True)

            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(self.device) / 1024**2)
            t_start_data = perf_counter_ns()

        loss_logs = {k: sum(v) / max(len(v), 1) for k, v in loss_logs.items()}
        return loss_logs, info_dict

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        dummy: torch.Tensor,
    ):
        self.model.eval()
        bs = condition.shape[0]
        latents = torch.randn((bs, *self.model.latent_shape), device=self.device)
        n_train_steps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(n_train_steps)

        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((bs,), t, device=self.device, dtype=torch.long)
            pred = self.model(latents, tstep=t_batch, condition=condition)
            step_output = self.noise_scheduler.step(pred, t, latents)
            latents = step_output.prev_sample

        # TODO(diff) temporary
        dummy = torch.zeros((bs, *dummy.shape[1:])).to(self.device)
        _, ae_cond, pad_axes = self.autoencoder.encode(dummy, condition=condition)
        decoded = self.autoencoder.decode(latents, pad_axes, condition=ae_cond)
        self.model.train()
        return decoded

    def evaluate(self, epoch):
        log_metric_dict, val_plots, self.loss_val_min = diff_evaluate(
            rank=self.rank,
            world_size=self.world_size,
            sample_fn=self.sample,
            valsets=self.valsets,
            valloaders=self.valloaders,
            opt=self.opt,
            lr_scheduler=self.scheduler,
            epoch=epoch,
            cfg=self.cfg,
            device=self.device,
            loss_val_min=self.loss_val_min,
        )
        return log_metric_dict, val_plots


class StudentTRunner(DDPMRunner):
    """https://arxiv.org/abs/2410.14171"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nu = getattr(self.cfg.model, "nu", 5.0)
        assert self.nu > 2, "not sure why? I think variance is undefined"
        self.distr = StudentT(torch.tensor(self.nu, device=self.device))

    def forward_step_diffusion(self, sample: Dict[str, torch.Tensor], condition):
        assert self.noise_scheduler.config.prediction_type == "epsilon"
        df = sample["df"]
        with torch.no_grad():
            latents, _, _ = self.autoencoder.encode(df, condition=condition)
        bs = latents.shape[0]
        # sampled from student's t distribution
        noise = self.distr.sample(latents.shape)
        # rest stays the same
        n_timesteps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, n_timesteps, (bs,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_output = self.model(noisy_latents, tstep=timesteps, condition=condition)

        # eq 9
        loss = F.mse_loss(model_output, noise, reduction="none")
        loss = loss.flatten(1).mean(1)

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_prod_t = alphas_cumprod[timesteps]
        snr = alpha_prod_t / (1 - alpha_prod_t)
        # NOTE: change, effective variance
        var_factor = self.nu / (self.nu - 2)
        snr = snr / var_factor

        snr_gamma = 5.0
        weights = torch.clamp(snr, max=snr_gamma) / snr
        return (loss * weights).mean()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        dummy: torch.Tensor,
        num_inference_steps: int = 100,
    ):
        self.model.eval()
        bs = condition.shape[0]
        # sampled from student's t distribution
        latents = self.distr.sample((bs, *self.model.latent_shape))
        # rest stays the same
        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((bs,), t, device=self.device, dtype=torch.long)
            pred = self.model(latents, tstep=t_batch, condition=condition)
            step_output = self.noise_scheduler.step(pred, t, latents)
            latents = step_output.prev_sample

        dummy = torch.zeros((bs, *dummy.shape[1:])).to(self.device)
        _, ae_cond, pad_axes = self.autoencoder.encode(dummy, condition=condition)
        decoded = self.autoencoder.decode(latents, pad_axes, condition=ae_cond)
        self.model.train()
        return decoded


class LaplaceFlowRunner(DDPMRunner):
    """Rectified Flow / Optimal Transport with Laplace Noise"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nu = getattr(self.cfg.model, "nu", 5.0)
        self.laplace_scale = getattr(self.cfg.model, "laplace_scale", 0.1)
        self.mix_ratio = getattr(self.cfg.model, "mix_ratio", 0.5)

        self.distr_student = StudentT(torch.tensor(self.nu, device=self.device))
        self.distr_laplace = Laplace(
            torch.tensor(0.0, device=self.device),
            torch.tensor(self.laplace_scale, device=self.device),
        )

    def _sample_mixture(self, shape):
        samples_laplace = self.distr_laplace.sample(shape)
        samples_student = self.distr_student.sample(shape)
        mask = torch.bernoulli(torch.full(shape, self.mix_ratio, device=self.device))
        return torch.where(mask.bool(), samples_laplace, samples_student)

    def forward_step_diffusion(self, sample: Dict[str, torch.Tensor], condition):
        df = sample["df"]
        with torch.no_grad():
            latents, _, _ = self.autoencoder.encode(df, condition=condition)
        bs = latents.shape[0]
        # x0 = self.distr.sample(latents.shape)
        x0, x1 = self._sample_mixture(latents.shape), latents
        n_train_steps = self.noise_scheduler.config.num_train_timesteps
        t_indices = torch.randint(0, n_train_steps, (bs,), device=self.device).long()
        t = (t_indices.float() / n_train_steps).view(-1, *[1] * (x1.ndim - 1))
        # linear scheduler
        xt = t * x1 + (1.0 - t) * x0
        target_v = x1 - x0
        pred = self.model(xt, tstep=t_indices, condition=condition)
        loss = F.mse_loss(pred, target_v)
        return loss

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, dummy: torch.Tensor):
        self.model.eval()
        bs = condition.shape[0]
        # latents = self.distr.sample((bs, *self.model.latent_shape))
        latents = self._sample_mixture((bs, *self.model.latent_shape))
        n_train_steps = self.noise_scheduler.config.num_train_timesteps

        class ODEFunc(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, t, x):
                t_idx = int(t.item() * n_train_steps)
                t_idx = min(max(t_idx, 0), n_train_steps - 1)
                t_batch = torch.full((bs,), t_idx, device=x.device, dtype=torch.long)
                return self.model(x, tstep=t_batch, condition=condition)

        t_span = torch.tensor([0.0, 1.0], device=self.device)
        func = ODEFunc(self.model)
        pred = odeint(func, latents, t_span, rtol=1e-5, atol=1e-5, method="rk4")[-1]

        # dummy = torch.zeros((bs, *dummy.shape[1:])).to(self.device)
        _, ae_cond, pad_axes = self.autoencoder.encode(dummy, condition=condition)
        decoded = self.autoencoder.decode(pred, pad_axes, condition=ae_cond)
        self.model.train()
        return decoded
