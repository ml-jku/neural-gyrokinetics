"""Workflow execution for various diffusion models."""

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
from torch.distributions import Normal, StudentT, Laplace
import scipy.optimize

from neugk.diffusion.models import get_diffusion_model, DummyAE
from neugk.diffusion.eval import DiffusionEvaluator
from neugk.dataset import CycloneAESample
from neugk.utils import exclude_from_weight_decay
from neugk.runner import BaseRunner
from neugk.losses import LossWrapper
from neugk.pinc.autoencoders.ae_utils import load_autoencoder


class DDPMRunner(BaseRunner):
    """Workflow runner for Denoising Diffusion Probabilistic Models (DDPM)."""

    def setup_components(self):
        """Initialize autoencoder, diffusion model, noise scheduler, and optimizer."""
        is_latent = "latent" in self.cfg.model.model_type
        ckp_path = self.cfg.ae_checkpoint

        # load autoencoder
        if is_latent:
            if not ckp_path or not os.path.exists(ckp_path):
                raise ValueError(f"AE not found at {ckp_path} (latent diffusion).")
            self.autoencoder, _, _ = load_autoencoder(ckp_path, device=self.device)
            self.trainset.precompute_latents(
                self.rank,
                dataloader=self.trainloader,
                autoencoder=self.autoencoder,
                device=self.device,
            )
            # compute scale
            self.latent_scale = 1.0 / (self.trainset.latent_stats.var**0.5).item()
        else:
            # pixel-space
            self.autoencoder = DummyAE()
            self.latent_scale = 1.0

        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        # setup model
        self.model = get_diffusion_model(self.cfg, self.autoencoder, self.trainset)
        self.model.to(self.device)
        self.latents_buffer = {}

        # setup noise schedule
        diff_cfg = self.cfg.model.get("diffusion", {}).get("scheduler", {})
        if not diff_cfg:
            # fallback for older or pixel-space configs that might have scheduler at top level
            diff_cfg = self.cfg.model.get("scheduler", {})

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diff_cfg.get("num_train_timesteps", 1000),
            beta_start=diff_cfg.get("beta_start", 0.0001),
            beta_end=diff_cfg.get("beta_end", 0.02),
            beta_schedule=diff_cfg.get("beta_schedule", "linear"),
            prediction_type=diff_cfg.get("prediction_type", "epsilon"),
        )

        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])

        # setup optimizer
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

        # setup loss wrapper and evaluator
        self.loss_wrap = LossWrapper(
            denormalize_fn=self.valsets[0].denormalize,
            separate_zf=self.cfg.dataset.separate_zf,
            real_potens=self.cfg.dataset.real_potens,
        )

        self.evaluator = DiffusionEvaluator(
            cfg=self.cfg,
            valsets=self.valsets,
            valloaders=self.valloaders,
            loss_wrap=self.loss_wrap,
        )

    def _load_checkpoints(self):
        """Restore diffusion model state from the latest checkpoint."""
        ckpt_path = os.path.join(self.cfg.output_path, "ckp.pth")
        self.diff_ckpt_dict = {}

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading Diffusion Checkpoint: {ckpt_path}")
            loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

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
        """Execute one diffusion forward pass and compute weighted MSE loss."""
        latents = sample["df"] * self.latent_scale
        bs = latents.shape[0]
        # sample noise
        noise = torch.randn_like(latents, device=self.device)
        n_timesteps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, n_timesteps, (bs,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # model inference
        model_output = self.model(noisy_latents, tstep=timesteps, condition=condition)
        # compute loss
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.flatten(1).mean(1)
        # weighting
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_prod_t = alphas_cumprod[timesteps]
        snr = alpha_prod_t / (1 - alpha_prod_t)
        snr_gamma = 5.0
        if pred_type == "v_prediction":
            weights = torch.clamp(snr, max=snr_gamma) / (snr + 1)
        elif pred_type == "epsilon":
            weights = torch.clamp(snr, max=snr_gamma) / (snr + 1e-8)

        return (loss * weights).mean()

    def train_epoch(self, epoch):
        """Perform one training epoch over the dataset."""
        loss_logs = defaultdict(list)
        info_dict = defaultdict(list)
        t_start_data = perf_counter_ns()

        for sample in self.pbar:
            try:
                reset_peak_memory_stats(self.device)
            except Exception:
                pass

            sample: CycloneAESample
            # prepare data
            xs = {
                k: getattr(sample, k).to(self.device, non_blocking=True)
                for k in self.input_fields
                if getattr(sample, k) is not None
            }
            condition = sample.conditioning.to(self.device)
            idx_data = {
                k: getattr(sample, k).to(device=self.device) for k in self.idx_keys
            }
            geometry = tree_map(lambda g: g.to(self.device), sample.geometry)

            # apply augmentations
            if self.augmentations:
                for aug_fn in self.augmentations:
                    xs = {k: aug_fn(v) for k, v in xs.items()}

            info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)
            t_start_fwd = perf_counter_ns()

            # forward pass
            with torch.autocast(str(self.device), self.amp_dtype, enabled=self.use_amp):
                loss = self.forward_step_diffusion(xs, condition)

            info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
            t_start_bkd = perf_counter_ns()

            # update weights
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

            # # memory management
            # if (self.cur_update_step % 100) == 0:
            #     del xs, condition, idx_data, geometry, loss
            #     memory_cleanup(self.device, aggressive=True)

            info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
            info_dict["memory_mb"].append(max_memory_allocated(self.device) / 1024**2)
            t_start_data = perf_counter_ns()

        # average logs
        loss_logs = {k: sum(v) / max(len(v), 1) for k, v in loss_logs.items()}
        return loss_logs, info_dict

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
    ):
        """Generate samples from noise via iterative denoising."""
        self.model.eval()
        bs = condition.shape[0]
        # start with noise
        latents = torch.randn((bs, *self.model.latent_shape), device=self.device)
        n_train_steps = self.noise_scheduler.config.num_train_timesteps
        self.noise_scheduler.set_timesteps(n_train_steps)

        # denoise loop
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((bs,), t, device=self.device, dtype=torch.long)
            pred = self.model(latents, tstep=t_batch, condition=condition)
            step_output = self.noise_scheduler.step(pred, t, latents)
            latents = step_output.prev_sample

        # decode result
        latents = latents / self.latent_scale
        decoded = self.autoencoder.decode(latents, condition=condition)
        self.model.train()
        return decoded

    def evaluate(self, epoch):
        """Execute evaluation pipeline and log results."""
        return self.evaluator(
            rank=self.rank,
            world_size=self.world_size,
            model=self.model,
            opt=self.opt,
            scheduler=self.scheduler,
            epoch=epoch,
            device=self.device,
            loss_val_min=self.loss_val_min,
            sample_fn=self.sample,
        )


class StudentTRunner(DDPMRunner):
    """Diffusion runner using Student's T distribution for noise sampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nu = getattr(self.cfg.model, "nu", 5.0)
        assert self.nu > 2, "not sure why? I think variance is undefined"
        self.distr = StudentT(torch.tensor(self.nu, device=self.device))

    def forward_step_diffusion(self, sample: Dict[str, torch.Tensor], condition):
        """Student's T noise forward pass implementation."""
        assert self.noise_scheduler.config.prediction_type == "epsilon"
        latents = sample["df"] * self.latent_scale
        bs = latents.shape[0]
        # sampled from student's t distribution
        noise = self.distr.sample(latents.shape).to(self.device)
        # add noise
        n_timesteps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, n_timesteps, (bs,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model_output = self.model(noisy_latents, tstep=timesteps, condition=condition)
        # compute loss
        loss = F.mse_loss(model_output, noise, reduction="none")
        loss = loss.flatten(1).mean(1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_prod_t = alphas_cumprod[timesteps]
        snr = alpha_prod_t / (1 - alpha_prod_t)
        # effective variance factor
        var_factor = self.nu / (self.nu - 2)
        snr = snr / var_factor
        snr_gamma = 5.0
        weights = torch.clamp(snr, max=snr_gamma) / (snr + 1e-8)
        return (loss * weights).mean()

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_inference_steps: int = 100,
    ):
        """Generate samples using iterative denoising with heavy-tailed priors."""
        self.model.eval()
        bs = condition.shape[0]

        latents = self.distr.sample((bs, *self.model.latent_shape)).to(self.device)

        # denoising loop
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((bs,), t, device=self.device, dtype=torch.long)
            pred = self.model(latents, tstep=t_batch, condition=condition)
            step_output = self.noise_scheduler.step(pred, t, latents)
            latents = step_output.prev_sample

        # finalize output
        latents = latents / getattr(self, "latent_scale", 1.0)
        decoded = self.autoencoder.decode(latents, condition=condition)
        self.model.train()
        return decoded


class EDMRunner(DDPMRunner):
    """Elucidated Diffusion Models runner with preconditioned training and sampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # scaling factor
        self.sigma_data = getattr(self.cfg.model.diffusion, "sigma_data", 1.0)
        # training params
        self.p_mean = getattr(self.cfg.model.diffusion, "p_mean", -1.2)
        self.p_std = getattr(self.cfg.model.diffusion, "p_std", 1.2)
        # sampling params
        self.sigma_min = getattr(self.cfg.model.diffusion, "sigma_min", 0.002)
        self.sigma_max = getattr(self.cfg.model.diffusion, "sigma_max", 80.0)
        self.rho = getattr(self.cfg.model.diffusion, "rho", 7.0)

        del self.noise_scheduler

    def _preconditioned_forward(
        self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor
    ):
        """Apply EDM preconditioning factors to input and model output."""
        expand_dims = [-1] + [1] * (x.ndim - 1)

        # compute factors
        c_skip = (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.log()

        c_skip = c_skip.view(*expand_dims)
        c_out = c_out.view(*expand_dims)
        c_in = c_in.view(*expand_dims)

        # run model
        F_theta = self.model(x * c_in, tstep=c_noise, condition=condition)
        return c_skip * x + c_out * F_theta

    def forward_step_diffusion(self, sample: dict, condition: torch.Tensor):
        """Execute preconditioned training step with weighted MSE."""
        latents = sample["df"] * self.latent_scale
        bs = latents.shape[0]
        # compute sigma
        noise = torch.randn((bs,), device=self.device)
        sigma = (noise * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(latents) * sigma.view(-1, *[1] * (latents.ndim - 1))
        noisy_latents = latents + noise
        D_theta = self._preconditioned_forward(noisy_latents, sigma, condition)
        # weighted loss
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        weight = weight.view(-1, *[1] * (latents.ndim - 1))
        loss = F.mse_loss(D_theta, latents, reduction="none")
        loss = (loss * weight).flatten(1).mean(1)
        return loss.mean()

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, steps: int = 5, latent_only: bool = False
    ):
        """Generate samples using Heun's 2nd order deterministic ODE solver."""
        self.model.eval()
        bs = condition.shape[0]
        # compute schedule
        step_indices = torch.arange(steps, dtype=torch.float32, device=self.device)
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (
            sigma_max_rho + step_indices / (steps - 1) * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
        # start with noise
        x = (
            torch.randn((bs, *self.model.latent_shape), device=self.device)
            * self.sigma_max
        )
        # iterate solver
        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_batch = torch.full((bs,), sigma_hat, device=self.device)
            # euler step
            denoised = self._preconditioned_forward(x, sigma_batch, condition)
            d_i = (x - denoised) / sigma_hat
            x_next = x + d_i * (sigma_next - sigma_hat)
            # heun correction
            if sigma_next != 0:
                sigma_next_batch = torch.full((bs,), sigma_next, device=self.device)
                denoised_next = self._preconditioned_forward(
                    x_next, sigma_next_batch, condition
                )
                d_prime = (x_next - denoised_next) / sigma_next
                x_next = x + 0.5 * (d_i + d_prime) * (sigma_next - sigma_hat)
            x = x_next
        # decode
        pred = x / getattr(self, "latent_scale", 1.0)
        if not latent_only:
            pred = self.autoencoder.decode(pred, condition=condition)
        self.model.train()
        return pred


class FlowMatchingRunner(DDPMRunner):
    """Runner for Latent Rectified Flow Matching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            "latent" in self.cfg.model.model_type
        ), "Flow matching only implemented for latent diffusion."

        # setup distributions
        if self.cfg.model.diffusion.noise_distribution == "gaussian":
            self.distr_gauss = Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )

        if self.cfg.model.diffusion.noise_distribution == "mixture":
            self.nu = getattr(self.cfg.model, "nu", 5.0)
            self.laplace_scale = getattr(self.cfg.model, "laplace_scale", 0.1)
            self.mix_ratio = getattr(self.cfg.model, "mix_ratio", 0.5)

            self.distr_student = StudentT(torch.tensor(self.nu, device=self.device))
            self.distr_laplace = Laplace(
                torch.tensor(0.0, device=self.device),
                torch.tensor(self.laplace_scale, device=self.device),
            )

    def _get_prior(self, shape):
        """Sample from the configured prior noise distribution."""
        if self.cfg.model.diffusion.noise_distribution == "gaussian":
            return self.distr_gauss.sample(shape)

        if self.cfg.model.diffusion.noise_distribution == "mixture":
            samples_laplace = self.distr_laplace.sample(shape)
            samples_student = self.distr_student.sample(shape)
            mask = torch.bernoulli(
                torch.full(shape, self.mix_ratio, device=self.device)
            )
            return torch.where(mask.bool(), samples_laplace, samples_student)

    def forward_step_diffusion(self, sample: dict, condition: torch.Tensor):
        """Execute one flow matching training step with optional optimal transport."""
        latents = sample["df"] * self.latent_scale
        bs = latents.shape[0]
        x0 = self._get_prior(latents.shape).to(self.device)
        x1 = latents
        # optimal transport
        if getattr(self.cfg.model.diffusion, "minibatch_ot", True):
            with torch.no_grad():
                x0_flat = x0.view(bs, -1)
                x1_flat = x1.view(bs, -1)
                cost_matrix = torch.cdist(x0_flat, x1_flat).cpu().numpy()
                row_ind, _ = scipy.optimize.linear_sum_assignment(cost_matrix)
                x0 = x0[torch.tensor(row_ind, device=self.device)]

        # sample time
        if getattr(self.cfg.model.diffusion, "continuous_time", True):
            tstep = torch.sigmoid(torch.randn((bs,), device=self.device))
            t = tstep.view(-1, *[1] * (x1.ndim - 1))
        else:
            n_train_steps = self.noise_scheduler.config.num_train_timesteps
            tstep = torch.randint(0, n_train_steps, (bs,), device=self.device).long()
            t = (tstep.float() / (n_train_steps - 1)).view(-1, *[1] * (x1.ndim - 1))

        # compute velocity
        xt = t * x1 + (1.0 - t) * x0
        target_v = x1 - x0
        pred = self.model(xt, tstep=tstep, condition=condition)
        return F.mse_loss(pred, target_v)

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, steps: int = 50, latent_only: bool = False
    ):
        """Generate samples by integrating the velocity field using Euler's method."""
        self.model.eval()
        bs = condition.shape[0]
        x = self._get_prior((bs, *self.model.latent_shape)).to(self.device)
        t_steps = torch.linspace(0.0, 1.0, steps + 1, device=self.device)

        # integrate ODE
        for i in range(steps):
            t_curr = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t_curr

            if getattr(self.cfg.model, "continuous_time", True):
                t_batch = torch.full((bs,), t_curr.item(), device=self.device)
            else:
                n_train_steps = self.noise_scheduler.config.num_train_timesteps
                t_batch = torch.full(
                    (bs,),
                    int(t_curr.item() * (n_train_steps - 1)),
                    device=self.device,
                    dtype=torch.long,
                )

            v_pred = self.model(x, tstep=t_batch, condition=condition)
            x = x + v_pred * dt

        # decode
        pred = x / getattr(self, "latent_scale", 1.0)
        if not latent_only:
            pred = self.autoencoder.decode(pred, condition=condition)
        self.model.train()
        return pred


class JiTRunner(DDPMRunner):
    """Runner for Just Image Transformers (https://arxiv.org/abs/2511.13720, JiT)."""

    def setup_components(self):
        """Initialize components and setup importance sampling history."""
        super().setup_components()
        self.use_importance_sampling = getattr(
            self.cfg.model.diffusion, "importance_sampling", True
        )
        if self.use_importance_sampling:
            self.n_timesteps = self.noise_scheduler.config.num_train_timesteps
            self.loss_history = torch.ones(self.n_timesteps, device=self.device)

    def _get_timesteps(self, bs: int):
        n_train_steps = self.noise_scheduler.config.num_train_timesteps
        if self.use_importance_sampling:
            return torch.randint(0, n_train_steps, (bs,), device=self.device).long()

        # importance sampling weights based on historical loss
        weights = self.loss_history / self.loss_history.sum()
        return torch.multinomial(weights, bs, replacement=True)

    def forward_step_diffusion(self, sample: dict, condition: torch.Tensor):
        x0 = sample["df"] * self.latent_scale
        bs = x0.shape[0]

        # sample noise and timesteps
        noise = torch.randn_like(x0)
        tstep = self._get_timesteps(bs)

        # add noise to get xt
        xt = self.noise_scheduler.add_noise(x0, noise, tstep)

        # model predicts x0 directly
        pred_x0 = self.model(xt, tstep=tstep, condition=condition)

        # compute loss on x0 prediction
        loss = F.mse_loss(pred_x0, x0, reduction="none")
        loss = loss.flatten(1).mean(1)

        # update loss history for importance sampling
        if getattr(self, "use_importance_sampling", False):
            with torch.no_grad():
                # simple moving average update
                for i in range(bs):
                    t = tstep[i].item()
                    self.loss_history[t] = (
                        0.99 * self.loss_history[t] + 0.01 * loss[i].detach()
                    )

        return loss.mean()

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, steps: int = 1, latent_only: bool = False
    ):
        self.model.eval()
        bs = condition.shape[0]
        n_train_steps = self.noise_scheduler.config.num_train_timesteps

        # start with pure noise
        xt = torch.randn((bs, *self.model.latent_shape), device=self.device)

        # few-step iterative refinement
        t_indices = torch.linspace(n_train_steps - 1, 0, steps)
        for i in range(steps):
            t_val = int(t_indices[i])
            t_batch = torch.full((bs,), t_val, device=self.device, dtype=torch.long)

            # predict x0
            x0_pred = self.model(xt, tstep=t_batch, condition=condition)

            if i < steps - 1:
                # move to next timestep (re-noise)
                t_next = int(t_indices[i + 1])
                noise = torch.randn_like(x0_pred)
                xt = self.noise_scheduler.add_noise(
                    x0_pred, noise, torch.tensor([t_next], device=self.device)
                )

        # decode
        pred = x0_pred / getattr(self, "latent_scale", 1.0)
        if not latent_only:
            pred = self.autoencoder.decode(pred, condition=condition)
        self.model.train()
        return pred
