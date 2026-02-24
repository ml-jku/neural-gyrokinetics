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
from torch.utils._pytree import tree_map
from torchdiffeq import odeint
import scipy.optimize

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
            self.trainset.precompute_latents(
                self.rank,
                dataloader=self.trainloader,
                autoencoder=self.autoencoder,
                device=self.device,
            )

            # for i, valloader in enumerate(self.valloaders):
            #     self.valsets[i].precompute_latents(
            #         self.rank,
            #         dataloader=valloader,
            #         autoencoder=self.autoencoder,
            #         device=self.device,
            #         latent_stats=self.trainset.latent_stats,
            #     )
            # standard practice is 1.0 / std
            self.latent_scale = 1.0 / (self.trainset.latent_stats.var**0.5).item()
        else:
            # dummy autoencoder for pixel diffusion
            self.autoencoder = DummyAE()

        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

        self.model = get_diffusion_model(self.cfg, self.autoencoder).to(self.device)

        self.latents_buffer = {}

        diff_cfg = self.cfg.model.diffusion.scheduler
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
        latents = sample["df"] * self.latent_scale
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
            weights = torch.clamp(snr, max=snr_gamma) / (snr + 1e-8)

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
        _, pad_axes = self.autoencoder.encode(dummy, condition=condition)
        latents = latents / self.latent_scale
        decoded = self.autoencoder.decode(latents, pad_axes, condition=condition)
        self.model.train()
        return decoded

    def evaluate(self, epoch):
        log_metric_dict, val_plots, self.loss_val_min = diff_evaluate(
            rank=self.rank,
            world_size=self.world_size,
            model=self.model,
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
        latents = sample["df"] * self.latent_scale
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
        weights = torch.clamp(snr, max=snr_gamma) / (snr + 1e-8)
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

        latents = self.distr.sample((bs, *self.model.latent_shape))

        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((bs,), t, device=self.device, dtype=torch.long)
            pred = self.model(latents, tstep=t_batch, condition=condition)
            step_output = self.noise_scheduler.step(pred, t, latents)
            latents = step_output.prev_sample

        dummy = torch.zeros((bs, *dummy.shape[1:])).to(self.device)
        _, pad_axes = self.autoencoder.encode(dummy, condition=condition)

        latents = latents / getattr(self, "latent_scale", 1.0)
        decoded = self.autoencoder.decode(latents, pad_axes, condition=condition)
        self.model.train()
        return decoded


class EDMRunner(DDPMRunner):
    """Elucidated Diffusion Models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # assumes latents scaled to var=1.0
        self.sigma_data = getattr(self.cfg.model, "sigma_data", 1.0)
        # training distribution
        self.p_mean = getattr(self.cfg.model, "p_mean", -1.2)
        self.p_std = getattr(self.cfg.model, "p_std", 1.2)
        # sampling schedule
        self.sigma_min = getattr(self.cfg.model, "sigma_min", 0.002)
        self.sigma_max = getattr(self.cfg.model, "sigma_max", 80.0)
        self.rho = getattr(self.cfg.model, "rho", 7.0)

        del self.noise_scheduler

    def _preconditioned_forward(
        self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor
    ):
        expand_dims = [-1] + [1] * (x.ndim - 1)

        # preconditioning factors
        c_skip = (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.log()

        c_skip = c_skip.view(*expand_dims)
        c_out = c_out.view(*expand_dims)
        c_in = c_in.view(*expand_dims)

        # predict changes
        F_theta = self.model(x * c_in, tstep=c_noise, condition=condition)
        return c_skip * x + c_out * F_theta

    def forward_step_diffusion(self, sample: dict, condition: torch.Tensor):
        latents = sample["df"] * self.latent_scale
        bs = latents.shape[0]
        noise = torch.randn((bs,), device=self.device)
        sigma = (noise * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(latents) * sigma.view(-1, *[1] * (latents.ndim - 1))
        noisy_latents = latents + noise
        D_theta = self._preconditioned_forward(noisy_latents, sigma, condition)
        # loss weighting
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        weight = weight.view(-1, *[1] * (latents.ndim - 1))
        loss = F.mse_loss(D_theta, latents, reduction="none")
        loss = (loss * weight).flatten(1).mean(1)
        return loss.mean()

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, dummy: torch.Tensor):
        self.model.eval()
        bs = condition.shape[0]
        num_steps = self.noise_scheduler.config.num_train_timesteps
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (
            sigma_max_rho
            + step_indices / (num_steps - 1) * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
        sigmas = torch.cat(
            [sigmas, torch.zeros_like(sigmas[:1])]
        )  # Add exactly 0 at the end
        x = (
            torch.randn((bs, *self.model.latent_shape), device=self.device)
            * self.sigma_max
        )
        # heun 2nd order ode solver
        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_batch = torch.full((bs,), sigma_hat, device=self.device)
            # predict x0
            denoised = self._preconditioned_forward(x, sigma_batch, condition)
            # euler
            d_i = (x - denoised) / sigma_hat
            x_next = x + d_i * (sigma_next - sigma_hat)
            # heun step
            if sigma_next != 0:
                sigma_next_batch = torch.full((bs,), sigma_next, device=self.device)
                denoised_next = self._preconditioned_forward(
                    x_next, sigma_next_batch, condition
                )
                d_prime = (x_next - denoised_next) / sigma_next
                x_next = x + 0.5 * (d_i + d_prime) * (sigma_next - sigma_hat)
            x = x_next
        x = x / getattr(self, "latent_scale", 1.0)
        _, pad_axes = self.autoencoder.encode(dummy, condition=condition)
        decoded = self.autoencoder.decode(x, pad_axes, condition=condition)
        self.model.train()
        return decoded


class FlowMatchingRunner(DDPMRunner):
    """Latent rectified flow matching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            "latent" in self.cfg.model.model_type
        ), "Flow matching only implemented for latent diffusion."

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
        latents = sample["df"] * self.latent_scale
        bs = latents.shape[0]
        x0 = self._get_prior(latents.shape)
        x1 = latents
        # minibatch OT option
        if getattr(self.cfg.model.diffusion, "minibatch_ot", True):
            with torch.no_grad():
                x0_flat = x0.view(bs, -1)
                x1_flat = x1.view(bs, -1)
                cost_matrix = torch.cdist(x0_flat, x1_flat).cpu().numpy()
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
                x0 = x0[torch.tensor(row_ind, device=self.device)]
                x1 = x1[torch.tensor(col_ind, device=self.device)]

        if getattr(self.cfg.model.diffusion, "continuous_time", True):
            # logit-normal continuous time
            tstep = torch.sigmoid(torch.randn((bs,), device=self.device))
            t = tstep.view(-1, *[1] * (x1.ndim - 1))
        else:
            # discrete uniform time
            n_train_steps = self.noise_scheduler.config.num_train_timesteps
            tstep = torch.randint(0, n_train_steps, (bs,), device=self.device).long()
            t = (tstep.float() / (n_train_steps - 1)).view(-1, *[1] * (x1.ndim - 1))

        # straight path
        xt = t * x1 + (1.0 - t) * x0
        target_v = x1 - x0
        pred = self.model(xt, tstep=tstep, condition=condition)
        return F.mse_loss(pred, target_v)

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, steps: int = 50, latent_only: bool = False
    ):
        self.model.eval()
        bs = condition.shape[0]
        x = self._get_prior((bs, *self.model.latent_shape))
        t_steps = torch.linspace(0.0, 1.0, steps + 1, device=self.device)

        # simple euler solve (path are linear, no need for rk4)
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

        pred = x / getattr(self, "latent_scale", 1.0)
        if not latent_only:
            # TODO temporary
            ch = 2 + 2 * self.trainset.separate_zf
            dummy = torch.zeros((1, ch, *self.trainset.resolution), device=self.device)
            _, pad_axes = self.autoencoder.encode(dummy, condition=condition)
            pred = self.autoencoder.decode(pred, pad_axes, condition=condition)
        self.model.train()
        return pred
