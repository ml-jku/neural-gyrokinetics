"""
PINC-specific loss wrappers and gradient balancers.
Extends base_losses to add support for Spectral, VAE, and VQVAE losses,
plus EMA normalization and custom Conflict-Free Gradient Descent (ConFIG) patching.
"""

from typing import List, Callable, Dict, Optional, Any
import warnings

import torch
import torch.nn.functional as F

from neugk.utils import recombine_zf
from neugk.physics.integrals import FluxIntegral
from neugk import physics
from neugk.losses import LossWrapper, GradientBalancer


def _wide_min_norm_solution(
    units: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute minimal-norm solution for the underdetermined case using gram matrix inversion"""
    row_norms = units.norm(dim=1)
    keep = row_norms > 0
    if keep.sum() == 0:
        return torch.zeros(units.shape[1], device=units.device, dtype=units.dtype)
    U = units[keep]
    w = weights[keep]
    G = U @ U.t()
    reg = eps * G.diag().mean()
    G = G + reg * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    a = torch.linalg.solve(G, w)
    return a @ U


class PINCLossWrapper(LossWrapper):
    """PINCLossWrapper class."""

    def __init__(
        self,
        weights: Dict,
        schedulers: Dict,
        denormalize_fn: Optional[Callable] = None,
        separate_zf: bool = False,
        real_potens: bool = False,
        loss_type: str = "mse",
        integral_loss_type: str = "mse",
        spectral_loss_type: str = "l1",
        dataset_stats: Optional[Dict] = None,
        ds: Optional[float] = None,
        ema_normalization_loss: Optional[List[str]] = None,
        ema_beta: float = 0.99,
        eval_loss_type: str = "mse",
        eval_integral_loss_type: str = "mse",
        eval_spectral_loss_type: str = "l1",
        augmentations: Optional[List[str]] = None,
        dataset: Optional[Any] = None,
        integral_precision: str = "float64",
        free_bits: float = 0.0,
    ):
        augmentations = augmentations or []
        masked_mode_modeling = "mask_modes" in augmentations

        # Initialize base class
        super().__init__(
            weights=weights,
            schedulers=schedulers,
            denormalize_fn=denormalize_fn,
            separate_zf=separate_zf,
            real_potens=real_potens,
            masked_mode_modeling=masked_mode_modeling,
        )

        self.free_bits = free_bits
        self._augmentation_losses: List[str] = []
        self.augmentations = augmentations
        self._register_augmentation_losses()

        # Extended loss categories
        self._vae_losses = ["beta_vae"]
        self._vqvae_losses = ["vq_commit"]
        self._spectral_losses = [
            "kxspec",
            "kyspec",
            "qspec",
            "phi_zf",
            "kxspec_monotonicity",
            "kyspec_monotonicity",
            "qspec_monotonicity",
            "mass",
        ]
        self._simsiam_losses = ["simsiam"]

        self.integrator = FluxIntegral(
            real_potens=real_potens,
            flux_fields=False,
            spectral_df=False,
            integral_precision=integral_precision,
        )
        self.integrator_spec = FluxIntegral(
            real_potens=real_potens,
            flux_fields=True,
            spectral_df=True,
            integral_precision=integral_precision,
        )

        self.loss_type = loss_type
        self.integral_loss_type = integral_loss_type
        self.spectral_loss_type = spectral_loss_type
        self.eval_loss_type = eval_loss_type
        self.eval_integral_loss_type = eval_integral_loss_type
        self.eval_spectral_loss_type = eval_spectral_loss_type

        self.dataset_stats = dataset_stats or {}
        self.ds = ds
        self.dataset = dataset
        self.loss_normalizer = {}
        self.normalize_losses = getattr(loss_type, "normalize_losses", False)

        self.ema_normalization_loss = ema_normalization_loss or []
        self.ema_beta = ema_beta
        self._ema_loss_scales = {}
        self._ema_initialized = set()

        self.complex_metrics = None

    def _register_augmentation_losses(self):
        """Populate _augmentation_losses and default weights based on self.augmentations.

        Only *extra* loss keys (not data-level keys like df_delta that should flow
        through compute_data_loss) are registered here. Data-level keys are added
        to self._data_losses in the base class instead.
        """
        for name in self.augmentations:
            if name == "mask_modes":
                # df_delta is already added to _data_losses by the base class;
                # no extra loss keys needed here.
                name = "df_delta"  # for weight registration and logging
                if not "df_delta" in self.weights:
                    self.weights.setdefault("df_delta", 1.0)
            else:
                if not name in self.weights:
                    self.weights.setdefault(name, 1.0)
                self._augmentation_losses.append(name)

    @property
    def all_losses(self):
        return (
            super().all_losses
            + self._vae_losses
            + self._vqvae_losses
            + self._spectral_losses
            + self._simsiam_losses
            + self._augmentation_losses
        )

    def _update_ema_loss_scale(self, loss_name: str, loss_value: torch.Tensor):
        if loss_name not in self.ema_normalization_loss:
            return
        curr_scale = loss_value.detach().item()
        if loss_name not in self._ema_initialized:
            self._ema_loss_scales[loss_name] = curr_scale
            self._ema_initialized.add(loss_name)
        else:
            self._ema_loss_scales[loss_name] = (
                self.ema_beta * self._ema_loss_scales[loss_name]
                + (1 - self.ema_beta) * curr_scale
            )

    def _apply_ema_normalization(
        self, losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            name: (
                val / self._ema_loss_scales[name]
                if name in self.ema_normalization_loss
                and name in self._ema_loss_scales
                and self._ema_loss_scales[name] > 1e-8
                else val
            )
            for name, val in losses.items()
        }

    def get_ema_statistics(self) -> Dict[str, torch.Tensor]:
        return {
            f"ema_scale_{k}": torch.tensor(v, dtype=torch.float32)
            for k, v in self._ema_loss_scales.items()
        }

    def _get_current_loss_types(self):
        if self.training:
            return {
                "data": self.loss_type,
                "int": self.integral_loss_type,
                "spec": self.spectral_loss_type,
            }
        return {
            "data": self.eval_loss_type,
            "int": self.eval_integral_loss_type,
            "spec": self.eval_spectral_loss_type,
        }

    def compute_per_mode_losses(
        self,
        preds: Dict[str, torch.Tensor],
        tgts: Dict[str, torch.Tensor],
        loss_type: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return per-mode MSE for df_delta during training.

        Assumes the last dimension of preds['df_delta'] indexes the modes.
        Returns a dict like {'mode_loss/ky=0': ..., 'mode_loss/ky=1': ..., ...}.
        """
        if "df_delta" not in preds or "df_delta" not in tgts:
            return {}

        p = preds["df_delta"]
        t = tgts["df_delta"]
        n_modes = p.shape[-1]

        per_mode = {}
        for m in range(n_modes):
            if loss_type == "mse":
                per_mode[f"mode_loss/ky={m}"] = F.mse_loss(p[..., m], t[..., m])
            elif loss_type == "relative_mse":
                per_mode[f"mode_loss/ky={m}"] = torch.sum(
                    (p[..., m] - t[..., m]) ** 2
                ) / (torch.sum(t[..., m] ** 2) + 1e-8)
            else:
                raise NotImplementedError(
                    f"Unsupported per-mode loss type: {loss_type}"
                )

        return per_mode

    def compute_vicreg_variance(
        self, z: torch.Tensor, eps: float = 1e-8
    ) -> Dict[str, torch.Tensor]:
        # flatten batch and spatial dimensions, keep latent dim
        z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(z.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(1.0 - std))
        return {"vicreg_variance": var_loss}

    def compute_vicreg_covariance(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = z.reshape(-1, z.shape[-1])
        z_centered = z - z.mean(dim=0)
        B, D = z_centered.shape

        # --- Covariance ---
        cov = (z_centered.T @ z_centered) / (B - 1)  # (D, D)
        off_diag = cov - torch.diag(cov.diag())
        cov_loss = (off_diag**2).sum() / D
        return {"vicreg_covariance": cov_loss}

    def compute_logdet(
        self, z: torch.Tensor, eps: float = 1e-8
    ) -> Dict[str, torch.Tensor]:
        z = z.reshape(-1, z.shape[-1]).float()
        d = z.shape[-1]
        z_std = (z - z.mean(dim=0)) / (z.std(dim=0) + eps)
        cov = (z_std.T @ z_std) / max(z.shape[0] - 1, 1)
        logdet = -torch.logdet(cov + eps * torch.eye(d, device=z.device)) / d
        # cov = torch.cov(z.T) + eps * torch.eye(z.shape[-1], device=z.device)
        # logdet = torch.logdet(cov) / d
        return {"logdet": logdet}

    def compute_data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8,
        loss_type: Optional[str] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        loss_type = loss_type or self._get_current_loss_types()["data"]
        return physics.compute_data_loss(
            pred,
            target,
            loss_type,
            eps,
            reduction,
            complex_metrics=self.complex_metrics,
        )

    def compute_integral_loss(
        self, pred, target, loss_type="mse", eps=1e-8, loss_name="flux_int"
    ):
        return physics.compute_integral_loss(
            pred,
            target,
            loss_type,
            eps,
            loss_name,
            dataset_stats=self.dataset_stats,
            ema_state=self.__dict__.setdefault("_int_ema_state", {}),
        )

    def compute_spectral_loss(self, pred, target, loss_type="l1", eps=1e-8):
        return physics.compute_spectral_loss(pred, target, loss_type, eps)

    def compute_vae_loss(self, preds):
        if "mu" not in preds or "logvar" not in preds:
            return {}
        kl_elementwise = -0.5 * (
            1 + preds["logvar"] - preds["mu"].pow(2) - preds["logvar"].exp()
        )
        if self.free_bits > 0:
            # Free bits (Kingma et al., 2016): clamp per-dimension KL to a
            # minimum of free_bits so every latent channel stays active
            # Average over batch first, clamp per latent dim, then average
            kl_per_dim = kl_elementwise.mean(0)
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
            return {"beta_vae": kl_per_dim.mean()}
        return {"beta_vae": kl_elementwise.mean()}

    def compute_vqvae_loss(self, preds):
        return {"vq_commit": preds.get("vq_commit_loss")}

    def compute_simsiam_loss(
        self, preds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        def dist(p, z):
            p = F.normalize(p.flatten(1), dim=1)
            z = F.normalize(z.flatten(1).detach(), dim=1)
            return 2 - 2 * torch.mean(torch.sum(p * z, dim=1))

        assert "z" in preds and "p" in preds, "simsiam requires z and p"
        z1, z2 = torch.chunk(preds["z"], 2)
        p1, p2 = torch.chunk(preds["p"], 2)
        return {"simsiam": 0.5 * (dist(p1, z2) + dist(p2, z1))}

    def integral_loss(self, geometry, preds, tgts, idx_data, integral_loss_type="mse"):
        if self.training:
            # try vectorization for dataset-wide normalization
            if (
                self.dataset is not None
                and getattr(self.dataset, "normalization_scope", None) == "dataset"
            ):
                pred_df = self.denormalize_fn(0, df=preds["df"])
                pred_phi = (
                    self.denormalize_fn(0, phi=preds["phi"]) if "phi" in preds else None
                )
                tgt_phi = self.denormalize_fn(0, phi=tgts["phi"])
                tgt_eflux = self.denormalize_fn(0, flux=tgts["flux"])
            else:
                pred_df, pred_phi, tgt_phi, tgt_eflux = [], [], [], []
                for b, f in enumerate(idx_data["file_index"].tolist()):
                    pred_df.append(self.denormalize_fn(f, df=preds["df"][b]))
                    if "phi" in preds:
                        p_phi = (
                            preds["phi"][b].unsqueeze(0)
                            if preds["phi"][b].ndim == 2
                            else preds["phi"][b]
                        )
                        pred_phi.append(self.denormalize_fn(f, phi=p_phi))
                    t_phi = (
                        tgts["phi"][b].unsqueeze(0)
                        if tgts["phi"][b].ndim == 2
                        else tgts["phi"][b]
                    )
                    tgt_phi.append(self.denormalize_fn(f, phi=t_phi))
                    tgt_eflux.append(self.denormalize_fn(f, flux=tgts["flux"][b]))

                pred_df, tgt_phi, tgt_eflux = (
                    torch.stack(pred_df),
                    torch.stack(tgt_phi),
                    torch.stack(tgt_eflux),
                )
                pred_phi = torch.stack(pred_phi) if pred_phi else None
        else:
            pred_df, pred_phi, tgt_phi, tgt_eflux = (
                preds["df"],
                preds.get("phi"),
                tgts["phi"],
                tgts["flux"],
            )
            if tgt_phi.ndim == 5 and tgt_phi.shape[1] == 1:
                tgt_phi = tgt_phi.squeeze(1)

        if self.separate_zf and pred_df.shape[1] > 2:
            pred_df = recombine_zf(pred_df, dim=1)

        pphi_int, (pflux, eflux, _) = self.integrator(geometry, pred_df, pred_phi)

        monitor = {
            "phi_int_mse": F.mse_loss(pphi_int, tgt_phi).detach(),
            "flux_int_mse": (
                torch.abs(pflux).mean()
                + F.l1_loss(eflux.squeeze(), tgt_eflux.squeeze())
            ).detach(),
        }
        int_losses = (
            {"flux_int": monitor["flux_int_mse"], "phi_int": monitor["phi_int_mse"]}
            if integral_loss_type == "mse"
            else {
                "phi_int": self.compute_integral_loss(
                    pphi_int, tgt_phi, integral_loss_type, loss_name="phi_int"
                ),
                "flux_int": torch.abs(pflux).mean()
                + self.compute_integral_loss(
                    eflux, tgt_eflux, integral_loss_type, loss_name="flux_int"
                ),
            }
        )

        return int_losses, monitor, {"phi": pphi_int, "pflux": pflux, "eflux": eflux}

    def compute_spectral_losses(self, preds, tgts, geometry):
        spec_losses = {}
        if self.ds is None or "df" not in preds or "df" not in tgts:
            return spec_losses

        loss_type = self._get_current_loss_types()["spec"]

        def prep(d):
            x = d["df"]
            if self.separate_zf and x.shape[1] > 2:
                x = torch.cat([x[:, 0::2].sum(1, True), x[:, 1::2].sum(1, True)], dim=1)
            return x.float(), d.get("phi")

        p_df, p_phi_raw = prep(preds)
        t_df, t_phi_raw = prep(tgts)

        p_phi, (_, p_ef, _) = self.integrator_spec(geometry, p_df, p_phi_raw)
        t_phi, (_, t_ef, _) = self.integrator_spec(geometry, t_df, t_phi_raw)

        p_fft = physics.phi_fft(preds.get("phi", p_phi))
        t_fft = physics.phi_fft(tgts.get("phi", t_phi))
        p_diag = physics.diagnostics(p_fft, p_ef, self.ds, aggregate="mean")
        t_diag = physics.diagnostics(t_fft, t_ef, self.ds, aggregate="mean")

        for k in ["kxspec", "kyspec", "qspec", "phi_zf"]:
            if k in p_diag and k in t_diag:
                spec_losses[k] = self.compute_spectral_loss(
                    p_diag[k], t_diag[k], loss_type
                )

        # sort-based monotonicity on the un-aggregated spectra (shared with the NF path)
        p_diag_f = physics.diagnostics(p_fft, p_ef, self.ds, aggregate="none")
        mono = physics.monotonicity_loss(p_diag_f, keys=("qspec", "kyspec"))
        for k in ("qspec", "kyspec"):
            spec_losses[f"{k}_monotonicity"] = mono[f"{k} monotonicity loss"]

        spec_losses["mass"] = physics.mass_loss(p_df, t_df)
        return spec_losses

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        tgts: Dict[str, torch.Tensor],
        idx_data: Optional[Dict[str, torch.Tensor]] = None,
        geometry: Optional[Dict[str, torch.Tensor]] = None,
        compute_integrals: bool = True,
        progress_remaining: float = 1.0,
        separate_zf: bool = False,
        loss_type: str = "mse",
    ):
        losses, int_losses, int_monitor = {}, {}, {}

        if self.training:
            for k, sched in self.schedulers.items():
                if k in self.weights:
                    self.weights[k] = sched(progress_remaining)

        if sum([self.weights.get(k, 0.0) for k in self._int_losses]) > 0 or (
            not self.training and compute_integrals
        ):
            int_losses, int_monitor, integrated = self.integral_loss(
                geometry, preds, tgts, idx_data, self._get_current_loss_types()["int"]
            )
            losses.update(int_losses)

        if sum([self.weights.get(k, 0.0) for k in self._vae_losses]) > 0:
            losses.update(self.compute_vae_loss(preds))
        if sum([self.weights.get(k, 0.0) for k in self._vqvae_losses]) > 0:
            losses.update(self.compute_vqvae_loss(preds))

        if (
            sum([self.weights.get(k, 0.0) for k in self._spectral_losses]) > 0
            or (not self.training and compute_integrals)
        ) and geometry is not None:
            losses.update(self.compute_spectral_losses(preds, tgts, geometry))

        if (
            self.training
            and sum([self.weights.get(k, 0.0) for k in self._simsiam_losses]) > 0
        ):
            losses.update(self.compute_simsiam_loss(preds))

        # 3. Augmentation losses (VICReg, etc.)
        if self.training:
            for name in self._augmentation_losses:
                if "vicreg" in name and not "latent" in preds:
                    warnings.warn(
                        f"Latents not found in predictions for augmentation loss: {name}"
                    )
                    continue
                if name != "df_delta":
                    losses.update(getattr(self, f"compute_{name}")(preds["latent"]))

        special_keys = set(
            self._int_losses
            + self._vae_losses
            + self._vqvae_losses
            + self._spectral_losses
            + self._simsiam_losses
            + self._augmentation_losses
        )

        available_keys = list(set(tgts.keys()) | set(preds.keys()) | special_keys)
        nonzero_keys = [k for k, w in self.weights.items() if w > 0.0]
        if any((n not in available_keys) for n in nonzero_keys):
            nonzero_keys = [n for n in nonzero_keys if n in available_keys]

        if self.training:
            all_keys = nonzero_keys
        else:
            all_keys = list(set(self.weights.keys()) | set(losses.keys()))

        data_keys = [k for k in all_keys if k not in special_keys]
        if not self.training:
            data_keys.remove("df_delta") if "df_delta" in data_keys else None
        for k in data_keys:
            p, t = preds.get(k, torch.zeros_like(tgts[k])), tgts[k]
            if p.shape != t.shape:
                if k == "phi":
                    p = p.unsqueeze(0)
                elif k == "flux":
                    # ensure same dimensionality for scalar flux
                    p, t = p.flatten(), t.flatten()
            losses[k] = (
                self.compute_data_loss(p[:, :2], t[:, :2], loss_type=loss_type)
                + self.compute_data_loss(p[:, 2:], t[:, 2:], loss_type=loss_type)
                if k == "df" and separate_zf
                else self.compute_data_loss(p, t, loss_type=loss_type)
            )

        if self.training:
            monitor_mse = {}
            current_loss_types = self._get_current_loss_types()
            for k in data_keys:
                if k not in preds:
                    continue
                p, t = preds[k], tgts[k]
                if k == "flux":
                    p, t = p.flatten(), t.flatten()

                if k == "df" and separate_zf:
                    monitor_mse[f"{k}_mse"] = F.mse_loss(
                        p[:, :2], t[:, :2]
                    ) + F.mse_loss(p[:, 2:], t[:, 2:])
                else:
                    monitor_mse[f"{k}_mse"] = F.mse_loss(p, t)

            for k, v in losses.items():
                self._update_ema_loss_scale(k, v)
            norm_losses = self._apply_ema_normalization(losses)

            per_mode_losses = {}
            if "mask_modes" in self.augmentations:
                with torch.no_grad():
                    per_mode_losses = self.compute_per_mode_losses(
                        preds, tgts, loss_type="relative_mse"
                    )

            total_loss = sum(
                self.weights.get(k, 0.0) * norm_losses.get(k, 0.0)
                for k in all_keys
                if k in norm_losses
            )
            log_losses = {
                k: losses[k]
                for k in all_keys
                if k in losses and self.weights.get(k, 0.0) > 0
            }
            # only log individual _mse if the main loss is not already mse
            if current_loss_types.get("data") != "mse" and monitor_mse:
                log_losses.update(monitor_mse)

            log_losses.update(
                {"total_mse": sum(monitor_mse.values()) if monitor_mse else 0.0}
            )
            log_losses.update(int_monitor)
            log_losses.update(per_mode_losses)
            if self.ema_normalization_loss:
                log_losses.update(self.get_ema_statistics())
            return total_loss, log_losses

        total_loss = sum(losses.get(k, 0.0) for k in all_keys if k in losses)
        losses.update(int_monitor)
        return (
            total_loss,
            losses,
            integrated,
            {
                k: {
                    "value": (v.item() if isinstance(v, torch.Tensor) else v),
                    "log10": (
                        float(torch.log10(torch.as_tensor(v)).item()) if v > 0 else 0
                    ),
                }
                for k, v in losses.items()
            },
        )


class PINCGradientBalancer(GradientBalancer):
    """PINCGradientBalancer class."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str,
        scaler: torch.amp.GradScaler,
        clip_grad: bool = True,
        clip_to: float = 1.0,
        n_tasks: Optional[int] = None,
        deepspeed_engine=None,
    ):
        super().__init__(
            optimizer,
            "none" if mode == "full" else mode,
            scaler,
            clip_grad,
            clip_to,
            n_tasks,
            deepspeed_engine,
        )
        self.mode = mode

        if mode == "full":
            import conflictfree.grad_operator
            from conflictfree.grad_operator import ConFIGOperator

            def ConFIG_update(
                grads,
                weight_model=None,
                length_model=None,
                use_least_square=True,
                losses=None,
            ):
                from conflictfree.weight_model import EqualWeight
                from conflictfree.length_model import ProjectionLength

                weight_model, length_model = (
                    weight_model or EqualWeight(),
                    length_model or ProjectionLength(),
                )
                grads = (
                    torch.stack(grads) if not isinstance(grads, torch.Tensor) else grads
                )

                with torch.no_grad():
                    weights = weight_model.get_weights(
                        gradients=grads, losses=losses, device=grads.device
                    )
                    units = torch.nan_to_num(
                        grads / grads.norm(dim=1).unsqueeze(1), nan=0.0
                    )
                    try:
                        best_dir = torch.linalg.lstsq(units, weights).solution
                    except Exception:
                        best_dir = _wide_min_norm_solution(units, weights)
                    return length_model.rescale_length(
                        target_vector=best_dir, gradients=grads, losses=losses
                    )

            conflictfree.grad_operator.ConFIG_update = ConFIG_update
            self.operator = ConFIGOperator()
