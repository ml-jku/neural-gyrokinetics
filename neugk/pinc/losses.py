"""
PINC-specific loss wrappers and gradient balancers.
Extends base_losses to add support for Spectral, VAE, and VQVAE losses, 
plus EMA normalization and custom Conflict-Free Gradient Descent (ConFIG) patching.
"""

from typing import List, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from neugk.integrals import FluxIntegral
from neugk.losses import LossWrapper, GradientBalancer


def _wide_min_norm_solution(
    units: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute minimal-norm solution of units x = weights for the
    underdetermined case (m << N) using Gram matrix inversion.
    """
    # Filter out any (near) zero rows to avoid singular G
    row_norms = units.norm(dim=1)
    keep = row_norms > 0
    if keep.sum() == 0:
        return torch.zeros(units.shape[1], device=units.device, dtype=units.dtype)
    U = units[keep]  # (m', N)
    w = weights[keep]  # (m',)
    # Gram matrix
    G = U @ U.t()  # (m', m')
    # add regularization
    reg = eps * G.diag().mean()
    G = G + reg * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    # Solve G a = w
    a = torch.linalg.solve(G, w)
    # Compose parameter-space solution
    x = a @ U
    return x


class PINCLossWrapper(LossWrapper):
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
    ):
        # Initialize base class
        super().__init__(
            weights=weights,
            schedulers=schedulers,
            denormalize_fn=denormalize_fn,
            separate_zf=separate_zf,
            real_potens=real_potens,
        )

        # Extended loss categories
        self._vae_losses = ["kl_div"]
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
        )

        self.integrator_spec = FluxIntegral(
            real_potens=real_potens,
            flux_fields=True,
            spectral_df=True,
        )

        # Loss configuration
        self.loss_type = loss_type
        self.integral_loss_type = integral_loss_type
        self.spectral_loss_type = spectral_loss_type
        self.eval_loss_type = eval_loss_type
        self.eval_integral_loss_type = eval_integral_loss_type
        self.eval_spectral_loss_type = eval_spectral_loss_type

        # Stats and Normalization
        self.dataset_stats = dataset_stats or {}
        self.ds = ds
        self.loss_normalizer = {}
        self.normalize_losses = getattr(loss_type, "normalize_losses", False)

        # EMA Normalization
        self.ema_normalization_loss = ema_normalization_loss or []
        self.ema_beta = ema_beta
        self._ema_loss_scales = {}
        self._ema_initialized = set()

        self.complex_metrics = None

    @property
    def all_losses(self):
        return (
            super().all_losses
            + self._vae_losses
            + self._vqvae_losses
            + self._spectral_losses
            + self._simsiam_losses
        )

    def _update_ema_loss_scale(self, loss_name: str, loss_value: torch.Tensor):
        if loss_name not in self.ema_normalization_loss:
            return
        current_scale = loss_value.detach().item()
        if loss_name not in self._ema_initialized:
            self._ema_loss_scales[loss_name] = current_scale
            self._ema_initialized.add(loss_name)
        else:
            self._ema_loss_scales[loss_name] = (
                self.ema_beta * self._ema_loss_scales[loss_name]
                + (1 - self.ema_beta) * current_scale
            )

    def _apply_ema_normalization(
        self, losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        normalized = {}
        for name, val in losses.items():
            if name in self.ema_normalization_loss and name in self._ema_loss_scales:
                scale = self._ema_loss_scales[name]
                normalized[name] = val / scale if scale > 1e-8 else val
            else:
                normalized[name] = val
        return normalized

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

    def compute_data_loss(
        self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        loss_type = self._get_current_loss_types()["data"]

        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type == "l1":
            return F.l1_loss(pred, target)
        if loss_type == "huber":
            return F.huber_loss(pred, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        if loss_type == "relative_mse":
            return ((pred - target) / (torch.abs(target) + eps)).pow(2).mean()
        if loss_type == "relative_l1":
            return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()
        if loss_type == "log_error":
            return F.mse_loss(
                torch.log(torch.abs(pred) + eps), torch.log(torch.abs(target) + eps)
            )
        if loss_type == "log_l1_error":
            return F.l1_loss(
                torch.log(torch.abs(pred) + eps), torch.log(torch.abs(target) + eps)
            )
        if loss_type == "log_cosh":
            return torch.log(torch.cosh(pred - target)).mean()
        if "complex" in loss_type and self.complex_metrics:
            p_c, t_c = self.complex_metrics.to_complex(
                pred
            ), self.complex_metrics.to_complex(target)
            return (
                self.complex_metrics.complex_mse(p_c, t_c).mean()
                if "mse" in loss_type
                else self.complex_metrics.complex_l1(p_c, t_c).mean()
            )

        raise ValueError(f"Unknown data loss type: {loss_type}")

    def compute_integral_loss(
        self, pred, target, loss_type="mse", eps=1e-8, loss_name="flux_int"
    ):
        # Specialized logic for integral losses (adaptive, normalized, etc.)
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type in ["relative_mse", "relative_l1", "log_error"]:
            # Reuse logic from data loss for standard relative types
            # Temporarily force data loss type to match integral request
            prev_type = self.loss_type if self.training else self.eval_loss_type
            if self.training:
                self.loss_type = loss_type
            else:
                self.eval_loss_type = loss_type
            loss = self.compute_data_loss(pred, target, eps)
            if self.training:
                self.loss_type = prev_type
            else:
                self.eval_loss_type = prev_type
            return loss

        if loss_type == "adaptive_relative":
            alpha = 0.01
            attr = f"_target_ema_{loss_name.split('_')[0]}"  # _target_ema_flux or _target_ema_phi

            if not hasattr(self, attr):
                setattr(self, attr, torch.abs(target).mean().item())
            else:
                setattr(
                    self,
                    attr,
                    alpha * torch.abs(target).mean().item()
                    + (1 - alpha) * getattr(self, attr),
                )

            scale = max(getattr(self, attr), eps)
            return ((pred - target) / scale).pow(2).mean()

        if loss_type in ["int_norm_mse", "int_norm_l1"]:
            key = "flux_std" if "flux" in loss_name else "phi_std"
            if key in self.dataset_stats:
                scale = max(float(self.dataset_stats[key]), eps)
                norm_err = (pred - target) / scale
                return (
                    norm_err.pow(2) if "mse" in loss_type else torch.abs(norm_err)
                ).mean()
            # Fallback
            return self.compute_data_loss(pred, target, eps)

        raise ValueError(f"Unknown integral loss type: {loss_type}")

    def compute_spectral_loss(self, pred, target, loss_type="l1", eps=1e-8):
        if loss_type in ["l1", "mse", "relative_l1", "relative_mse"]:
            # Reuse data loss logic
            prev = self.loss_type if self.training else self.eval_loss_type
            if self.training:
                self.loss_type = loss_type
            else:
                self.eval_loss_type = loss_type
            loss = self.compute_data_loss(pred, target, eps)
            if self.training:
                self.loss_type = prev
            else:
                self.eval_loss_type = prev
            return loss

        if "normalized" in loss_type:
            scale = torch.mean(torch.abs(target)) + eps
            if "l1" in loss_type:
                return F.l1_loss(pred / scale, target / scale)
            return F.mse_loss(pred / scale, target / scale)

        if "log" in loss_type:
            # log_l1, log_mse, log_relative_l1
            if "relative" in loss_type:
                pred = pred / (pred.sum() + eps)
                target = target / (target.sum() + eps)
            pl, tl = torch.log(torch.abs(pred) + eps), torch.log(
                torch.abs(target) + eps
            )
            return F.l1_loss(pl, tl) if "l1" in loss_type else F.mse_loss(pl, tl)

        raise ValueError(f"Unknown spectral loss type: {loss_type}")

    def compute_vae_loss(self, preds):
        if "mu" not in preds or "logvar" not in preds:
            return {}
        # KL(q(z|x) || N(0,I))
        return {
            "kl_div": -0.5
            * torch.mean(
                1 + preds["logvar"] - preds["mu"].pow(2) - preds["logvar"].exp()
            )
        }

    def compute_vqvae_loss(self, preds):
        return {"vq_commit": preds.get("vq_commit_loss", None)}

    def compute_simsiam_loss(
        self, preds: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        def D(p, z):
            p, z = p.flatten(1), z.flatten(1)
            z = z.detach()
            p, z = F.normalize(p, dim=1), F.normalize(z, dim=1)
            # mse instead of basic similarity
            return 2 - 2 * torch.mean(torch.sum(p * z, dim=1))
            # return -torch.mean(torch.sum(p * z, dim=1))

        assert all(k in preds for k in ["z", "p"]), "SimSiam requires z and p in preds."
        z1, z2 = torch.chunk(preds["z"], 2)
        p1, p2 = torch.chunk(preds["p"], 2)
        return {"simsiam": 0.5 * (D(p1, z2) + D(p2, z1))}

    def integral_loss(self, geometry, preds, tgts, idx_data, integral_loss_type="mse"):
        # Note: Override base implementation to return 3-tuple (losses, monitor, integrated)
        # and support custom loss types

        # 1. Denormalize (reusing base logic manually because base method signature is fixed)
        if self.training:
            pred_df, pred_phi, tgt_phi, tgt_eflux = [], [], [], []
            for b, f in enumerate(idx_data["file_index"].tolist()):
                pred_df.append(self.denormalize_fn(f, df=preds["df"][b]))
                if "phi" in preds:
                    p_phi = preds["phi"][b]
                    if p_phi.ndim == 2:
                        p_phi = p_phi.unsqueeze(0)
                    pred_phi.append(self.denormalize_fn(f, phi=p_phi))

                t_phi = tgts["phi"][b]
                if t_phi.ndim == 2:
                    t_phi = t_phi.unsqueeze(0)
                tgt_phi.append(self.denormalize_fn(f, phi=t_phi))
                tgt_eflux.append(self.denormalize_fn(f, flux=tgts["flux"][b]))

            pred_df = torch.stack(pred_df)
            pred_phi = torch.stack(pred_phi) if pred_phi else None
            tgt_phi = torch.stack(tgt_phi)
            tgt_eflux = torch.stack(tgt_eflux)
        else:
            pred_df, pred_phi = preds["df"], preds.get("phi")
            tgt_phi, tgt_eflux = tgts["phi"], tgts["flux"]

            # squeeze channel dim to match integrator output shape
            if tgt_phi.ndim == 5 and tgt_phi.shape[1] == 1:
                tgt_phi = tgt_phi.squeeze(1)

        # Merge zonal flows if needed
        if self.separate_zf and pred_df.shape[1] > 2:
            if pred_df.shape[1] == 4:
                pred_df = pred_df[:, [0, 1]] + pred_df[:, [2, 3]]
            else:
                pred_df = torch.cat(
                    [pred_df[:, 0::2].sum(1, True), pred_df[:, 1::2].sum(1, True)],
                    dim=1,
                )

        pphi_int, (pflux, eflux, _) = self.integrator(geometry, pred_df, pred_phi)

        int_losses = {}
        monitor = {}

        # Always compute MSE for monitoring
        monitor["phi_int_mse"] = F.mse_loss(pphi_int, tgt_phi)
        # momentum flux should be 0
        monitor["flux_int_mse"] = torch.abs(pflux).mean()
        # heat flux target
        monitor["flux_int_mse"] += F.l1_loss(eflux.squeeze(), tgt_eflux.squeeze())

        # Compute actual training objective
        if integral_loss_type == "mse":
            int_losses["flux_int"] = monitor["flux_int_mse"]
            int_losses["phi_int"] = monitor["phi_int_mse"]
        else:
            int_losses["phi_int"] = self.compute_integral_loss(
                pphi_int, tgt_phi, integral_loss_type, loss_name="phi_int"
            )
            eflux_loss = self.compute_integral_loss(
                eflux, tgt_eflux, integral_loss_type, loss_name="flux_int"
            )
            int_losses["flux_int"] = torch.abs(pflux).mean() + eflux_loss

        return int_losses, monitor, {"phi": pphi_int, "pflux": pflux, "eflux": eflux}

    def phi_fft(self, phi, norm="forward"):
        phi = phi.float()
        phi_complex = (
            torch.view_as_complex(phi.permute(0, 2, 3, 4, 1).contiguous())
            if phi.shape[1] == 2
            else phi.squeeze(1).to(torch.complex64)
        )
        return torch.fft.fftshift(
            torch.fft.fftn(phi_complex, dim=(1, 3), norm=norm), dim=(1,)
        )

    def diagnostics(self, phi_fft, eflux_field, ds, zf_mode=0, aggregate="mean"):
        diag = {}
        nx, ns, ny = phi_fft.shape[1:]

        # Kx, Ky spectra
        kxspec = torch.sum(torch.abs(phi_fft) ** 2, dim=(2, 3)) * ds
        kyspec = torch.sum(torch.abs(phi_fft) ** 2, dim=(1, 2)) * ds

        diag["kxspec"] = (
            kxspec
            if aggregate == "none"
            else (
                torch.sum(kxspec, dim=1)
                if aggregate == "mean"
                else kxspec[:, kxspec.shape[1] // 2]
            )
        )
        diag["kyspec"] = (
            kyspec
            if aggregate == "none"
            else (
                torch.sum(kyspec, dim=1)
                if aggregate == "mean"
                else kyspec[:, kyspec.shape[1] // 2]
            )
        )

        # Zonal Flow
        f_zf = phi_fft.clone()
        f_zf[..., :zf_mode] = 0.0
        f_zf[..., zf_mode + 1 :] = 0.0
        f_zf = torch.fft.fftshift(f_zf, dim=(1,))
        diag["phi_zf"] = torch.fft.irfftn(f_zf, dim=(1, 3), norm="forward", s=[nx, ny])

        # Q Spectrum
        dims = (1, 2, 3, 4) if eflux_field.dim() == 6 else (0, 1, 2, 3)
        diag["qspec"] = eflux_field.sum(dims)
        if eflux_field.dim() == 5:
            diag["qspec"] = diag["qspec"].unsqueeze(0)

        return diag

    def compute_spectral_losses(self, preds, tgts, geometry):
        spec_losses = {}
        if self.ds is None or "df" not in preds or "df" not in tgts:
            return spec_losses

        loss_type = self._get_current_loss_types()["spec"]

        # Prepare inputs (float32, handle channels)
        def prep(d):
            x = d["df"]
            if x.shape[1] == 4:
                x = x[:, [0, 1]] + x[:, [2, 3]]
            return x.float(), d.get("phi")

        p_df, p_phi_raw = prep(preds)
        t_df, t_phi_raw = prep(tgts)

        # Integrate & FFT
        p_phi, (_, p_ef, _) = self.integrator_spec(geometry, p_df, p_phi_raw)
        t_phi, (_, t_ef, _) = self.integrator_spec(geometry, t_df, t_phi_raw)

        p_fft, t_fft = self.phi_fft(preds.get("phi", p_phi)), self.phi_fft(
            tgts.get("phi", t_phi)
        )

        # Diagnostics
        p_diag = self.diagnostics(p_fft, p_ef, self.ds)
        t_diag = self.diagnostics(t_fft, t_ef, self.ds)

        # Standard Spectral Losses
        for k in ["kxspec", "kyspec", "qspec", "phi_zf"]:
            if k in p_diag and k in t_diag:
                spec_losses[k] = self.compute_spectral_loss(
                    p_diag[k], t_diag[k], loss_type
                )

        # Monotonicity Losses
        p_diag_f = self.diagnostics(p_fft, p_ef, self.ds, aggregate="none")
        t_diag_f = self.diagnostics(t_fft, t_ef, self.ds, aggregate="none")

        for k in ["qspec", "kyspec"]:
            if k in p_diag_f and k in t_diag_f:
                try:
                    p_s, t_s = torch.nan_to_num(
                        torch.log1p(p_diag_f[k])
                    ), torch.nan_to_num(torch.log1p(t_diag_f[k]))
                    losses = []
                    for b in range(p_s.shape[0]):
                        pk = torch.argmax(p_s[b]).item()
                        pt, tt = p_s[b, pk:], t_s[b, pk:]
                        if len(pt) > 1:
                            tol = torch.clamp((tt[1:] - tt[:-1]).max(), min=0)
                            losses.append(
                                torch.mean(torch.clamp((pt[1:] - pt[:-1]) - tol, min=0))
                            )
                    spec_losses[f"{k}_monotonicity"] = (
                        torch.stack(losses).mean()
                        if losses
                        else torch.tensor(0.0, device=p_s.device)
                    )
                except Exception:
                    spec_losses[f"{k}_monotonicity"] = torch.tensor(
                        0.0, device=p_diag_f[k].device
                    )

        # Mass Conservation
        spec_losses["mass"] = self.compute_spectral_loss(
            p_df.sum(), t_df.sum(), "log_l1"
        )
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
    ):
        losses = {}
        int_losses, int_monitor = {}, {}

        # Scheduler Update
        if self.training:
            for k in self.schedulers:
                if k in self.weights:
                    self.weights[k] = self.schedulers[k](progress_remaining)

        # 1. Compute Integrals
        do_ints = not self.training and compute_integrals
        if sum([self.weights.get(k, 0.0) for k in self._int_losses]) > 0 or do_ints:
            # Use effective integral loss type
            eff_type = self._get_current_loss_types()["int"]
            int_losses, int_monitor, integrated = self.integral_loss(
                geometry, preds, tgts, idx_data, eff_type
            )
            losses.update(int_losses)

        # 2. Compute VAE/VQVAE/Spectral
        if sum([self.weights.get(k, 0.0) for k in self._vae_losses]) > 0:
            losses.update(self.compute_vae_loss(preds))

        if sum([self.weights.get(k, 0.0) for k in self._vqvae_losses]) > 0:
            losses.update(self.compute_vqvae_loss(preds))

        do_spec = not self.training and compute_integrals
        if (
            sum([self.weights.get(k, 0.0) for k in self._spectral_losses]) > 0
            or do_spec
        ) and geometry is not None:
            losses.update(self.compute_spectral_losses(preds, tgts, geometry))

        # simsiam only in training
        if (
            self.training
            and sum([self.weights.get(k, 0.0) for k in self._simsiam_losses]) > 0
        ):
            losses.update(self.compute_simsiam_loss(preds))

        special_keys = set(
            self._int_losses
            + self._vae_losses
            + self._vqvae_losses
            + self._spectral_losses
            + self._simsiam_losses
        )
        available_keys = list(set(tgts.keys()) | set(preds.keys()) | special_keys)
        nonzero_keys = [k for k, w in self.weights.items() if w > 0.0]
        if any((n not in available_keys) for n in nonzero_keys):
            # TODO communicate weight dict mismatch to the user
            # warnings.warn(f"keys mitmatch: {available_keys} vs {nonzero_keys}")
            nonzero_keys = [n for n in nonzero_keys if n in available_keys]

        if self.training:
            all_keys = nonzero_keys
        else:

            all_keys = list(set(self.weights.keys()) | set(losses.keys()))

        data_keys = [k for k in all_keys if k not in special_keys]

        for k in data_keys:
            if k not in preds:
                preds[k] = torch.zeros_like(tgts[k])

            p, t = preds[k], tgts[k]
            if p.shape != t.shape and k == "phi":
                p = p.unsqueeze(0)

            if k == "df" and separate_zf:
                losses[k] = self.compute_data_loss(
                    p[:, :2], t[:, :2]
                ) + self.compute_data_loss(p[:, 2:], t[:, 2:])
            else:
                losses[k] = self.compute_data_loss(p, t)

        # 4. Final Aggregation & EMA
        if self.training:
            # Monitoring MSEs
            monitor_mse = {}
            with torch.no_grad():
                for k in data_keys:
                    if k not in preds:
                        continue
                    if k == "df" and separate_zf:
                        monitor_mse[f"{k}_mse"] = F.mse_loss(
                            preds[k][:, :2], tgts[k][:, :2]
                        ) + F.mse_loss(preds[k][:, 2:], tgts[k][:, 2:])
                    else:
                        monitor_mse[f"{k}_mse"] = F.mse_loss(preds[k], tgts[k])

            # EMA Normalization
            for k, v in losses.items():
                self._update_ema_loss_scale(k, v)
            norm_losses = self._apply_ema_normalization(losses)

            # Reweight
            total_loss = sum(
                self.weights.get(k, 0.0) * norm_losses.get(k, 0.0)
                for k in all_keys
                if k in norm_losses
            )

            # Return dict for logging
            log_losses = {
                k: losses[k]
                for k in all_keys
                if k in losses and self.weights.get(k, 0.0) > 0
            }
            log_losses.update({"total_mse": sum(monitor_mse.values())})
            log_losses.update(int_monitor)
            if self.ema_normalization_loss:
                log_losses.update(self.get_ema_statistics())

            return total_loss, log_losses

        else:
            # Eval mode: return unweighted sum of available requested losses
            total_loss = sum(losses.get(k, 0.0) for k in all_keys if k in losses)
            losses.update(int_monitor)

            # Stats for debugging
            stats = {}
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                stats[k] = {
                    "value": val,
                    "log10": (
                        float(torch.log10(torch.tensor(val)).item()) if val > 0 else 0
                    ),
                }

            return total_loss, losses, integrated, stats


class PINCGradientBalancer(GradientBalancer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str,
        scaler: torch.amp.GradScaler,
        clip_grad: bool = True,
        clip_to: float = 1.0,
        n_tasks: Optional[int] = None,
    ):
        # Initialize with 'none' first to skip standard operator setup if we need custom logic
        init_mode = "none" if mode == "full" else mode
        super().__init__(optimizer, init_mode, scaler, clip_grad, clip_to, n_tasks)

        self.mode = mode  # Restore intended mode

        if mode == "full":
            # Monkey patch ConFIG for wide matrices
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

                if weight_model is None:
                    weight_model = EqualWeight()
                if length_model is None:
                    length_model = ProjectionLength()
                if not isinstance(grads, torch.Tensor):
                    grads = torch.stack(grads)

                with torch.no_grad():
                    weights = weight_model.get_weights(
                        gradients=grads, losses=losses, device=grads.device
                    )
                    units = torch.nan_to_num(
                        grads / (grads.norm(dim=1).unsqueeze(1)), nan=0.0
                    )

                    try:
                        best_dir = torch.linalg.lstsq(units, weights).solution
                    except Exception:
                        best_dir = _wide_min_norm_solution(units, weights)

                    return length_model.rescale_length(
                        target_vector=best_dir, gradients=grads, losses=losses
                    )

            # Apply patch and init operator
            conflictfree.grad_operator.ConFIG_update = ConFIG_update
            self.operator = ConFIGOperator()

        elif mode == "pseudo":

            self._debug_step = 0
            # Operator already init by super()

    def forward(self, model, weighted_loss, losses):
        # PINC adds debug printing for pseudo momentum
        if (
            self.mode == "pseudo"
            and hasattr(self, "_debug_step")
            and self._debug_step < 5
        ):
            from conflictfree.utils import get_gradient_vector

            self.optimizer.zero_grad(set_to_none=True)
            idx, loss_i = self.loss_selector.select(1, losses)

            print(
                f"Pseudo momentum step {self._debug_step}: "
                f"selected idx={idx}, loss={loss_i.item():.6f}"
            )
            self._debug_step += 1

            self.scaler.scale(loss_i).backward()
            self.operator.update_gradient(model, idx, get_gradient_vector(model))

            if self.clip_grad:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(model.parameters(), self.clip_to)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return model

        return super().forward(model, weighted_loss, losses)
