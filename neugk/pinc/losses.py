"""
PINC-specific loss wrappers and gradient balancers.
Extends base_losses to add support for Spectral, VAE, and VQVAE losses,
plus EMA normalization and custom Conflict-Free Gradient Descent (ConFIG) patching.
"""

from typing import List, Callable, Dict, Optional, Any

import torch
import torch.nn.functional as F

from neugk.utils import recombine_zf
from neugk.integrals import FluxIntegral
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
        dataset: Optional[Any] = None,
        masked_mode_modeling: bool = False,
    ):
        super().__init__(
            weights=weights,
            schedulers=schedulers,
            denormalize_fn=denormalize_fn,
            separate_zf=separate_zf,
            real_potens=real_potens,
            masked_mode_modeling=masked_mode_modeling,
        )

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
            real_potens=real_potens, flux_fields=False, spectral_df=False
        )
        self.integrator_spec = FluxIntegral(
            real_potens=real_potens, flux_fields=True, spectral_df=True
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

    def compute_data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8,
        loss_type: Optional[str] = None,
    ) -> torch.Tensor:
        loss_type = loss_type or self._get_current_loss_types()["data"]

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
            p_c, t_c = (
                self.complex_metrics.to_complex(pred),
                self.complex_metrics.to_complex(target),
            )
            return (
                self.complex_metrics.complex_mse(p_c, t_c).mean()
                if "mse" in loss_type
                else self.complex_metrics.complex_l1(p_c, t_c).mean()
            )

        raise ValueError(f"unknown data loss type: {loss_type}")

    def compute_integral_loss(
        self, pred, target, loss_type="mse", eps=1e-8, loss_name="flux_int"
    ):
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type in ["relative_mse", "relative_l1", "log_error"]:
            return self.compute_data_loss(pred, target, eps, loss_type=loss_type)

        if loss_type == "adaptive_relative":
            alpha = 0.01
            attr = f"_target_ema_{loss_name.split('_')[0]}"
            curr_mean = torch.abs(target).mean().item()

            setattr(
                self,
                attr,
                (
                    curr_mean
                    if not hasattr(self, attr)
                    else alpha * curr_mean + (1 - alpha) * getattr(self, attr)
                ),
            )
            return ((pred - target) / max(getattr(self, attr), eps)).pow(2).mean()

        if loss_type in ["int_norm_mse", "int_norm_l1"]:
            key = "flux_std" if "flux" in loss_name else "phi_std"
            if key in self.dataset_stats:
                norm_err = (pred - target) / max(float(self.dataset_stats[key]), eps)
                return (
                    norm_err.pow(2) if "mse" in loss_type else torch.abs(norm_err)
                ).mean()
            return self.compute_data_loss(pred, target, eps, loss_type=loss_type)

        raise ValueError(f"unknown integral loss type: {loss_type}")

    def compute_spectral_loss(self, pred, target, loss_type="l1", eps=1e-8):
        if loss_type in ["l1", "mse", "relative_l1", "relative_mse"]:
            return self.compute_data_loss(pred, target, eps, loss_type=loss_type)

        if "normalized" in loss_type:
            scale = torch.mean(torch.abs(target)) + eps
            return (
                F.l1_loss(pred / scale, target / scale)
                if "l1" in loss_type
                else F.mse_loss(pred / scale, target / scale)
            )

        if "log" in loss_type:
            if "relative" in loss_type:
                pred, target = pred / (pred.sum() + eps), target / (target.sum() + eps)
            pl = torch.log(torch.abs(pred) + eps)
            tl = torch.log(torch.abs(target) + eps)
            return F.l1_loss(pl, tl) if "l1" in loss_type else F.mse_loss(pl, tl)

        raise ValueError(f"unknown spectral loss type: {loss_type}")

    def compute_vae_loss(self, preds):
        if "mu" not in preds or "logvar" not in preds:
            return {}
        return {
            "kl_div": -0.5
            * torch.mean(
                1 + preds["logvar"] - preds["mu"].pow(2) - preds["logvar"].exp()
            )
        }

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
            "phi_int_mse": F.mse_loss(pphi_int, tgt_phi),
            "flux_int_mse": torch.abs(pflux).mean()
            + F.l1_loss(eflux.squeeze(), tgt_eflux.squeeze()),
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
        nx, _, ny = phi_fft.shape[1:]

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

        f_zf = phi_fft.clone()
        f_zf[..., :zf_mode] = f_zf[..., zf_mode + 1 :] = 0.0
        diag["phi_zf"] = torch.fft.irfftn(
            torch.fft.fftshift(f_zf, dim=(1,)), dim=(1, 3), norm="forward", s=[nx, ny]
        )

        diag["qspec"] = eflux_field.sum(
            (1, 2, 3, 4) if eflux_field.dim() == 6 else (0, 1, 2, 3)
        )
        if eflux_field.dim() == 5:
            diag["qspec"] = diag["qspec"].unsqueeze(0)

        return diag

    def compute_spectral_losses(self, preds, tgts, geometry):
        spec_losses = {}
        if self.ds is None or "df" not in preds or "df" not in tgts:
            return spec_losses

        loss_type = self._get_current_loss_types()["spec"]

        def prep(d):
            x = d["df"]
            if self.separate_zf and x.shape[1] > 2:
                # generalized separate_zf recombination
                x = torch.cat([x[:, 0::2].sum(1, True), x[:, 1::2].sum(1, True)], dim=1)
            return x.float(), d.get("phi")

        p_df, p_phi_raw = prep(preds)
        t_df, t_phi_raw = prep(tgts)

        p_phi, (_, p_ef, _) = self.integrator_spec(geometry, p_df, p_phi_raw)
        t_phi, (_, t_ef, _) = self.integrator_spec(geometry, t_df, t_phi_raw)

        p_fft, t_fft = (
            self.phi_fft(preds.get("phi", p_phi)),
            self.phi_fft(tgts.get("phi", t_phi)),
        )
        p_diag, t_diag = (
            self.diagnostics(p_fft, p_ef, self.ds),
            self.diagnostics(t_fft, t_ef, self.ds),
        )

        for k in ["kxspec", "kyspec", "qspec", "phi_zf"]:
            if k in p_diag and k in t_diag:
                spec_losses[k] = self.compute_spectral_loss(
                    p_diag[k], t_diag[k], loss_type
                )

        p_diag_f, t_diag_f = (
            self.diagnostics(p_fft, p_ef, self.ds, aggregate="none"),
            self.diagnostics(t_fft, t_ef, self.ds, aggregate="none"),
        )

        for k in ["qspec", "kyspec"]:
            if k in p_diag_f and k in t_diag_f:
                try:
                    p_s = torch.nan_to_num(torch.log1p(p_diag_f[k]))
                    t_s = torch.nan_to_num(torch.log1p(t_diag_f[k]))
                    losses = [
                        torch.mean(
                            torch.clamp(
                                (
                                    p_s[b, torch.argmax(p_s[b]).item() :][1:]
                                    - p_s[b, torch.argmax(p_s[b]).item() :][:-1]
                                )
                                - torch.clamp(
                                    (
                                        t_s[b, torch.argmax(p_s[b]).item() :][1:]
                                        - t_s[b, torch.argmax(p_s[b]).item() :][:-1]
                                    ).max(),
                                    min=0,
                                ),
                                min=0,
                            )
                        )
                        for b in range(p_s.shape[0])
                        if len(p_s[b, torch.argmax(p_s[b]).item() :]) > 1
                    ]
                    spec_losses[f"{k}_monotonicity"] = (
                        torch.stack(losses).mean()
                        if losses
                        else torch.tensor(0.0, device=p_s.device)
                    )
                except Exception:
                    spec_losses[f"{k}_monotonicity"] = torch.tensor(
                        0.0, device=p_diag_f[k].device
                    )

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
            if p.shape != t.shape and k == "phi":
                p = p.unsqueeze(0)
            losses[k] = (
                self.compute_data_loss(p[:, :2], t[:, :2])
                + self.compute_data_loss(p[:, 2:], t[:, 2:])
                if k == "df" and separate_zf
                else self.compute_data_loss(p, t)
            )

        if self.training:
            monitor_mse = {
                f"{k}_mse": (
                    F.mse_loss(preds[k][:, :2], tgts[k][:, :2])
                    + F.mse_loss(preds[k][:, 2:], tgts[k][:, 2:])
                    if k == "df" and separate_zf
                    else F.mse_loss(preds[k], tgts[k])
                )
                for k in data_keys
                if k in preds
            }
            for k, v in losses.items():
                self._update_ema_loss_scale(k, v)
            norm_losses = self._apply_ema_normalization(losses)

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
            log_losses.update({"total_mse": sum(monitor_mse.values())})
            log_losses.update(int_monitor)
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
    ):
        super().__init__(
            optimizer,
            "none" if mode == "full" else mode,
            scaler,
            clip_grad,
            clip_to,
            n_tasks,
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
