"""Evaluation metrics for complex-valued fields and physical quantities."""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import defaultdict
from abc import abstractmethod

import warnings
import torch.distributed as dist
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from neugk.utils import save_model_and_config


class ComplexMetrics:
    """Computes various metrics for complex-valued tensors."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def to_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor from [bs, c, ...] to complex representation"""
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        channels = tensor.shape[1]
        if channels == 2:
            return torch.complex(tensor[:, 0], tensor[:, 1])
        if channels % 2 == 0:
            # sum real and imaginary parts if split
            real_parts = tensor[:, 0::2].sum(dim=1)
            imag_parts = tensor[:, 1::2].sum(dim=1)
            return torch.complex(real_parts, imag_parts)
        raise ValueError(f"expected even number of channels, got {channels}")

    def complex_pearson(
        self, z1: torch.Tensor, z2: torch.Tensor, dims: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complex pearson correlation coefficient"""
        dims = dims or list(range(1, z1.dim()))

        mu1 = z1.mean(dim=dims, keepdim=True)
        mu2 = z2.mean(dim=dims, keepdim=True)

        z1_c = (z1 - mu1).to(torch.complex64)
        z2_c = (z2 - mu2).to(torch.complex64)

        cov = (z1_c * z2_c.conj()).mean(dim=dims)
        var1 = (z1_c.abs() ** 2).mean(dim=dims)
        var2 = (z2_c.abs() ** 2).mean(dim=dims)

        corr = (cov / torch.sqrt(var1 * var2 + self.epsilon)).to(z1.dtype)

        if torch.allclose(z1, z2, rtol=1e-10, atol=1e-10):
            return torch.ones_like(corr.abs()), torch.zeros_like(corr.angle())
        return corr.abs(), corr.angle()

    def phase_locking_value(
        self, z1: torch.Tensor, z2: torch.Tensor, dims: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Phase locking value (plv)"""
        dims = dims or list(range(1, z1.dim()))

        mask = (z1.abs() > self.epsilon) & (z2.abs() > self.epsilon)
        z1_unit = torch.where(mask, z1 / z1.abs(), torch.zeros_like(z1))
        z2_unit = torch.where(mask, z2 / z2.abs(), torch.zeros_like(z2))

        valid_count = mask.sum(dim=dims, keepdim=True).float()
        phase_diff = z1_unit * z2_unit.conj()

        return torch.where(
            valid_count > 0,
            phase_diff.sum(dim=dims).abs() / valid_count.squeeze(),
            torch.ones_like(valid_count.squeeze()),
        )

    def complex_ssim(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        dims: Optional[List[int]] = None,
        c1: float = 0.01,
        c2: float = 0.03,
    ) -> torch.Tensor:
        """Complex structural similarity index (cssim)"""
        dims = dims or list(range(1, z1.dim()))

        mu1 = z1.mean(dim=dims)
        mu2 = z2.mean(dim=dims)
        mu1_k = z1.mean(dim=dims, keepdim=True)
        mu2_k = z2.mean(dim=dims, keepdim=True)

        var1 = ((z1 - mu1_k).abs() ** 2).mean(dim=dims)
        var2 = ((z2 - mu2_k).abs() ** 2).mean(dim=dims)
        cov12 = ((z1 - mu1_k) * (z2 - mu2_k).conj()).mean(dim=dims)

        data_range = max(z1.abs().max().item(), z2.abs().max().item())
        c1 = (c1 * data_range) ** 2
        c2 = (c2 * data_range) ** 2

        num = (2 * mu1.abs() * mu2.abs() + c1) * (2 * cov12.abs() + c2)
        den = (mu1.abs() ** 2 + mu2.abs() ** 2 + c1) * (var1 + var2 + c2)
        return num / den

    def complex_mse(
        self, z1: torch.Tensor, z2: torch.Tensor, dims: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Complex mean squared error"""
        dims = dims or list(range(1, z1.dim()))
        diff = z1 - z2
        return torch.mean(diff.real**2 + diff.imag**2, dim=dims).mean()

    def complex_l1(
        self, z1: torch.Tensor, z2: torch.Tensor, dims: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Complex l1 norm (mean absolute error)"""
        dims = dims or list(range(1, z1.dim()))
        diff = z1 - z2
        return torch.mean(diff.real.abs() + diff.imag.abs(), dim=dims).mean()

    def spectral_energy_metric(
        self, z1: torch.Tensor, z2: torch.Tensor, dims: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Spectral energy difference metric"""
        dims = dims or list(range(1, z1.dim()))
        energy1 = (z1.abs() ** 2).sum(dim=dims)
        energy2 = (z2.abs() ** 2).sum(dim=dims)
        return (torch.abs(energy1 - energy2) / (energy1 + self.epsilon)).mean()

    def _to_spectral_domain(
        self, z: torch.Tensor, spatial_dims: Tuple[int, int] = (-2, -1)
    ) -> torch.Tensor:
        """Transform complex tensor to spectral domain using fft"""
        z_fft = torch.fft.fftn(z, dim=spatial_dims, norm="forward")
        return torch.fft.fftshift(z_fft, dim=spatial_dims)

    def magnitude_weighted_phase_coherence(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        spatial_dims: Tuple[int, int] = (-2, -1),
    ) -> torch.Tensor:
        """Magnitude-weighted phase coherence (mwpc)"""
        z1_fft = self._to_spectral_domain(z1, spatial_dims)
        z2_fft = self._to_spectral_domain(z2, spatial_dims)

        mag_prod = z1_fft.abs() * z2_fft.abs()
        phase_factor = torch.where(
            mag_prod > self.epsilon,
            (z1_fft * z2_fft.conj()) / mag_prod,
            torch.zeros_like(z1_fft),
        )

        num = (mag_prod * phase_factor).sum(dim=spatial_dims)
        den = mag_prod.sum(dim=spatial_dims)
        return (num.abs() / (den + self.epsilon)).mean()

    def complex_psnr(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        spatial_dims: Tuple[int, int] = (-2, -1),
    ) -> torch.Tensor:
        """Peak signal-to-noise ratio for complex-valued data"""
        peak_value = z1.abs().flatten(start_dim=1).max(dim=1)[0]
        mse = ((z1 - z2).abs() ** 2).flatten(start_dim=1).mean(dim=1)
        return (20 * torch.log10(peak_value / (torch.sqrt(mse) + self.epsilon))).mean()

    def kx_ky_analysis(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        spatial_dims: Tuple[int, int] = (-2, -1),
    ) -> Dict[str, float]:
        """Compute kx and ky directional spectral analysis"""
        # transform to spectral
        z1_fft = self._to_spectral_domain(z1, spatial_dims)
        z2_fft = self._to_spectral_domain(z2, spatial_dims)
        psd1 = z1_fft.abs() ** 2
        psd2 = z2_fft.abs() ** 2

        # compute directional spectra
        kx_spec1, kx_spec2 = psd1.sum(dim=spatial_dims[-1]), psd2.sum(
            dim=spatial_dims[-1]
        )
        ky_spec1, ky_spec2 = psd1.sum(dim=spatial_dims[-2]), psd2.sum(
            dim=spatial_dims[-2]
        )

        def compute_corr(spec1, spec2):
            flat1 = spec1.flatten(start_dim=0, end_dim=-2) if spec1.dim() > 1 else spec1
            flat2 = spec2.flatten(start_dim=0, end_dim=-2) if spec2.dim() > 1 else spec2
            flat1 = flat1.unsqueeze(0 if flat1.dim() == 1 else 1)
            flat2 = flat2.unsqueeze(0 if flat2.dim() == 1 else 1)
            return F.cosine_similarity(flat1, flat2, dim=-1).mean().item()

        # radial analysis
        non_spatial_dims = tuple(range(1, len(psd1.shape) - 2))
        psd1_avg = psd1.mean(dim=non_spatial_dims) if non_spatial_dims else psd1
        psd2_avg = psd2.mean(dim=non_spatial_dims) if non_spatial_dims else psd2

        rad_corr = F.cosine_similarity(
            psd1_avg.mean(dim=0).flatten().unsqueeze(0),
            psd2_avg.mean(dim=0).flatten().unsqueeze(0),
            dim=1,
        ).item()

        p1, p2 = psd1.sum(dim=spatial_dims).mean(), psd2.sum(dim=spatial_dims).mean()

        return {
            "radial_spectral_correlation": rad_corr,
            "kx_correlation": compute_corr(kx_spec1, kx_spec2),
            "ky_correlation": compute_corr(ky_spec1, ky_spec2),
            "total_spectral_power_ratio": (p2 / (p1 + self.epsilon)).item(),
        }

    def evaluate_all(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
        dims: Optional[List[int]] = None,
        return_dict: bool = True,
        include_spectral: bool = False,
        spatial_dims: Tuple[int, int] = (-2, -1),
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """Evaluate all metrics for complex predictions vs ground truth"""
        # basic metrics
        z_preds = self.to_complex(preds)
        z_gts = self.to_complex(gts)

        results = {
            "ssim": self.complex_ssim(z_preds, z_gts, dims).mean().item(),
            "mse": self.complex_mse(z_preds, z_gts, dims).mean().item(),
        }

        # spectral analysis
        if include_spectral:
            try:
                z_preds_fft = self._to_spectral_domain(z_preds, spatial_dims)
                z_gts_fft = self._to_spectral_domain(z_gts, spatial_dims)
                avg_dims = list(range(1, z_preds_fft.dim()))

                mag, _ = self.complex_pearson(z_preds_fft, z_gts_fft, dims=avg_dims)
                plv = self.phase_locking_value(z_preds_fft, z_gts_fft, dims=avg_dims)
                kx_ky = self.kx_ky_analysis(z_preds, z_gts, spatial_dims)
                w_plv = self.magnitude_weighted_phase_coherence(
                    z_preds, z_gts, spatial_dims
                )

                results.update(
                    {
                        "spectral_pearson_magnitude": mag.mean().item(),
                        "spectral_phase_locking": plv.mean().item(),
                        "magnitude_weighted_phase_coherence": w_plv.item(),
                        **kx_ky,
                    }
                )
            except Exception as e:
                print(f"warning: spectral analysis failed: {e}")
                results["spectral_analysis_error"] = str(e)

        if return_dict:
            return results
        return torch.stack(
            [torch.tensor(results["ssim"]), torch.tensor(results["mse"])]
        )


def validation_metrics(
    preds: Dict[str, torch.Tensor],
    tgts: Dict[str, torch.Tensor],
    geometry: Dict[str, torch.Tensor],
    loss_wrap: nn.Module,
    eval_integrals: bool = True,
) -> Tuple[
    Dict[str, torch.Tensor],
    Optional[Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]],
]:
    """Compute validation metrics across sequences if applicable"""
    # detect sequence
    is_sequence = False
    if "df" in preds and preds["df"].ndim == 8:
        is_sequence = True
    elif len(preds) > 0 and preds[list(preds.keys())[0]].ndim == 8:
        is_sequence = True

    n_steps = (
        preds["df"].shape[0]
        if is_sequence and "df" in preds
        else (preds[list(preds.keys())[0]].shape[0] if is_sequence else 1)
    )

    metrics_all = defaultdict(list)
    integrated_all = []
    complex_metrics = ComplexMetrics() if "df" in preds else None

    # iterate over steps
    for n in range(n_steps):
        n_pred = {k: v[n] for k, v in preds.items()} if is_sequence else preds
        n_tgt = {k: v[n] for k, v in tgts.items()} if is_sequence else tgts

        for k in set(n_pred) & set(n_tgt):
            assert n_pred[k].shape == n_tgt[k].shape, f"shape mismatch for {k}[{n}]"

        # compute losses
        res = loss_wrap(
            preds=n_pred,
            tgts=n_tgt,
            geometry=geometry,
            compute_integrals=eval_integrals,
        )
        n_int = None
        if len(res) == 4:
            _, n_losses, n_int, _ = res
        elif len(res) == 3:
            _, n_losses, n_int = res
        else:
            _, n_losses = res

        integrated_all.append(n_int)
        for k, v in n_losses.items():
            metrics_all[k].append(
                v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)
            )

        # complex evaluation
        if complex_metrics and "df" in n_pred and "df" in n_tgt:
            for ck, cv in complex_metrics.evaluate_all(
                n_pred["df"], n_tgt["df"]
            ).items():
                metrics_all[f"complex_{ck}"].append(
                    torch.tensor(cv, dtype=torch.float32)
                )

    # finalize metrics
    metrics_all = {k: torch.stack(v) for k, v in metrics_all.items()}
    if not is_sequence:
        metrics_all = {k: v.squeeze(0) for k, v in metrics_all.items()}
        integrated_all = integrated_all[0] if integrated_all else None

    return metrics_all, integrated_all


class BaseEvaluator:
    def __init__(
        self,
        cfg: Any,
        valsets: List[Any],
        valloaders: List[Any],
        loss_wrap: Optional[nn.Module] = None,
    ):
        self.cfg = cfg
        self.valsets = valsets
        self.valloaders = valloaders
        self.loss_wrap = loss_wrap

    def _is_eval_epoch(self, epoch: int) -> bool:
        return epoch % self.cfg.validation.validate_every_n_epochs == 0 or epoch == 1

    def _recombine_zf(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 == 0:
            return torch.cat([x[:, 0::2].sum(1, True), x[:, 1::2].sum(1, True)], dim=1)
        return x

    def _sync_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        n_timesteps_acc: torch.Tensor,
        device: torch.device,
        world_size: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if dist.is_initialized() and world_size > 1:
            cur_ts = n_timesteps_acc.reshape(1, -1).to(device)
            gathered_ts = [torch.zeros_like(cur_ts) for _ in range(world_size)]
            dist.all_gather(gathered_ts, cur_ts)
            n_timesteps_acc = torch.cat(gathered_ts).sum(0).cpu()

            for m in metrics:
                cur_metric = metrics[m].reshape(1, -1).to(device)
                gathered_ms = [torch.zeros_like(cur_metric) for _ in range(world_size)]
                dist.all_gather(gathered_ms, cur_metric)
                metrics[m] = torch.cat(gathered_ms).sum(0).cpu()
        return metrics, n_timesteps_acc

    def _denormalize_batch(
        self,
        data: Dict[str, torch.Tensor],
        idx_data: Dict[str, torch.Tensor],
        denormalize_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Standard denormalization for physics fields."""
        for k in {"df", "phi", "flux"} & set(data):
            data[k] = torch.stack(
                [
                    denormalize_fn(f, **{k: data[k][b]})
                    for b, f in enumerate(idx_data["file_index"].tolist())
                ]
            )
        return data

    def _denormalize_rollout(
        self,
        rollout: Dict[str, torch.Tensor],
        idx_data: Dict[str, torch.Tensor],
        denormalize_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Denormalization for rollout predictions (with time dimension)."""
        for k in rollout:
            rollout[k] = torch.stack(
                [
                    torch.stack(
                        [
                            denormalize_fn(f, **{k: rollout[k][t, b]})
                            for b, f in enumerate(idx_data["file_index"].tolist())
                        ]
                    )
                    for t in range(rollout[k].shape[0])
                ]
            )
        return rollout

    def _accumulate_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        metrics_i: Dict[str, torch.Tensor],
        n_timesteps_acc: torch.Tensor,
        weight: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        for k, v in metrics_i.items():
            if k not in metrics:
                metrics[k] = (
                    torch.zeros_like(v)
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(0.0)
                )

            val = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)
            if metrics[k].ndim == 0:
                metrics[k] += (val if val.ndim == 0 else val.mean()) * weight
            else:
                # Handle sequence metrics (like in GyroSwin)
                cur_len = val.shape[-1]
                tot_len = metrics[k].shape[-1]
                if cur_len < tot_len:
                    padding = torch.zeros(tot_len - cur_len, dtype=val.dtype)
                    metrics[k] += torch.cat([val, padding], dim=-1) * weight
                else:
                    metrics[k] += val[:tot_len] * weight

        n_timesteps_acc += weight
        return metrics, n_timesteps_acc

    def _finalize_logs(
        self,
        log_metric_dict: Dict[str, float],
        metrics: Dict[str, torch.Tensor],
        n_timesteps_acc: torch.Tensor,
        valname: str,
    ) -> Dict[str, float]:
        for m, v in metrics.items():
            if v.sum() != 0.0:
                if v.ndim == 0:
                    log_metric_dict[f"{valname}/{m}"] = (v / n_timesteps_acc).item()
                else:
                    # Sequence metrics
                    avg_v = v / n_timesteps_acc.clamp(min=1)
                    for t in range(v.shape[0]):
                        log_metric_dict[f"{valname}/{m}_x{t + 1}"] = avg_v[t].item()
        return log_metric_dict

    def _save_checkpoint(
        self,
        rank: int,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        val_loss: float,
        loss_val_min: float,
    ) -> float:
        if rank == 0:
            loss_val_min = save_model_and_config(
                model, opt, scheduler, self.cfg, epoch, val_loss, loss_val_min
            )
        return loss_val_min

    def get_iterator(
        self, valloader: Any, val_idx: int, rank: int, desc: Optional[str] = None
    ) -> Any:
        if self.cfg.logging.tqdm and (not dist.is_initialized() or rank == 0):
            if desc is None:
                desc = "validation holdout " + (
                    "trajectories" if val_idx == 0 else "samples"
                )
            return tqdm(valloader, desc=desc)
        return valloader

    @abstractmethod
    def __call__(
        self,
        rank: int,
        world_size: int,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        device: torch.device,
        loss_val_min: float,
        **kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, Any], float]:
        raise NotImplementedError
