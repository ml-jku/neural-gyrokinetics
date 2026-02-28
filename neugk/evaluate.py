"""Evaluation metrics for complex-valued fields and physical quantities."""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import defaultdict
from abc import abstractmethod

import torch.distributed as dist
from tqdm import tqdm

import torch
from torch import nn

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
            # sum real and imaginary parts if split (e.g. bands)
            return torch.complex(tensor[:, 0::2].sum(dim=1), tensor[:, 1::2].sum(dim=1))
        raise ValueError(f"expected even number of channels, got {channels}")

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

        mu1_k = z1.mean(dim=dims, keepdim=True)
        mu2_k = z2.mean(dim=dims, keepdim=True)

        var1 = ((z1 - mu1_k).abs() ** 2).mean(dim=dims)
        var2 = ((z2 - mu2_k).abs() ** 2).mean(dim=dims)
        cov12 = ((z1 - mu1_k) * (z2 - mu2_k).conj()).mean(dim=dims)

        data_range = max(z1.abs().max().item(), z2.abs().max().item())
        c1 = (c1 * data_range) ** 2
        c2 = (c2 * data_range) ** 2

        num = (2 * mu1_k.abs() * mu2_k.abs() + c1) * (2 * cov12.abs() + c2)
        den = (mu1_k.abs() ** 2 + mu2_k.abs() ** 2 + c1) * (var1 + var2 + c2)
        return (num / den).mean()

    def evaluate_all(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate basic metrics for complex predictions vs ground truth"""
        z_preds = self.to_complex(preds)
        z_gts = self.to_complex(gts)

        return {
            "ssim": self.complex_ssim(z_preds, z_gts).item(),
            "mse": torch.mean((z_preds - z_gts).abs() ** 2).item(),
        }


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
        dataset: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Standard denormalization for physics fields."""
        # shallow copy to avoid modifying input dict in-place
        data = data.copy()

        # determine expected batch size from indices
        batch_size = len(idx_data["file_index"])

        # try vectorization for dataset-wide normalization
        if (
            dataset is not None
            and getattr(dataset, "normalization_scope", None) == "dataset"
        ):
            for k in {"df", "phi", "flux"} & set(data):
                if data[k] is not None:
                    data[k] = denormalize_fn(0, **{k: data[k]}, **kwargs)
                    assert (
                        data[k].shape[0] == batch_size
                    ), f"Batch size mismatch after vectorized denorm for {k}"
            return data

        for k in {"df", "phi", "flux"} & set(data):
            if data[k] is not None:
                samples = []
                for b, f in enumerate(idx_data["file_index"].tolist()):
                    # extract individual kwargs if they are batched
                    curr_kwargs = {
                        kk: (
                            vv[b]
                            if (torch.is_tensor(vv) and vv.shape[0] == batch_size)
                            else vv
                        )
                        for kk, vv in kwargs.items()
                    }
                    samples.append(denormalize_fn(f, **{k: data[k][b]}, **curr_kwargs))

                data[k] = torch.stack(samples)
                assert (
                    data[k].shape[0] == batch_size
                ), f"Batch size mismatch after loop denorm for {k}"
        return data

    def _denormalize_rollout(
        self,
        rollout: Dict[str, torch.Tensor],
        idx_data: Dict[str, torch.Tensor],
        denormalize_fn: Callable,
        dataset: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Denormalization for rollout predictions (with time dimension)."""
        if not rollout:
            return rollout

        # determine dimensions from any present field that matches standard physics fields
        fields_to_denorm = {"df", "phi", "flux"} & set(rollout)
        if not fields_to_denorm:
            return rollout.copy()

        any_k = next(iter(fields_to_denorm))
        any_v = rollout[any_k]
        T, B = any_v.shape[:2]

        # reshape time into batch
        flattened_rollout = {
            k: v.flatten(0, 1) if k in fields_to_denorm else v
            for k, v in rollout.items()
        }

        # match indices to flattened rollout
        flattened_idx_data = {"file_index": idx_data["file_index"].repeat(T)}

        # full batch denormalization
        denorm_data = self._denormalize_batch(
            flattened_rollout, flattened_idx_data, denormalize_fn, dataset
        )

        # Reshape back to (T, B, ...)
        res = {}
        for k, v in denorm_data.items():
            if k in fields_to_denorm:
                res[k] = v.view(T, B, *v.shape[1:])
            else:
                res[k] = v
        return res

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
                    if v.shape[0] > 1:
                        for t in range(v.shape[0]):
                            log_metric_dict[f"{valname}/{m}_x{t + 1}"] = avg_v[t].item()
                    else:
                        log_metric_dict[f"{valname}/{m}"] = avg_v[0].item()
        return log_metric_dict

    def _get_val_loss(self, log_metric_dict: Dict[str, float], default_metric: str = "df") -> float:
        m_name = self.cfg.validation.get("model_selection_metric", default_metric)
        val_loss = log_metric_dict.get(f"val_traj/{m_name}")
        
        if val_loss is None:
            # handle multi-step keys by averaging across sequence
            relevant_vals = [
                v for k, v in log_metric_dict.items()
                if k.startswith(f"val_traj/{m_name}_x")
            ]
            val_loss = sum(relevant_vals) / len(relevant_vals) if relevant_vals else 0.0
        return val_loss

    def _save_checkpoint(
        self,
        rank: int,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        log_metric_dict: Dict[str, float],
        loss_val_min: float,
        default_metric: str = "df",
    ) -> float:
        val_loss = self._get_val_loss(log_metric_dict, default_metric=default_metric)
        
        if rank == 0:
            loss_val_min = save_model_and_config(
                model, opt, scheduler, self.cfg, epoch, val_loss, loss_val_min
            )

        if dist.is_initialized():
            # synchronize best loss across all ranks
            lv_tensor = torch.tensor(
                loss_val_min, device=next(model.parameters()).device
            )
            dist.broadcast(lv_tensor, src=0)
            loss_val_min = lv_tensor.item()

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
