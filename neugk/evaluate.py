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
        self.model_selection_metric: str = cfg.validation.get(
            "model_selection_metric", "df"
        )

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

    def _get_val_loss(
        self,
        log_metric_dict: Dict[str, float],
        default_metric: Optional[str] = None,
    ) -> float:
        m_name = default_metric or self.model_selection_metric

        # try direct lookup first (supports fully-qualified keys like
        # "val_traj/probe_flux_val_rmse" as well as short names like "df")
        val_loss = log_metric_dict.get(m_name)

        # fall back to val_traj/<metric> prefix
        if val_loss is None:
            val_loss = log_metric_dict.get(f"val_traj/{m_name}")

        if val_loss is None:
            # handle multi-step keys by averaging across sequence
            relevant_vals = [
                v
                for k, v in log_metric_dict.items()
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
        default_metric: Optional[str] = None,
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

    @torch.no_grad()
    def collect_latents(
        self,
        rank: int,
        dataloader: torch.utils.data.DataLoader,
        extraction_fn: Callable,
        device: torch.device,
        desc: Optional[str] = "collect latents",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect model latents and target fluxes.

        Args:
            rank: Process rank.
            dataloader: Data to process.
            extraction_fn: Function that takes (sample, device) and returns (latent, flux).
                           Latent should be pooled if necessary.
            device: Computing device.
            desc: Progress bar description.
        """
        latents: List[torch.Tensor] = []
        fluxes: List[torch.Tensor] = []

        # setup iterator
        use_tqdm = (not dist.is_initialized() or rank == 0) and desc
        iterator = tqdm(dataloader, desc=desc) if use_tqdm else dataloader

        for sample in iterator:
            z, flux = extraction_fn(sample, device)

            # global average pool if necessary
            if z.ndim > 2:
                # handle both (B, C, ...) and (B, ..., C)
                if z.shape[1] < z.shape[-1]:
                    # likely (B, C, ...)
                    zpool = z.view(z.shape[0], z.shape[1], -1).mean(-1)
                else:
                    # likely (B, ..., C)
                    zpool = z.view(z.shape[0], -1, z.shape[-1]).mean(1)
            else:
                zpool = z

            latents.append(zpool.cpu())
            fluxes.append(flux.view(flux.shape[0], -1).cpu())

        return torch.cat(latents, 0), torch.cat(fluxes, 0)

    def run_probing_evaluation(
        self,
        rank: int,
        trainloader: torch.utils.data.DataLoader,
        extraction_fn: Callable,
        device: torch.device,
        epoch: int,
        log_metric_dict: Dict[str, float],
        val_plots: Dict[str, Any],
        probe_cfg: Optional[Dict[str, Any]] = None,
        probe_targets: Optional[List[str]] = None,
        val_extraction_fns: Optional[List[Callable]] = None,
        dataset_for_stats=None,
    ):
        """Run linear probing and generate t-SNE plots.

        When *probe_cfg* is provided the method performs multi-target probing
        with per-target denormalization and RMSE.  Otherwise the original
        single-target behaviour is used (backward-compatible).

        Args:
            extraction_fn: ``(sample, device) -> (latent, target)`` for training.
            val_extraction_fns: Optional per-valset extraction functions.  When
                ``None``, *extraction_fn* is reused for every validation loader.
            probe_cfg: Dict with ``"sizes"`` (list of ints, one per target) and
                optionally ``"targets"`` (list of target names).
            probe_targets: List of target names (used for metric keys and
                denormalization).
            dataset_for_stats: Dataset whose ``.stats`` are used to build
                denormalization tensors.  Only required when *probe_cfg* is set.
        """
        import numpy as np
        from neugk.plot_utils import plot_latent_tsne

        # 1. compute probe weights on trainset
        x_train, y_train = self.collect_latents(
            rank, trainloader, extraction_fn, device, desc="probe trainset"
        )
        x_train_b = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)

        if probe_cfg is not None:
            w = torch.linalg.pinv(x_train_b) @ y_train
        else:
            res = torch.linalg.lstsq(x_train_b, y_train, driver="gels")
            w = res.solution

        y_train_pred = x_train_b @ w

        # build denormalization tensors for multi-target mode
        norm_mean = norm_std = None
        if probe_cfg is not None and dataset_for_stats is not None:
            cat_mean, cat_std = [], []
            for tgt_name in probe_targets:
                stats = dataset_for_stats.stats.get(tgt_name, {})
                mean = stats["full"]["mean"] if stats else 0.0
                std = stats["full"]["std"] if stats else 1.0
                cat_mean.append(np.atleast_1d(mean))
                cat_std.append(np.atleast_1d(std))
            norm_mean = torch.as_tensor(np.concatenate(cat_mean), device=y_train.device)
            norm_std = torch.as_tensor(np.concatenate(cat_std), device=y_train.device)

        if probe_cfg is None:
            # single-target: report overall train RMSE
            train_rmse = torch.sqrt(torch.mean((y_train_pred - y_train) ** 2))
            log_metric_dict["val_traj/probe_train_rmse"] = train_rmse.item()

        # 2. evaluate on validation sets
        for val_idx, valloader in enumerate(self.valloaders):
            valname = "val_traj" if val_idx == 0 else "val_samples"
            val_fn = (
                val_extraction_fns[val_idx]
                if val_extraction_fns is not None
                else extraction_fn
            )
            x_val, y_val = self.collect_latents(
                rank, valloader, val_fn, device, desc=None
            )
            x_val_b = torch.cat([x_val, torch.ones(x_val.shape[0], 1)], dim=1)
            y_val_pred = x_val_b @ w

            if probe_cfg is not None and norm_mean is not None:
                # multi-target: denormalize and per-target RMSE
                y_val_pred = y_val_pred * norm_std + norm_mean
                y_val = y_val * norm_std + norm_mean
                sizes = [int(s) for s in probe_cfg["sizes"]]
                pred_splits = torch.split(y_val_pred, sizes, dim=1)
                target_splits = torch.split(y_val, sizes, dim=1)
                for name, pred, target in zip(
                    probe_targets, pred_splits, target_splits
                ):
                    if name in ["fluxspec", "kyspec"]:
                        pred = torch.expm1(pred)
                        target = torch.expm1(target)
                    val_rmse = torch.sqrt(torch.mean((pred - target) ** 2))
                    log_metric_dict[f"{valname}/probe_{name}_val_rmse"] = (
                        val_rmse.item()
                    )
            else:
                # single-target: overall RMSE
                val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val) ** 2))
                log_metric_dict[f"{valname}/probe_val_rmse"] = val_rmse.item()

            # t-SNE plots for first validation set
            if val_idx == 0:
                tsne_targets = (
                    list(probe_cfg.get("tsne_targets", ["flux"]))
                    if probe_cfg is not None
                    else None
                )
                if tsne_targets is not None and probe_targets is not None:
                    sizes = [int(s) for s in probe_cfg["sizes"]]
                    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
                    for tsne_tgt in tsne_targets:
                        if tsne_tgt not in probe_targets:
                            continue
                        idx = probe_targets.index(tsne_tgt)
                        col_start = offsets[idx]
                        y_color = y_val[:, col_start : col_start + sizes[idx]].mean(
                            dim=1
                        )
                        val_plots[f"latent_tsne_{tsne_tgt}"] = plot_latent_tsne(
                            x_val,
                            y_color,
                            title=f"t-SNE colored by {tsne_tgt} (Epoch {epoch})",
                        )
                else:
                    y_color = y_val[:, 0] if y_val.ndim > 1 else y_val
                    val_plots["latent_tsne"] = plot_latent_tsne(
                        x_val, y_color, title=f"Latent Space t-SNE (Epoch {epoch})"
                    )

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
