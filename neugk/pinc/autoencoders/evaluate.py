import warnings
from functools import partial
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neugk.dataset import CycloneAESample
from neugk.eval import validation_metrics
from neugk.plot_utils import generate_val_plots
from neugk.utils import save_model_and_config


def denormalize_single(
    preds: Dict[str, torch.Tensor], idx_data: Dict[str, torch.Tensor], denormalize_fn
) -> Dict[str, torch.Tensor]:
    """Denormalize physics data keys in predictions"""
    for k in {"df", "phi", "flux"} & set(preds):
        preds[k] = torch.stack(
            [
                denormalize_fn(f, **{k: preds[k][b]})
                for b, f in enumerate(idx_data["file_index"].tolist())
            ]
        )
    return preds


def autoencoder_eval(
    xs: Dict[str, torch.Tensor], condition: Dict[str, torch.Tensor], ae: nn.Module
) -> Dict[str, torch.Tensor]:
    """Forward pass for autoencoder using dataframe"""
    return ae(xs["df"], condition=condition)


@torch.no_grad()
def evaluate(
    rank: int,
    world_size: int,
    model: nn.Module,
    loss_wrap: nn.Module,
    valsets: List[Dataset],
    valloaders: List[DataLoader],
    opt: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    epoch: int,
    cfg: DictConfig,
    device: torch.device,
    loss_val_min: float,
):
    """Evaluate autoencoder model on validation datasets"""
    eval_integrals = cfg.validation.eval_integrals
    use_tqdm = cfg.logging.tqdm
    input_fields = cfg.dataset.input_fields
    idx_keys = ["file_index", "timestep_index"]
    log_metric_dict, val_plots = {}, {}

    if epoch % cfg.validation.validate_every_n_epochs != 0 and epoch != 1:
        return log_metric_dict, val_plots, loss_val_min

    model.eval()
    loss_wrap.eval().cpu()

    for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):
        eval_fn = partial(autoencoder_eval, ae=model)
        valname = "val_traj" if val_idx == 0 else "val_samples"
        metrics = {key: torch.tensor(0.0) for key in loss_wrap.all_losses}
        n_timesteps_acc = torch.tensor(0.0)

        if use_tqdm and (not dist.is_initialized() or rank == 0):
            valloader = tqdm(
                valloader,
                desc=f"validation holdout {'trajectories' if val_idx == 0 else 'samples'}",
            )

        for idx, sample in enumerate(valloader):
            sample: CycloneAESample
            xs = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in input_fields
                if getattr(sample, k) is not None
            }
            tgts = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in ["df", "phi", "flux", "avg_flux"]
                if getattr(sample, k) is not None
            }

            condition = (
                sample.conditioning.to(device)
                if sample.conditioning is not None
                else None
            )
            idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

            preds = eval_fn(xs, condition)
            preds = denormalize_single(preds, idx_data, valset.denormalize)
            tgts = denormalize_single(tgts, idx_data, valset.denormalize)

            tgts = {k: v.cpu() for k, v in tgts.items()}
            preds = {k: v.cpu() for k, v in preds.items()}

            if getattr(cfg.dataset, "separate_zf", False):

                def _recombine_zf(x):
                    return torch.cat(
                        [x[:, 0::2].sum(1, True), x[:, 1::2].sum(1, True)], dim=1
                    )

                for d in [preds, tgts]:
                    for k in d:
                        if d[k].dim() > 2 and d[k].shape[1] % 2 == 0:
                            d[k] = _recombine_zf(d[k])

            metrics_i, integrated_i = validation_metrics(
                tgts=tgts,
                preds=preds,
                geometry=sample.geometry,
                loss_wrap=loss_wrap,
                eval_integrals=eval_integrals,
            )

            preds["phi_int"], preds["flux_int"] = integrated_i.get(
                "phi"
            ), integrated_i.get("eflux")

            for k, v in metrics_i.items():
                metrics[k] = metrics.get(k, torch.tensor(0.0)) + (
                    v if v.dim() == 0 else v.mean()
                )
            n_timesteps_acc += 1

            if not val_plots:
                batch_idx = torch.randint(
                    0, len(idx_data["timestep_index"].tolist()), (1,)
                ).item()
                preds_plots = {k: preds.get(k) for k in ["df", "phi_int", "flux_int"]}
                preds_plots = {
                    k: (v[batch_idx] if v is not None and "flux" not in k else v)
                    for k, v in preds_plots.items()
                }
                plot_tgts = {k: tgts[k][batch_idx] for k in tgts}

                if val_idx == 0:
                    val_plots.update(
                        generate_val_plots(
                            rollout=preds_plots,
                            gt=plot_tgts,
                            ts=sample.timestep,
                            phase="random draw",
                        )
                    )
                elif idx == 0:
                    val_plots.update(
                        generate_val_plots(
                            rollout=preds_plots,
                            gt=plot_tgts,
                            timestep=sample.timestep,
                            phase="holdout samples",
                        )
                    )

        if dist.is_initialized():
            cur_ts = n_timesteps_acc.reshape(1, -1).to(device)
            gathered_ts = [torch.zeros_like(cur_ts) for _ in range(world_size)]
            dist.all_gather(gathered_ts, cur_ts)
            n_timesteps_acc = torch.cat(gathered_ts).sum(0).cpu()

            for m in metrics:
                cur_metric = metrics[m].reshape(1, -1).to(device)
                gathered_ms = [torch.zeros_like(cur_metric) for _ in range(world_size)]
                dist.all_gather(gathered_ms, cur_metric)
                metrics[m] = torch.cat(gathered_ms).sum(0).cpu()

        for m, v in metrics.items():
            if v.sum() != 0.0:
                log_metric_dict[f"{valname}/{m}"] = (v / n_timesteps_acc).item()

    if rank == 0:
        val_loss = metrics.get("df", torch.tensor(0.0)).mean()
        loss_val_min = save_model_and_config(
            model,
            optimizer=opt,
            scheduler=lr_scheduler,
            cfg=cfg,
            epoch=epoch,
            val_loss=val_loss,
            loss_val_min=loss_val_min,
        )
    else:
        warnings.warn(f"checkpoints will not be stored for rank {rank}")

    return log_metric_dict, val_plots, loss_val_min
