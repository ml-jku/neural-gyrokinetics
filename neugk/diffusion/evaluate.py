from typing import List, Callable

from functools import partial
from collections import defaultdict
import re
import warnings

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from tqdm import tqdm
from omegaconf import DictConfig

from neugk.utils import save_model_and_config
from neugk.dataset import CycloneAESample
from neugk.eval import validation_metrics
from neugk.plot_utils import generate_val_plots, avg_flux_confidence
from neugk.losses import LossWrapper


def denormalize_single(preds, idx_data, denormalize_fn):
    # denormalize physics data keys
    physics_keys = {"df", "phi", "flux"}
    for k in preds:
        if k in physics_keys:
            preds[k] = torch.stack(
                [
                    denormalize_fn(f, **{k: preds[k][b]})
                    for b, f in enumerate(idx_data["file_index"].tolist())
                ]
            )
    return preds


@torch.no_grad
def evaluate(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    sample_fn: Callable,
    valsets: List[Dataset],
    valloaders: List[DataLoader],
    opt: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    epoch: int,
    cfg: DictConfig,
    device: torch.device,
    loss_val_min: float,
):
    # Validation loop
    eval_integrals = cfg.validation.eval_integrals
    idx_keys = ["file_index", "timestep_index"]
    log_metric_dict = {}
    val_plots = {}
    val_freq = cfg.validation.validate_every_n_epochs

    # loss wrapper in evaluation mode (no weights / schedule)
    loss_wrap = LossWrapper(
        denormalize_fn=valsets[0].denormalize,
        separate_zf=cfg.dataset.separate_zf,
        real_potens=cfg.dataset.real_potens,
    )
    loss_wrap.eval()

    # Validation loop
    if (epoch % val_freq) == 0 or epoch == 1:
        for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):

            valname = "val_traj" if val_idx == 0 else "val_samples"
            metrics = {}
            for key in loss_wrap.all_losses:
                metrics[key] = torch.tensor(0.0)
            n_timesteps_acc = torch.tensor(0.0)
            if cfg.logging.tqdm and (not dist.is_initialized() or rank == 0):
                valloader = tqdm(
                    valloader,
                    desc=(
                        "Validation holdout trajectories"
                        if val_idx == 0
                        else "Validation holdout samples"
                    ),
                )

            # lists to collect flux across all batches for the two TODOs
            all_preds_flux = []
            all_tgts_flux = []

            for idx, sample in enumerate(valloader):
                sample: CycloneAESample
                # for integral evaluation, etc
                # NOTE: validation target are not latent
                tgts = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in ["df", "phi", "flux", "avg_flux"]
                    if getattr(sample, k) is not None
                }
                condition = sample.conditioning.to(device)
                geometry = sample.geometry
                idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

                # get the rolled out validation trajectories
                preds = sample_fn(condition)

                # NOTE denormalize always true for integrals if denormalize:
                preds = denormalize_single(
                    preds, idx_data, partial(valset.denormalize, condition=condition)
                )
                tgts = denormalize_single(tgts, idx_data, valset.denormalize)

                tgts = {k: v.cpu() for k, v in tgts.items()}
                preds = {k: v.cpu() for k, v in preds.items()}

                if cfg.dataset.separate_zf:

                    def _recombine_zf(x):
                        x = torch.cat(
                            [x[:, 0::2].sum(1, True), x[:, 1::2].sum(1, True)], dim=1
                        )
                        return x

                    # apply recombine_zf to preds and tgts
                    for k in preds:
                        if preds[k].dim() > 2 and preds[k].shape[1] % 2 == 0:
                            preds[k] = _recombine_zf(preds[k])
                    for k in tgts:
                        if tgts[k].dim() > 2 and tgts[k].shape[1] % 2 == 0:
                            tgts[k] = _recombine_zf(tgts[k])

                metrics_i, integrated_i = validation_metrics(
                    tgts=tgts,
                    preds=preds,
                    geometry=geometry,
                    loss_wrap=loss_wrap,
                    eval_integrals=eval_integrals,
                )

                # add integrated quantities to rollout for comparison
                preds["phi_int"] = integrated_i["phi"]
                preds["flux_int"] = integrated_i["eflux"]

                # collect avg_flux and file_index for plotting and rmse
                if "flux_int" in preds and "avg_flux" in tgts:
                    all_preds_flux.append(
                        {
                            "val": preds["flux_int"].cpu(),
                            "file_index": idx_data["file_index"].cpu(),
                        }
                    )
                    all_tgts_flux.append(
                        {
                            "val": tgts["avg_flux"].cpu(),
                            "file_index": idx_data["file_index"].cpu(),
                        }
                    )

                for key in metrics_i.keys():
                    if key not in metrics:
                        metrics[key] = torch.tensor(0.0)
                    if metrics_i[key].dim() == 0:
                        metrics[key] += metrics_i[key]
                    elif metrics_i[key].dim() == 1:
                        metrics[key] += metrics_i[key].mean()
                n_timesteps_acc += 1

                # temporary dicts for plotting
                # TODO(gg) for now limit number of plots
                if len(val_plots) == 0:
                    t_idx = idx_data["timestep_index"].tolist()
                    batch_idx = torch.randint(0, len(t_idx), (1,)).item()
                    # TODO(diff) batch_idx only used for df/phi, flux needs the batch
                    #            with more trajectories need to ensure sample within one
                    # use df, phi_int, flux_int for plotting
                    preds_plots = {
                        "df": preds["df"] if "df" in preds else None,
                        "phi": preds["phi_int"] if "phi_int" in preds else None,
                        "flux": preds["flux_int"] if "flux_int" in preds else None,
                    }

                    preds_plots = {
                        k: (
                            preds_plots[k][batch_idx]
                            if "flux" not in k
                            else preds_plots[k]
                        )
                        for k in preds_plots
                    }
                    tgts = {k: tgts[k][batch_idx] for k in tgts}
                    if val_idx == 0:
                        # holdout trajectories valset
                        plots = generate_val_plots(
                            rollout=preds_plots,
                            gt=tgts,
                            ts=sample.timestep,
                            phase="Random draw",
                        )
                        val_plots.update(plots)
                    else:
                        # holdout samples valset
                        if idx == 0:
                            plots = generate_val_plots(
                                rollout=preds_plots,
                                gt=tgts,
                                timestep=sample.timestep,
                                phase="Holdout samples",
                            )
                            val_plots.update(plots)

            if dist.is_initialized():
                # for phase in metrics.keys():
                cur_ts = n_timesteps_acc.reshape(1, -1).to(device)
                gathered_ts = [
                    torch.zeros_like(cur_ts, dtype=cur_ts.dtype, device=cur_ts.device)
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_ts, cur_ts)
                n_timesteps_acc = torch.cat(gathered_ts).sum(0).cpu()

                for m in metrics.keys():
                    cur_metric = metrics[m].reshape(1, -1).to(device)
                    gathered_ms = [
                        torch.zeros_like(
                            cur_metric,
                            dtype=cur_metric.dtype,
                            device=cur_metric.device,
                        )
                        for _ in range(world_size)
                    ]
                    dist.all_gather(gathered_ms, cur_metric)
                    gathered_ms = torch.cat(gathered_ms)
                    metrics[m] = gathered_ms.sum(0).cpu()

            for m in metrics.keys():
                if metrics[m].sum() != 0.0:
                    metrics[m] = metrics[m] / n_timesteps_acc
                    log_metric_dict[f"{valname}/{m}"] = metrics[m].item()

            # fluxes averaged per trajetory
            if len(all_preds_flux) > 0 and (not dist.is_initialized() or rank == 0):
                fpreds = torch.cat([b["val"].flatten() for b in all_preds_flux])
                ftgts = torch.cat([b["val"].flatten() for b in all_tgts_flux])
                fidxs = torch.cat([b["file_index"].flatten() for b in all_preds_flux])

                # group by trajectory to calculate actual avg_flux
                grouped_preds = defaultdict(list)
                grouped_tgts = {}

                for p, t, i in zip(fpreds, ftgts, fidxs):
                    idx = re.search(r"iteration_\d+", valset.files[i.item()]).group()
                    grouped_preds[idx].append(p.item())
                    grouped_tgts[idx] = t.item()

                traj_ids = sorted(list(grouped_preds.keys()))
                pred_means = np.array([np.mean(grouped_preds[i]) for i in traj_ids])
                pred_stds = np.array([np.std(grouped_preds[i]) for i in traj_ids])
                tgt_vals = np.array([grouped_tgts[i] for i in traj_ids])

                # compute rmse on the trajectory averages
                avg_flux_rmse = np.sqrt(((pred_means - tgt_vals) ** 2).mean())
                log_metric_dict[f"{valname}/avg_flux_rmse"] = avg_flux_rmse.item()

                val_plots["avg_flux_UQ"] = avg_flux_confidence(
                    pred_means, pred_stds, tgt_vals, traj_ids
                )

        if not rank:
            val_loss = log_metric_dict["val_traj/avg_flux_rmse"]
            loss_val_min = save_model_and_config(
                model,
                optimizer=opt,
                scheduler=lr_scheduler,
                cfg=cfg,
                epoch=epoch,
                # TODO decide target metric
                val_loss=val_loss,
                loss_val_min=loss_val_min,
            )
        else:
            warnings.warn(f"checkpoints will not be stored for rank {rank}")

    return log_metric_dict, val_plots, loss_val_min
