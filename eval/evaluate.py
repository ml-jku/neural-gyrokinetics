from typing import List, Dict
import warnings
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from tqdm import tqdm

from utils import split_batch_into_phases, save_model_and_config
from dataset import CycloneSample
from eval import (
    validation_metrics,
    generate_val_plots,
    get_rollout_fn,
)


def denormalize_rollout(rollout, gts, idx_data, denormalize_fn):
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
        gts[k] = torch.stack(
            [
                denormalize_fn(f, **{k: gts[k][b]})
                for b, f in enumerate(idx_data["file_index"].tolist())
            ]
        )
    return rollout, gts


@torch.no_grad
def evaluate(
    rank: int,
    world_size: int,
    model: nn.Module,
    loss_wrap: nn.Module,
    valsets: List[Dataset],
    valloaders: List[DataLoader],
    opt: torch.optim.Optimizer,
    epoch: int,
    cfg: Dict,
    device: torch.device,
    loss_val_min: float,
):
    # Validation loop
    n_eval_steps = cfg.validation.n_eval_steps
    bundle_seq_length = cfg.model.bundle_seq_length
    tot_eval_steps = n_eval_steps * bundle_seq_length
    predict_delta = cfg.training.predict_delta
    eval_integrals = cfg.validation.eval_integrals
    use_tqdm = cfg.logging.tqdm
    use_amp = cfg.use_amp
    input_fields = cfg.dataset.input_fields
    output_fields = list(cfg.model.loss_weights.keys())
    conditioning = cfg.model.conditioning
    idx_keys = ["file_index", "timestep_index"]
    log_metric_dict = {}
    val_plots = {}
    val_freq = cfg.validation.validate_every_n_epochs

    # Validation loop
    if (epoch % val_freq) == 0 or epoch == 1:
        model.eval()
        loss_wrap.eval().cpu()  # eval mode (all metrics)
        for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):

            rollout_fn = get_rollout_fn(
                n_steps=n_eval_steps,
                bundle_steps=cfg.model.bundle_seq_length,
                dataset=valset,
                predict_delta=predict_delta,
                device=str(device),
                use_amp=use_amp,
            )

            valname = "val_traj" if val_idx == 0 else "val_samples"
            metrics = defaultdict(dict)
            for phase in ["linear", "saturated"]:
                for key in loss_wrap.all_losses:
                    metrics[phase][key] = torch.zeros([tot_eval_steps])

            n_timesteps_acc = {
                "linear": torch.zeros([tot_eval_steps]),
                "saturated": torch.zeros([tot_eval_steps]),
            }
            if use_tqdm or (dist.is_initialized() and not rank):
                valloader = tqdm(
                    valloader,
                    desc=(
                        "Validation holdout trajectories"
                        if val_idx == 0
                        else "Validation holdout samples"
                    ),
                )

            for idx, sample in enumerate(valloader):
                sample: CycloneSample
                inputs = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in input_fields
                }
                gts = {
                    k: getattr(sample, f"y_{k}").to(device, non_blocking=True)
                    for k in output_fields
                }
                conds = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in conditioning
                }
                idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

                # TODO: dont hardcode this
                phase_change = 24 if cfg.dataset.offset == 0 else 0
                inputs_list, gts_list, conds_list, idx_data_list, phase_list = (
                    split_batch_into_phases(
                        phase_change,
                        inputs,
                        gts,
                        conds,
                        idx_data,
                    )
                )

                # Iterate over the splits
                for i in range(len(inputs_list)):
                    inputs = inputs_list[i]
                    gts = gts_list[i]
                    conds = conds_list[i]
                    idx_data = idx_data_list[i]
                    phase = phase_list[i]

                    # get the rolled out validation trajectories
                    rollout = rollout_fn(model, inputs, idx_data, conds)

                    # denormalize rollout and target for evaluation / plots
                    # NOTE denormalize always true for integrals if denormalize:
                    rollout, gts = denormalize_rollout(
                        rollout, gts, idx_data, valset.denormalize
                    )

                    # TODO: smarter (i.e. use timeindex when we output a dataclass from the dataset)
                    metrics_i = validation_metrics(
                        rollout,
                        idx_data,
                        bundle_seq_length,
                        valset,
                        loss_wrap=loss_wrap,
                        eval_integrals=eval_integrals,
                    )

                    for key in metrics_i.keys():
                        if metrics_i[key].shape[-1] < tot_eval_steps:
                            # end of dataset, need to pad the tensor
                            diff = tot_eval_steps - metrics_i[key].shape[-1]
                            metrics_i[key] = torch.cat(
                                [
                                    metrics_i[key],
                                    torch.zeros(diff, dtype=metrics_i[key].dtype),
                                ],
                                dim=-1,
                            )
                            metrics[phase][key] += metrics_i[key]
                            validated_steps = (
                                torch.arange(1, tot_eval_steps + 1)
                                <= metrics_i[key].shape[-1]
                            ).to(dtype=metrics_i[key].dtype)
                        else:
                            metrics[phase][key] += metrics_i[key]
                            validated_steps = torch.ones([tot_eval_steps])
                    n_timesteps_acc[phase] += validated_steps

                    if val_idx == 0:
                        # holdout trajectories valset
                        if idx in [0, 20]:
                            plots = generate_val_plots(
                                rollout=rollout,
                                gt=gts,
                                ts=conds["timestep"],
                                phase=(
                                    "Saturated Phase"
                                    if idx == 20 or cfg.dataset.offset > 0
                                    else "Linear Phase"
                                ),
                            )
                            val_plots.update(plots)
                    else:
                        # holdout samples valset
                        if idx == 0:
                            plots = generate_val_plots(
                                rollout=rollout,
                                gt=gts,
                                ts=conds["timestep"],
                                phase="Holdout samples",
                            )
                            val_plots.update(plots)

            if dist.is_initialized():
                for phase in metrics.keys():
                    cur_ts = n_timesteps_acc[phase].to(device)
                    gathered_ts = [
                        torch.zeros_like(
                            cur_ts, dtype=cur_ts.dtype, device=cur_ts.device
                        )
                        for _ in range(world_size)
                    ]
                    dist.all_gather(gathered_ts, cur_ts)
                    n_timesteps_acc[phase] = torch.cat(gathered_ts).sum(0).cpu()

                    for m in metrics[phase].keys():
                        cur_metric = metrics[phase][m].to(device)
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
                        if len(gathered_ms.shape) == 2:
                            # account for different metrics axis if not present
                            gathered_ms = gathered_ms.unsqueeze(1)
                        metrics[phase][m] = gathered_ms.sum(0).cpu()

                # TODO: for some reason deadlocks
                # gathered_plots = [object() for _ in range(world_size)]
                # dist.all_gather_object(gathered_plots, val_plots)
                # val_plots = {}
                # for d in gathered_plots:
                #     val_plots.update(d)

            for ph in metrics.keys():
                for m in metrics[ph].keys():
                    if metrics[ph][m].sum() != 0.0:
                        metrics[ph][m] = metrics[ph][m] / n_timesteps_acc[ph]
                        vals = metrics[ph][m]
                        for t in range(tot_eval_steps):
                            log_metric_dict[f"{valname}/{m}_{ph}_x{t + 1}"] = vals[t]

            if val_idx == 0:
                # trajectoy validation
                n_timesteps_acc_model_saving = n_timesteps_acc

        if cfg.ckpt_path is not None and not rank:
            mse_sat = log_metric_dict["val_traj/df_saturated_x1"]
            mse_lin = log_metric_dict["val_traj/df_linear_x1"]
            sat_ts = n_timesteps_acc_model_saving["saturated"]
            lin_ts = n_timesteps_acc_model_saving["linear"]
            if not cfg.dataset.offset:
                val_loss = (mse_sat * sat_ts + mse_lin * lin_ts) / (sat_ts + lin_ts)
            else:
                # skipping linear phase
                val_loss = (mse_sat * sat_ts) / sat_ts
            # Save model if validation loss on trajectories improves
            loss_val_min = save_model_and_config(
                model,
                optimizer=opt,
                cfg=cfg,
                epoch=epoch,
                # TODO decide target metric
                val_loss=val_loss.mean(),
                loss_val_min=loss_val_min,
            )
        else:
            warnings.warn("`cfg.ckpt_path` is not set: checkpoints will not be stored")

    return log_metric_dict, val_plots, loss_val_min
