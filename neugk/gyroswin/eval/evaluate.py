from typing import List, Dict
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from tqdm import tqdm

from neugk.utils import save_model_and_config
from neugk.dataset import CycloneSample
from neugk.gyroswin.eval import validation_metrics, generate_val_plots, get_rollout_fn


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
    scheduler: torch.optim.lr_scheduler,
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
    use_amp = cfg.amp.enable
    use_bf16 = use_amp and cfg.amp.bfloat and torch.cuda.is_bf16_supported()
    input_fields = cfg.dataset.input_fields
    output_fields = [
        k
        for k in cfg.model.loss_weights.keys()
        if cfg.model.loss_weights[k] > 0.0 or cfg.model.loss_scheduler[k]
    ]
    if cfg.model.name in ["pointnet", "transolver", "transformer"]:
        input_fields.append("position")
    if set(output_fields) != set(["df", "phi", "flux"]):
        eval_integrals = False
    if eval_integrals:
        output_fields = ["df", "phi", "flux"]  # all fields for integral evaluation
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
                use_bf16=use_bf16,
            )

            valname = "val_traj" if val_idx == 0 else "val_samples"
            metrics = {}
            for key in loss_wrap.all_losses:
                metrics[key] = torch.zeros([tot_eval_steps])
            n_timesteps_acc = torch.zeros([tot_eval_steps])
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

                # get the rolled out validation trajectories
                rollout = rollout_fn(model, inputs, idx_data, conds)

                # denormalize rollout and target for evaluation / plots
                # NOTE denormalize always true for integrals if denormalize:
                rollout, gts = denormalize_rollout(
                    rollout, gts, idx_data, valset.denormalize
                )

                metrics_i, integrated_i = validation_metrics(
                    rollout,
                    idx_data,
                    bundle_seq_length,
                    dataset=valset,
                    output_fields=output_fields,
                    loss_wrap=loss_wrap,
                    eval_integrals=eval_integrals,
                )
                # add integrated potentials to rollout for comparison
                if integrated_i[0] is not None:
                    rollout["phi_int"] = torch.stack(
                        [integrated_i[t]["phi"] for t in range(len(integrated_i))]
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
                        metrics[key] += metrics_i[key]
                        validated_steps = (
                            torch.arange(1, tot_eval_steps + 1)
                            <= metrics_i[key].shape[-1]
                        ).to(dtype=metrics_i[key].dtype)
                    else:
                        metrics[key] += metrics_i[key]
                        validated_steps = torch.ones([tot_eval_steps])
                n_timesteps_acc += validated_steps
                if not "position" in inputs:
                    # no plots for field-like baselines
                    if val_idx == 0:
                        # holdout trajectories valset
                        t_idx = idx_data["timestep_index"].tolist()
                        batch_idx = torch.randint(0, len(t_idx), (1,)).item()
                        rollout = {k: rollout[k][:, batch_idx].cpu() for k in rollout}
                        gts = {k: gts[k][batch_idx].cpu() for k in gts}
                        plots = generate_val_plots(
                            rollout=rollout,
                            gt=gts,
                            ts=conds["timestep"],
                            phase=f"Random draw",
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

            # for ph in metrics.keys():
            for m in metrics.keys():
                if metrics[m].sum() != 0.0:
                    metrics[m] = metrics[m] / n_timesteps_acc
                    vals = metrics[m]
                    for t in range(tot_eval_steps):
                        log_metric_dict[f"{valname}/{m}_x{t + 1}"] = vals[t]

        if not rank:
            # Save model if validation loss on trajectories improves
            val_loss = metrics[cfg.validation.model_selection_metric].mean()
            loss_val_min = save_model_and_config(
                model,
                optimizer=opt,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                val_loss=val_loss,
                loss_val_min=loss_val_min,
            )
        else:
            warnings.warn(f"checkpoints will not be stored for rank {rank}")

    return log_metric_dict, val_plots, loss_val_min
