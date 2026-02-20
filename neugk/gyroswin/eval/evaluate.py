from typing import List, Dict, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from einops import rearrange

from tqdm import tqdm
from collections import defaultdict
import warnings

from neugk.utils import save_model_and_config
from neugk.dataset import CycloneSample, CycloneDataset
from neugk.eval import validation_metrics
from neugk.plot_utils import generate_val_plots


def denormalize_rollout(rollout, idx_data, denormalize_fn):
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


def get_target_rollout(
    output_fields,
    dataset: CycloneDataset,
    idx_data: Dict,
    n_eval_steps,
    bundle_seq_length,
):
    tgts = defaultdict(list)
    for t in range(0, n_eval_steps, bundle_seq_length):
        sample: CycloneSample = dataset.get_at_time(
            idx_data["file_index"].long(),
            (idx_data["timestep_index"] + t).long(),
            get_normalized=False,
            num_workers=4,
        )
        for key in output_fields:
            tgts[key].append(getattr(sample, f"y_{key}"))
    for key in tgts.keys():
        if bundle_seq_length == 1:
            tgts[key] = torch.stack(tgts[key], 0)
        else:
            if key == "flux":
                tgts[key] = rearrange(torch.stack(tgts[key], 1), "b t -> t b")
            elif key == "phi":
                tgts[key] = rearrange(torch.stack(tgts[key], 1), "b t ... -> t b ...")
            else:
                tgts[key] = rearrange(
                    torch.stack(tgts[key], 2), "b c t ... -> t b c ..."
                )
    return tgts


def get_rollout_fn(
    n_steps: int,
    bundle_steps: int,
    dataset: Dataset,
    predict_delta: bool = False,
    use_amp: bool = False,
    use_bf16: bool = False,
    device: str = "cuda",
) -> Callable:
    # correct step size by adding last bundle
    # n_steps_ = n_steps + bundle_steps - 1

    def _rollout(
        model: nn.Module,
        inputs: Dict,
        idx_data: Dict,
        conds: Dict,
    ) -> torch.Tensor:
        # cap the steps depending on the current max timestep
        rollout_steps = []
        for i, f_idx in enumerate(idx_data["file_index"].tolist()):
            ts_left = dataset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i])
            ts_left = ts_left // bundle_steps - 1
            rollout_steps.append(min(ts_left, n_steps))
        rollout_steps = min(rollout_steps)

        tot_ts = rollout_steps * bundle_steps
        inputs_t = inputs.copy()
        preds = defaultdict(list)
        # get corresponding timesteps
        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, ts_step))
            for ts in idx_data["timestep_index"].tolist()
        ]
        fluxes = []
        tsteps = dataset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i in range(0, rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    conds["timestep"] = tsteps[:, i].to(device)
                    pred = model(**inputs_t, **conds)
                    if "flux" in pred:
                        fluxes.append(pred["flux"].cpu())
                        del pred["flux"]

                    if predict_delta:
                        for key in pred.keys():
                            pred[key] = pred[key] + inputs_t[key]

                    for key in inputs_t.keys():
                        if key in pred:
                            # Position is not in pred, and is constant
                            inputs_t[key] = pred[key].clone().float()

                for key in pred:
                    # add time dim if not there
                    preds[key].append(
                        pred[key].cpu().unsqueeze(2)
                        if pred[key].ndim in [4, 5, 7]
                        else pred[key].cpu()
                    )

        for key in preds.keys():
            # only return desired size
            if "position" in inputs_t:
                preds[key] = torch.cat([p.unsqueeze(2) for p in preds[key]], 2)
            else:
                preds[key] = torch.cat(preds[key], 2)
            preds[key] = rearrange(preds[key], "b c t ... -> t b c ...")
            preds[key] = preds[key][: rollout_steps * bundle_steps, :, ...]
        if len(fluxes) > 0:
            preds["flux"] = rearrange(torch.stack(fluxes, dim=-1), "b t -> t b")
        # to float32 for integrals etc
        preds = {k: p.to(dtype=torch.float32) for k, p in preds.items()}
        return preds

    return _rollout


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
                bundle_steps=bundle_seq_length,
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
            if use_tqdm and (not dist.is_initialized() or rank == 0):
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
                conds = {
                    k: getattr(sample, k).to(device, non_blocking=True)
                    for k in conditioning
                }
                idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

                # get target rollout
                # TODO can speed up by returning targets only once / doing batched
                tgts = get_target_rollout(
                    output_fields, valset, idx_data, n_eval_steps, bundle_seq_length
                )

                # get the rolled out validation trajectories
                rollout = rollout_fn(model, inputs, idx_data, conds)

                # denormalize rollout and target for evaluation / plots
                # NOTE denormalize always true for integrals if denormalize:
                rollout = denormalize_rollout(rollout, idx_data, valset.denormalize)

                tgts = {k: v.cpu() for k, v in tgts.items()}
                rollout = {k: v.cpu() for k, v in rollout.items()}

                if cfg.dataset.separate_zf:

                    def _recombine_zf(x):
                        x = torch.cat(
                            [x[:, :, 0::2].sum(2, True), x[:, :, 1::2].sum(2, True)],
                            dim=2,
                        )
                        return x

                    # apply recombine_zf to preds and tgts
                    if rollout["df"].shape[2] % 2 == 0:
                        rollout["df"] = _recombine_zf(rollout["df"])
                    if tgts["df"].shape[2] % 2 == 0:
                        tgts["df"] = _recombine_zf(tgts["df"])

                metrics_i, integrated_i = validation_metrics(
                    tgts=tgts,
                    preds=rollout,
                    geometry=sample.geometry,
                    loss_wrap=loss_wrap,
                    eval_integrals=eval_integrals,
                )

                # add integrated potentials to rollout for comparison
                if integrated_i[0] is not None:
                    rollout["phi_int"] = torch.stack(
                        [integrated_i[t]["phi"] for t in range(len(integrated_i))]
                    )

                for key in metrics.keys():
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
                if "position" not in inputs:
                    # no plots for field-like baselines
                    if val_idx == 0:
                        # holdout trajectories valset
                        t_idx = idx_data["timestep_index"].tolist()
                        batch_idx = torch.randint(0, len(t_idx), (1,)).item()
                        rollout = {k: rollout[k][:, batch_idx].cpu() for k in rollout}
                        plot_gt = {k: tgts[k][0][batch_idx].cpu() for k in tgts}
                        plots = generate_val_plots(
                            rollout=rollout,
                            gt=plot_gt,
                            ts=conds["timestep"],
                            phase="Random draw",
                        )
                        val_plots.update(plots)
                    else:
                        # holdout samples valset
                        if idx == 0:
                            plots = generate_val_plots(
                                rollout=rollout,
                                gt=tgts,  # TODO ?
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
