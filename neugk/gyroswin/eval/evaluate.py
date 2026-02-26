"""Evaluation pipeline for the GyroSwin model."""

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
from neugk.dataset import CycloneDataset
from neugk.eval import validation_metrics
from neugk.plot_utils import generate_val_plots


def denormalize_rollout(
    rollout: Dict[str, torch.Tensor], idx_data: Dict, denormalize_fn: Callable
):
    """Denormalize rollout predictions over batch and time."""
    # iterate over fields
    for k, v in rollout.items():
        rollout[k] = torch.stack(
            [
                torch.stack(
                    [
                        denormalize_fn(f, **{k: v[t, b]})
                        for b, f in enumerate(idx_data["file_index"].tolist())
                    ]
                )
                for t in range(v.shape[0])
            ]
        )
    return rollout


def get_target_rollout(
    output_fields: List[str],
    dataset: CycloneDataset,
    idx_data: Dict,
    n_eval_steps: int,
    bundle_seq_length: int,
):
    """Fetch ground truth trajectory targets for rollout comparison."""
    tgts = defaultdict(list)
    # fetch sequence
    for t in range(0, n_eval_steps, bundle_seq_length):
        sample = dataset.get_at_time(
            idx_data["file_index"].long(),
            (idx_data["timestep_index"] + t).long(),
            get_normalized=False,
            num_workers=4,
        )
        for key in output_fields:
            tgts[key].append(getattr(sample, f"y_{key}"))

    # stack fields
    for key, val in tgts.items():
        if bundle_seq_length == 1:
            tgts[key] = torch.stack(val, 0)
        elif key == "flux":
            tgts[key] = rearrange(torch.stack(val, 1), "b t -> t b")
        elif key == "phi":
            tgts[key] = rearrange(torch.stack(val, 1), "b t ... -> t b ...")
        else:
            tgts[key] = rearrange(torch.stack(val, 2), "b c t ... -> t b c ...")
    return dict(tgts)


def get_rollout_fn(
    n_steps: int,
    bundle_steps: int,
    dataset: Dataset,
    predict_delta: bool = False,
    use_amp: bool = False,
    use_bf16: bool = False,
    device: str = "cuda",
) -> Callable:
    """Return a function that performs autoregressive model rollouts."""

    def _rollout(
        model: nn.Module, inputs: Dict, idx_data: Dict, conds: Dict
    ) -> torch.Tensor:
        # compute actual steps
        rollout_steps = min(
            [
                (dataset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i]))
                // bundle_steps
                - 1
                for i, f_idx in enumerate(idx_data["file_index"].tolist())
            ]
            + [n_steps]
        )

        # initialization
        tot_ts = rollout_steps * bundle_steps
        inputs_t = inputs.copy()
        preds = defaultdict(list)

        # compute timesteps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, bundle_steps))
            for ts in idx_data["timestep_index"].tolist()
        ]
        tsteps = dataset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))
        fluxes = []
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # rollout loop
        with torch.no_grad():
            for i in range(rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    conds["timestep"] = tsteps[:, i].to(device)
                    pred = model(**inputs_t, **conds)

                    if "flux" in pred:
                        fluxes.append(pred.pop("flux").cpu())

                    if predict_delta:
                        for k in pred:
                            pred[k] = pred[k] + inputs_t[k]

                    # update next input
                    for k in inputs_t:
                        if k in pred:
                            inputs_t[k] = pred[k].clone().float()

                # cache predictions
                for k, v in pred.items():
                    preds[k].append(
                        v.cpu().unsqueeze(2) if v.ndim in [4, 5, 7] else v.cpu()
                    )

        # finalize predictions
        for k, v in preds.items():
            if "position" in inputs_t:
                preds[k] = torch.cat([p.unsqueeze(2) for p in v], 2)
            else:
                preds[k] = torch.cat(v, 2)
            preds[k] = rearrange(preds[k], "b c t ... -> t b c ...")[
                : rollout_steps * bundle_steps
            ]

        if fluxes:
            preds["flux"] = rearrange(torch.stack(fluxes, dim=-1), "b t -> t b")

        return {k: p.to(dtype=torch.float32) for k, p in preds.items()}

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
    """Run rollout evaluation on all validation sets and compute physics metrics."""
    # constants setup
    n_eval_steps = cfg.validation.n_eval_steps
    bundle_seq_length = cfg.model.bundle_seq_length
    tot_eval_steps = n_eval_steps * bundle_seq_length

    predict_delta = cfg.training.predict_delta
    eval_integrals = cfg.validation.eval_integrals
    use_tqdm = cfg.logging.tqdm
    use_amp = cfg.amp.enable
    use_bf16 = use_amp and cfg.amp.bfloat and torch.cuda.is_bf16_supported()

    # field mapping
    input_fields = list(cfg.dataset.input_fields)
    if cfg.model.name in ["pointnet", "transolver", "transformer"]:
        input_fields.append("position")

    output_fields = [
        k
        for k, w in cfg.model.loss_weights.items()
        if w > 0.0 or cfg.model.loss_scheduler[k]
    ]
    if set(output_fields) != {"df", "phi", "flux"}:
        eval_integrals = False
    if eval_integrals:
        output_fields = ["df", "phi", "flux"]

    # validation check
    if epoch % cfg.validation.validate_every_n_epochs != 0 and epoch != 1:
        return {}, {}, loss_val_min

    # prepare modules
    model.eval()
    loss_wrap.eval().cpu()

    log_metric_dict = {}
    val_plots = {}

    # validation loop
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
        metrics = {k: torch.zeros([tot_eval_steps]) for k in loss_wrap.all_losses}
        n_timesteps_acc = torch.zeros([tot_eval_steps])

        if use_tqdm and (not dist.is_initialized() or rank == 0):
            valloader = tqdm(
                valloader,
                desc=f"validation holdout {'trajectories' if val_idx == 0 else 'samples'}",
            )

        # batch loop
        for idx, sample in enumerate(valloader):
            inputs = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in input_fields
            }
            conds = {
                k: getattr(sample, k).to(device, non_blocking=True)
                for k in cfg.model.conditioning
            }
            idx_data = {
                k: getattr(sample, k).to(device)
                for k in ["file_index", "timestep_index"]
            }

            # run rollout
            tgts = get_target_rollout(
                output_fields, valset, idx_data, n_eval_steps, bundle_seq_length
            )
            rollout = rollout_fn(model, inputs, idx_data, conds)

            # denormalize
            rollout = denormalize_rollout(rollout, idx_data, valset.denormalize)
            tgts = {k: v.cpu() for k, v in tgts.items()}
            rollout = {k: v.cpu() for k, v in rollout.items()}

            # handle zonal flow
            if cfg.dataset.separate_zf:

                def _recombine_zf(x):
                    return torch.cat(
                        [x[:, :, 0::2].sum(2, True), x[:, :, 1::2].sum(2, True)], dim=2
                    )

                if rollout["df"].shape[2] % 2 == 0:
                    rollout["df"] = _recombine_zf(rollout["df"])
                if tgts["df"].shape[2] % 2 == 0:
                    tgts["df"] = _recombine_zf(tgts["df"])

            # compute validation metrics
            metrics_i, integrated_i = validation_metrics(
                tgts=tgts,
                preds=rollout,
                geometry=sample.geometry,
                loss_wrap=loss_wrap,
                eval_integrals=eval_integrals,
            )

            if integrated_i and integrated_i[0] is not None:
                rollout["phi_int"] = torch.stack([i["phi"] for i in integrated_i])

            # accumulate metrics
            for k in metrics.keys():
                cur_len = metrics_i[k].shape[-1]
                if cur_len < tot_eval_steps:
                    metrics_i[k] = torch.cat(
                        [
                            metrics_i[k],
                            torch.zeros(
                                tot_eval_steps - cur_len, dtype=metrics_i[k].dtype
                            ),
                        ],
                        dim=-1,
                    )
                    metrics[k] += metrics_i[k]
                    n_timesteps_acc += (
                        torch.arange(1, tot_eval_steps + 1) <= cur_len
                    ).to(dtype=metrics_i[k].dtype)
                else:
                    metrics[k] += metrics_i[k]
                    n_timesteps_acc += torch.ones([tot_eval_steps])

            # update plots
            if "position" not in inputs:
                if val_idx == 0:
                    batch_idx = torch.randint(
                        0, len(idx_data["timestep_index"]), (1,)
                    ).item()
                    rollout_plot = {k: v[:, batch_idx] for k, v in rollout.items()}
                    plot_gt = {k: v[0][batch_idx] for k, v in tgts.items()}
                    val_plots.update(
                        generate_val_plots(
                            rollout_plot, plot_gt, conds["timestep"], "random draw"
                        )
                    )
                elif idx == 0:
                    val_plots.update(
                        generate_val_plots(
                            rollout, tgts, conds["timestep"], "holdout samples"
                        )
                    )

        # sync ddp metrics
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

        # finalize logs
        for m in metrics:
            if metrics[m].sum() != 0.0:
                metrics[m] /= n_timesteps_acc.clamp(min=1)
                for t in range(tot_eval_steps):
                    log_metric_dict[f"{valname}/{m}_x{t + 1}"] = metrics[m][t]

    # persist checkpoints
    if rank == 0:
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
