import warnings
from functools import partial
from collections import defaultdict
import torch
import torch.distributed as dist
from tqdm import tqdm

from utils import split_batch_into_phases, save_model_and_config
from train.losses import relative_norm_mse
from dataset import CycloneSample
from eval import (
    validation_metrics,
    generate_val_plots,
    dump_rollout,
    request_gkw_sim,
    get_rollout_fn,
    get_flux_plot
)

def evaluate(rank, world_size, model, valsets, valloaders, opt, epoch, cfg, device, gkw_args,
             loss_val_min):
    # Validation loop
    n_eval_steps = cfg.validation.n_eval_steps
    bundle_seq_length = cfg.model.bundle_seq_length
    tot_eval_steps = n_eval_steps * bundle_seq_length
    predict_delta = cfg.training.predict_delta
    denormalize = cfg.validation.denormalize
    use_gkw = cfg.validation.use_gkw
    use_tqdm = cfg.logging.tqdm
    use_amp = cfg.use_amp
    input_fields = cfg.dataset.input_fields
    outputs = cfg.model.losses
    conditioning = cfg.model.conditioning
    idx_keys = ["file_index", "timestep_index"]
    log_metric_dict = {}
    gkw_kfiles = []
    val_plots = {}
    val_freq = cfg.validation.validate_every_n_epochs

    # Validation loop
    if (epoch % val_freq) == 0 or epoch == 1:
        model.eval()
        for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):

            rollout_fn = get_rollout_fn(
                n_steps=n_eval_steps,
                bundle_steps=cfg.model.bundle_seq_length,
                dataset=valset,
                predict_delta=predict_delta,
                device=str(device),
                use_amp=use_amp,
            )

            valname = (
                "holdout_trajectories" if val_idx == 0 else "holdout_samples"
            )
            # TODO configurable metric list
            metric_fn_list = {
                # NOTE: average across all dimensions except timesteps
                "relative_norm_mse": partial(relative_norm_mse, dim_to_keep=0),
            }
            metrics = defaultdict(dict)
            for phase in ["linear", "saturated"]:
                for key in cfg.model.losses:
                    metrics[phase][key] = torch.zeros([len(metric_fn_list), tot_eval_steps])

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
            with torch.no_grad():
                for idx, sample in enumerate(valloader):
                    sample: CycloneSample
                    inputs = {k: getattr(sample, k).to(device, non_blocking=True) for k in input_fields}
                    gts = {k: getattr(sample, f"y_{k}").to(device, non_blocking=True) for k in outputs}
                    conds = {k: getattr(sample, k).to(device, non_blocking=True) for k in conditioning}
                    idx_data = {k: getattr(sample, k).to(device) for k in idx_keys}

                    # TODO: dont hardcode this
                    phase_change = 24 if cfg.dataset.offset == 0 else 0
                    inputs_list, gts_list, conds_list, idx_data_list, phase_list = split_batch_into_phases(
                        phase_change,
                        inputs, gts, conds,
                        idx_data,
                    )

                    # Iterate over the splits
                    for i in range(len(inputs_list)):
                        inputs = inputs_list[i]
                        gts = gts_list[i]
                        conds = conds_list[i]
                        idx_data = idx_data_list[i]
                        phase = phase_list[i]

                        # get the rolled out validation trajectories
                        rollout = rollout_fn(
                            model,
                            inputs,
                            idx_data,
                            conds
                        )

                        # denormalize rollout and target for evaluation / plots
                        if denormalize:
                            # denormalize rollout and target for evaluation / plots
                            for key in input_fields:
                                rollout[key] = torch.stack(
                                    [
                                        torch.stack(
                                            [
                                                valset.denormalize(
                                                    rollout[key][t, b], f
                                                )
                                                for b, f in enumerate(
                                                idx_data["file_index"].tolist()
                                            )
                                            ]
                                        )
                                        for t in range(rollout[key].shape[0])
                                    ]
                                )
                                gts[key] = torch.stack(
                                    [
                                        valset.denormalize(gts[key][b], f)
                                        for b, f in enumerate(idx_data["file_index"].tolist())
                                    ]
                                )

                        # TODO: smarter (i.e. use timeindex when we output a dataclass from the dataset)
                        metrics_i = validation_metrics(
                            rollout,
                            idx_data,
                            bundle_seq_length,
                            valset,
                            metrics_fns=metric_fn_list,
                            get_normalized=not denormalize,
                        )

                        if use_gkw:
                            # onestep predictions (time dimension = 0)
                            rollout_dict = {
                                int(t_idx.item()): rollout["df"][0, b]
                                .cpu()
                                .numpy()
                                for b, t_idx in enumerate(idx_data["timestep_index"])
                            }
                            # TODO put original file in config?
                            src_config_path = (
                                valset.files[0]
                                .split("/")[-1]
                                .replace("_ifft", "")
                                .replace("_separate_zf", "")
                                .replace(".h5", "")
                            )
                            src_config_path = (
                                f"/restricteddata/ukaea/"
                                f"gyrokinetics/raw/{src_config_path}"
                            )
                            gkw_kfiles.extend(
                                dump_rollout(
                                    rollout_dict, gkw_args["dump_path"], src_config_path
                                )
                            )

                        for key in metrics_i.keys():
                            if metrics_i[key].shape[-1] < tot_eval_steps:
                                # end of dataset, need to pad the tensor
                                diff = tot_eval_steps - metrics_i[key].shape[-1]
                                metrics_i[key] = torch.cat(
                                    [
                                        metrics_i[key],
                                        torch.zeros(
                                            [metrics_i[key].shape[0], diff],
                                            dtype=metrics_i[key].dtype,
                                        ),
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
                                        else
                                        "Linear Phase"
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
                        torch.zeros_like(cur_ts, dtype=cur_ts.dtype, device=cur_ts.device)
                        for _ in range(world_size)]
                    dist.all_gather(gathered_ts, cur_ts)
                    n_timesteps_acc[phase] = torch.cat(gathered_ts).sum(0).cpu()

                    for m in metrics[phase].keys():
                        cur_metric = metrics[phase][m].to(device)
                        gathered_ms = [torch.zeros_like(cur_metric, dtype=cur_metric.dtype, device=cur_metric.device)
                                    for _ in range(world_size)]
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

            for phase in metrics.keys():
                for out_type in metrics[phase].keys():
                    metrics[phase][out_type] = metrics[phase][out_type] / n_timesteps_acc[phase].unsqueeze(0)

            for idx, metric_name in enumerate(metric_fn_list):
                for phase in metrics.keys():
                    for out_type in metrics[phase].keys():
                        vals = metrics[phase][out_type][idx, ...]
                        for t in range(tot_eval_steps):
                            log_metric_dict[
                                f"val_{valname}/{out_type}_{metric_name}_{phase}_x{t + 1}"
                            ] = vals[t]

            if val_idx == 0:
                # trajectoy validation
                n_timesteps_acc_model_saving = n_timesteps_acc

        if use_gkw:
            # run gkw on accumulated rollout
            gkw_args["futures"][epoch] = gkw_args["executor"].submit(
                request_gkw_sim, gkw_args["dump_path"],
                gkw_kfiles, cfg.logging.run_id
            )

        if cfg.ckpt_path is not None and not rank:
            mse_sat = log_metric_dict[
                "val_holdout_trajectories/df_relative_norm_mse_saturated_x1"
            ]
            mse_lin = log_metric_dict[
                "val_holdout_trajectories/df_relative_norm_mse_linear_x1"
            ]
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
            warnings.warn(
                "`cfg.ckpt_path` is not set: checkpoints will not be stored"
            )

    # check gkw future once a epoch
    if use_gkw:
        fluxes, potentials = {}, {}
        done_futures = []
        for future_epoch, fut in gkw_args["gkw_futures"].items():
            if fut.done():
                done_futures.append(future_epoch)
                fluxes[future_epoch], potentials[future_epoch] = fut.result()
                flux_plot, flux_error = get_flux_plot(
                    fluxes[future_epoch], dataset=valsets[0]
                )
                gkw_logs = {
                    "Flux": flux_plot,
                    "val_holdout_trajectories/flux_mse": flux_error,
                }
                log_metric_dict = log_metric_dict | gkw_logs

        # close finished futures
        for k in done_futures:
            del gkw_args["gkw_futures"][k]

    return log_metric_dict, val_plots, loss_val_min