from datetime import datetime, timedelta

from functools import partial
import torch
from torch.cuda import reset_peak_memory_stats, max_memory_allocated
from torch.nn.utils import clip_grad_norm_
import warnings
from tqdm import tqdm
from time import perf_counter_ns
from collections import defaultdict
from transformers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from concurrent.futures import ThreadPoolExecutor

from dataset import get_data, CycloneSample
from train import get_pushforward_fn, relative_norm_mse, pretrain_autoencoder
from eval import get_rollout_fn, validation_metrics, generate_val_plots, get_flux_plot
from eval.gkw_client import request_gkw_sim, dump_rollout
from utils import (
    load_model_and_config,
    save_model_and_config,
    setup_logging,
    split_batch_into_phases,
)


import torch


def get_xnet(cfg, dataset):
    # TODO need to standardize modules everywhere (eg for different inputs)

    latent_dim = cfg.model.latent_dim
    if not cfg.dataset.separate_zf:
        problem_dim = len(cfg.dataset.active_keys)
    else:
        problem_dim = 4
        problem_dim += (
            (cfg.dataset.split_into_bands - 1) * 2
            if cfg.dataset.split_into_bands
            else 0
        )

    from experimental.swin_xnet import SwinXnet
    from models.utils import ContinuousConditionEmbed

    df_patch_size = cfg.model.swin.patch_size
    phi_patch_size = (14, 2, 8)
    df_window_size = cfg.model.swin.window_size
    phi_window_size = (14, 4, 6)
    df_base_resolution = dataset.resolution
    phi_base_resolution = dataset.phi_resolution
    num_heads = cfg.model.swin.num_heads
    depth = cfg.model.swin.depth
    num_layers = cfg.model.num_layers
    gradient_checkpoint = cfg.model.swin.gradient_checkpoint
    patching_hidden_ratio = cfg.model.swin.merging_hidden_ratio
    unmerging_hidden_ratio = cfg.model.swin.unmerging_hidden_ratio
    c_multiplier = cfg.model.swin.c_multiplier
    abs_pe = cfg.model.swin.abs_pe
    act_fn = getattr(torch.nn, cfg.model.swin.act_fn)

    cond_fn = None
    n_cond = cfg.model.swin.timestep_conditioning + cfg.model.swin.itg_conditioning
    if n_cond > 0:
        cond_fn = ContinuousConditionEmbed(128, n_cond)

    if cfg.model.bundle_seq_length > 1:
        raise NotImplementedError

    model = SwinXnet(
        dim=latent_dim,
        df_base_resolution=df_base_resolution,  # TODO
        phi_base_resolution=phi_base_resolution,  # TODO
        df_patch_size=df_patch_size,
        phi_patch_size=phi_patch_size,
        df_window_size=df_window_size,
        phi_window_size=phi_window_size,
        depth=depth,
        num_heads=num_heads,
        in_channels=problem_dim,
        out_channels=problem_dim,
        num_layers=num_layers,
        use_checkpoint=gradient_checkpoint,
        drop_path=0.1,
        abs_pe=abs_pe,
        c_multiplier=c_multiplier,
        merging_hidden_ratio=patching_hidden_ratio,
        unmerging_hidden_ratio=unmerging_hidden_ratio,
        conditioning=cond_fn,
        act_fn=act_fn,
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model


def ddp_setup(rank, world_size):
    init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


def runner(rank, cfg, world_size):
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else "cpu"
    if cfg.use_ddp and world_size > 1:
        ddp_setup(rank, world_size)
        use_ddp = True
    else:
        use_ddp = False

    if not rank:
        data_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
        cfg.logging.run_name = f"{cfg.model.name}_{data_and_time}"
        writer = setup_logging(cfg)
    else:
        writer = None

    # TODO currently only support one resolution for all cyclones
    datasets, dataloaders, augmentations = get_data(cfg)
    if len(datasets) == 3:
        # holdout trajectories and holdout samples for validation
        trainset, valsets = datasets[0], datasets[1:]
        trainloader, valloaders = dataloaders[0], dataloaders[1:]
    else:
        # only holdout trajectories for validation
        trainset, valsets = datasets
        valsets = [valsets]
        trainloader, valloaders = dataloaders
        valloaders = [valloaders]

    if not cfg.dataset.separate_zf:
        problem_dim = len(cfg.dataset.active_keys)
    else:
        problem_dim = 4
        problem_dim += (
            (cfg.dataset.split_into_bands - 1) * 2
            if cfg.dataset.split_into_bands
            else 0
        )

    model = get_xnet(cfg, dataset=trainset)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank])

    bundle_seq_length = cfg.model.bundle_seq_length
    use_gkw = cfg.validation.use_gkw
    if use_gkw:
        if cfg.dataset.normalization_scope == "sample":
            raise UserWarning(
                "Cannot denormalize with `normalization_scope='sample'`. "
                "Cannot compute correct GKW results."
            )
        gkw_executor = ThreadPoolExecutor(max_workers=1)
        # TODO hardcoded for now
        gkw_dump_path = "/system/user/publicdata/gyrokinetics/dumps/test_gkw_client"
        gkw_futures = {}

    opt_state_dict = None
    if cfg.load_ckp is True and cfg.ckpt_path is not None:
        # TODO move config loading to here (now in main.py)
        model, opt_state_dict, _ = load_model_and_config(
            cfg.ckpt_path, model=model, device=device
        )

    if cfg.mode == "train":
        n_epochs = cfg.training.n_epochs
        total_steps = n_epochs * len(trainloader)

        # optimizer config
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        use_amp = cfg.use_amp

        scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        if cfg.training.scheduler is not None:
            scheduler = get_scheduler(
                name=cfg.training.scheduler,
                optimizer=opt,
                num_warmup_steps=total_steps // 6,
                num_training_steps=total_steps,
            )

        if opt_state_dict is not None:
            opt.load_state_dict(opt_state_dict)

        # configure loss
        predict_delta = cfg.training.predict_delta
        # and pushforward
        pf_cfg = cfg.training.pushforward
        pushforward_fn = None
        # TODO
        # if sum(pf_cfg.unrolls) > 0:
        #     pushforward_fn = get_pushforward_fn(
        #         n_unrolls_schedule=pf_cfg.unrolls,
        #         probs_schedule=pf_cfg.probs,
        #         epoch_schedule=pf_cfg.epochs,
        #         predict_delta=predict_delta,
        #         dataset=trainset,
        #         bundle_steps=bundle_seq_length,
        #         use_amp=use_amp,
        #     )

        loss_val_min = torch.inf

        if cfg.training.pretraining:
            model = pretrain_autoencoder(
                model, cfg, trainloader, valloaders, writer, device
            )  # only valuate on the holdout trajectories, not the holdout samples
            if not hasattr(model, "module") and use_ddp:
                model = DDP(model, device_ids=[rank])

        use_tqdm = cfg.logging.tqdm if not use_ddp else False
        for epoch in range(1, n_epochs + 1):
            train_mse = 0
            train_df_mse = 0
            train_phi_mse = 0
            model.train()
            info_dict = defaultdict(list)
            t_start_data = perf_counter_ns()

            if use_tqdm or (use_ddp and not rank):
                trainloader = tqdm(trainloader, "Training")
            for i, sample in enumerate(trainloader):
                # reset_peak_memory_stats(device)
                sample: CycloneSample
                df = sample.x.to(device, non_blocking=True)
                y_df = sample.y.to(device, non_blocking=True)
                phi = sample.x_poten.to(device, non_blocking=True)
                y_phi = sample.y_poten.to(device, non_blocking=True)
                ts = sample.timestep.to(device)
                itg = sample.itg.to(device)

                # TODO should augmentations take place before moving to GPU?
                if augmentations is not None:
                    for aug_fn in augmentations:
                        df = aug_fn(df)

                # dataloading timings
                info_dict["data_ms"].append((perf_counter_ns() - t_start_data) / 1e6)

                if pushforward_fn:
                    start_pf = perf_counter_ns()
                    # accessory information for pf (to retreive unrolled target)
                    file_idx = sample.file_index.to(device)
                    ts_index = sample.timestep_index.to(device)
                    x, ts, y = pushforward_fn(
                        model, x, ts, itg, ts_index, y, file_idx, epoch
                    )
                    info_dict["pf_ms"].append((perf_counter_ns() - start_pf) / 1e6)
                else:
                    info_dict["pf_ms"].append(0.0)

                t_start_fwd = perf_counter_ns()

                with torch.autocast(str(cfg.device), dtype=amp_dtype, enabled=use_amp):
                    pred_df, pred_phi = model(df, phi, timestep=ts, itg=itg)
                    if predict_delta:
                        pred_df = df + pred_df
                        pred_phi = phi + pred_phi
                    df_loss = relative_norm_mse(pred_df, y_df)
                    phi_loss = relative_norm_mse(pred_phi, y_phi)
                    loss = df_loss + phi_loss

                # forward timing
                info_dict["forward_ms"].append((perf_counter_ns() - t_start_fwd) / 1e6)
                t_start_bkd = perf_counter_ns()

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if cfg.training.clip_grad:
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if cfg.training.scheduler is not None:
                    scheduler.step()

                train_mse += loss.item()
                train_df_mse += df_loss.item()
                train_phi_mse += phi_loss.item()

                info_dict["backward_ms"].append((perf_counter_ns() - t_start_bkd) / 1e6)
                info_dict["memory_mb"].append(max_memory_allocated(device) / 1024**2)
                t_start_data = perf_counter_ns()

            train_mse /= len(trainloader)
            train_df_mse /= len(trainloader)
            train_phi_mse /= len(trainloader)
            train_losses_dict = {
                "train/relative_norm_mse": train_mse,
                "train/df_mse": train_df_mse,
                "train/phi_mse": train_phi_mse,
                "train/lr": (
                    scheduler.get_last_lr()[0]
                    if cfg.training.scheduler
                    else cfg.training.learning_rate
                ),
            }
            info_dict = {f"info/{k}": sum(v) / len(v) for k, v in info_dict.items()}

            # Validation loop
            separate_zf = cfg.dataset.separate_zf
            n_eval_steps = cfg.validation.n_eval_steps
            val_freq = cfg.validation.validate_every_n_epochs
            tot_eval_steps = n_eval_steps * bundle_seq_length
            log_metric_dict = {}
            gkw_kfiles = []
            val_plots = {}
            if (epoch % val_freq) == 0 or epoch == 1 and not rank:
                # Validation loop
                model.eval()
                for val_idx, (valset, valloader) in enumerate(zip(valsets, valloaders)):

                    rollout_fn = get_rollout_fn(
                        problem_dim=problem_dim,
                        n_steps=n_eval_steps,
                        bundle_steps=bundle_seq_length,
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
                    metrics = {
                        "linear": torch.zeros([len(metric_fn_list), tot_eval_steps]),
                        "saturated": torch.zeros([len(metric_fn_list), tot_eval_steps]),
                    }
                    n_timesteps_acc = {
                        "linear": torch.zeros([tot_eval_steps]),
                        "saturated": torch.zeros([tot_eval_steps]),
                    }
                    if use_tqdm or (use_ddp and not rank):
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
                            df = sample.x.to(device, non_blocking=True)
                            phi = sample.x_poten.to(device, non_blocking=True)
                            y = sample.y.to(device, non_blocking=True)
                            y_phi = sample.y_poten.to(device, non_blocking=True)
                            ts = sample.timestep.to(device)
                            itg = sample.itg.to(device)
                            file_idx = sample.file_index.to(device)
                            ts_index = sample.timestep_index.to(device)

                            # TODO: dont hardcode this
                            phase_change = 24
                            (
                                df_list,
                                y_list,
                                ts_list,
                                itg_list,
                                file_idx_list,
                                ts_index_list,
                                phase_list,
                                phi_list,
                                y_phi_list,
                            ) = split_batch_into_phases(
                                phase_change,
                                df,
                                y,
                                ts,
                                itg,
                                file_idx,
                                ts_index,
                                phi=phi,
                                y_phi=y_phi,
                            )

                            # Iterate over the splits
                            for i in range(len(df_list)):
                                df = df_list[i]
                                phi = phi_list[i]
                                y = y_list[i]
                                y_phi = y_phi_list[i]
                                ts = ts_list[i]
                                itg = itg_list[i]
                                file_idx = file_idx_list[i]
                                ts_index = ts_index_list[i]
                                phase = phase_list[i]

                                # get the rolled out validation trajectories
                                df_rollout, phi_rollout = rollout_fn(
                                    model,
                                    df,
                                    phi,
                                    file_idx=file_idx,
                                    ts_index_0=ts_index,
                                    itg=itg,
                                )

                                # denormalize rollout and target for evaluation / plots
                                df_rollout = torch.stack(
                                    [
                                        torch.stack(
                                            [
                                                valset.denormalize(df_rollout[t, b], f)
                                                for b, f in enumerate(file_idx.tolist())
                                            ]
                                        )
                                        for t in range(df_rollout.shape[0])
                                    ]
                                )
                                y = torch.stack(
                                    [
                                        valset.denormalize(y[b], f)
                                        for b, f in enumerate(file_idx.tolist())
                                    ]
                                )

                                # TODO: smarter (i.e. use timeindex when we output a dataclass from the dataset)
                                metrics_i = validation_metrics(
                                    df_rollout,
                                    file_idx,
                                    ts_index,
                                    bundle_seq_length,
                                    valset,
                                    metric_fn_list,
                                )
                                if use_gkw:
                                    # onestep predictions (time dimension = 0)
                                    rollout_dict = {
                                        int(t_idx.item()): df_rollout[0, b]
                                        .cpu()
                                        .numpy()
                                        for b, t_idx in enumerate(ts_index)
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
                                            rollout_dict, gkw_dump_path, src_config_path
                                        )
                                    )

                                if metrics_i.shape[-1] < tot_eval_steps:
                                    # end of dataset, need to pad the tensor
                                    diff = tot_eval_steps - metrics_i.shape[-1]
                                    metrics_i = torch.cat(
                                        [
                                            metrics_i,
                                            torch.zeros(
                                                [metrics_i.shape[0], diff],
                                                dtype=metrics_i.dtype,
                                            ),
                                        ],
                                        dim=-1,
                                    )
                                    metrics[phase] += metrics_i
                                    validated_steps = (
                                        torch.arange(1, tot_eval_steps + 1)
                                        <= metrics_i.shape[-1]
                                    ).to(dtype=metrics_i.dtype)
                                else:
                                    metrics[phase] += metrics_i
                                    validated_steps = torch.ones([tot_eval_steps])
                                n_timesteps_acc[phase] += validated_steps

                                if val_idx == 0:
                                    # holdout trajectories valset
                                    if idx in [0, 20]:
                                        plots = generate_val_plots(
                                            x_rollout=df_rollout,
                                            y=y,
                                            ts=ts,
                                            phase=(
                                                "Linear Phase"
                                                if idx == 0
                                                else "Saturated Phase"
                                            ),
                                            phi_rollout=phi_rollout,
                                            y_phi=y_phi,
                                        )
                                        val_plots.update(plots)
                                else:
                                    # holdout samples valset
                                    if idx == 0:
                                        plots = generate_val_plots(
                                            x_rollout=df_rollout,
                                            y=y,
                                            ts=ts,
                                            phase="Holdout samples",
                                            phi_rollout=phi_rollout,
                                            y_phi=y_phi,
                                        )
                                        val_plots.update(plots)

                    for key in metrics.keys():
                        metrics[key] = metrics[key] / n_timesteps_acc[key].unsqueeze(0)

                    for idx, metric_name in enumerate(metric_fn_list):
                        for phase, metric in metrics.items():
                            vals = metric[idx, ...]
                            for t in range(tot_eval_steps):
                                log_metric_dict[
                                    f"val_{valname}/{metric_name}_{phase}_x{t + 1}"
                                ] = vals[t]

                    if val_idx == 0:
                        # trajectoy validation
                        n_timesteps_acc_model_saving = n_timesteps_acc

                if use_gkw:
                    # run gkw on accumulated rollout
                    gkw_futures[epoch] = gkw_executor.submit(
                        request_gkw_sim, gkw_dump_path, gkw_kfiles, cfg.logging.run_id
                    )

                if cfg.ckpt_path is not None and not rank:
                    mse_sat = log_metric_dict[
                        "val_holdout_trajectories/relative_norm_mse_saturated_x1"
                    ]
                    mse_lin = log_metric_dict[
                        "val_holdout_trajectories/relative_norm_mse_linear_x1"
                    ]
                    sat_ts = n_timesteps_acc_model_saving["saturated"]
                    lin_ts = n_timesteps_acc_model_saving["linear"]
                    val_loss = (mse_sat * sat_ts + mse_lin * lin_ts) / (sat_ts + lin_ts)
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
                for future_epoch, fut in gkw_futures.items():
                    if fut.done():
                        done_futures.append(future_epoch)
                        fluxes[future_epoch], potentials[future_epoch] = fut.result()
                        flux_plot, flux_error = get_flux_plot(
                            fluxes[future_epoch], dataset=valset
                        )
                        gkw_logs = {
                            "Flux": flux_plot,
                            "val_holdout_trajectories/flux_mse": flux_error,
                        }
                        if writer and not rank:
                            # NOTE not possible to log at previous step...
                            writer.log(gkw_logs)
                # close finished futures
                for k in done_futures:
                    del gkw_futures[k]

            if use_ddp:
                # Add barrier for all ranks and wait for rank 0
                torch.distributed.barrier()
            # log to wandb
            epoch_logs = train_losses_dict | log_metric_dict
            if writer and not rank:
                wandb_logs = epoch_logs | info_dict
                # log epoch details
                if not val_plots:
                    writer.log(wandb_logs)
                else:
                    writer.log(wandb_logs, commit=False)
                    writer.log(val_plots)

            epoch_str = str(epoch).zfill(len(str(int(n_epochs))))
            total_time = (
                info_dict["info/forward_ms"]
                + info_dict["info/backward_ms"]
                + info_dict["info/data_ms"]
                + info_dict["info/pf_ms"]
            )
            if not rank:
                print(
                    f"Epoch: {epoch_str}, "
                    f"{', '.join([f'{k}: {v:.5f}' for k, v in epoch_logs.items()])}"
                    f", step time: {total_time:.2f}ms"
                )
        if use_gkw:
            # TODO wait some more if some gkw process is still hanging
            pass

        if writer:
            writer.finish()

    if cfg.mode == "rollout":
        raise NotImplementedError("TODO")
