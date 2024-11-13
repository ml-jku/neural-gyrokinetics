import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb

from dataset import get_data
from models import get_model
from eval.plot_utils import (
    get_gifs3x3,
    get_val_trajectory_plots,
    get_wandb_tables,
    get_power_spectra,
)
from train import (
    mse_timesteps,
    mae_timesteps,
    get_pushforward_trick,
    get_base_train_loss,
)
from eval import get_rollout, validation_metrics, ssim_tensor
from utils import load_model_and_config, save_model_and_config


def runner(cfg):
    (_, valset), (trainloader, valloader), augmentations = get_data(cfg)

    model = get_model(cfg)

    data_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    cfg.logging.run_name = f"{cfg.model.name}_{data_and_time}"

    device = cfg.device
    active_keys = valset.active_keys

    optimizer_state_dict = None
    if cfg.ckpt_path is not None:
        # TODO move config loading to here (now in main.py)
        model, optimizer_state_dict, _ = load_model_and_config(
            cfg.ckpt_path, model=model, device=device
        )

    model = model.to(device)

    if cfg.mode == "train":
        wandb_run = None
        if cfg.logging.wandb:
            wandb_config = OmegaConf.to_container(cfg)
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.logging.run_name,
                config=wandb_config,
                # save_code=True,
            )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        # configure loss
        loss_fn = get_base_train_loss(
            cfg.training.predict_delta, cfg.training.bundle_seq_length
        )
        pf_cfg = cfg.training.pushforward
        pushforward_fn = None
        if sum(pf_cfg.unrolls) > 0:
            pushforward_fn = get_pushforward_trick(
                pf_cfg.unrolls,
                pf_cfg.probs,
                schecule=pf_cfg.epochs,
                predict_delta=cfg.training.predict_delta,
                bundle_steps=cfg.training.bundle_seq_length,
            )

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        loss_val_min = torch.inf
        use_tqdm = cfg.logging.tqdm

        for epoch in range(cfg.training.n_epochs):
            train_mse = 0
            if use_tqdm:
                trainloader = tqdm(trainloader, "Training")
            for sample in trainloader:
                x, grid, y, ts = sample

                optimizer.zero_grad()

                if pushforward_fn:
                    x, grid, y, ts = pushforward_fn(model, x, grid, y, ts, epoch)

                loss = loss_fn(model, x, grid, y, ts)
                loss.backward()
                optimizer.step()

                train_mse += loss.item()

            train_mse /= len(trainloader)
            train_losses_dict = {"train/relative_norm_mse": train_mse}

            # Validation loop
            model.eval()
            # TODO configurable metric list
            # non_averaging_mse = partial(F.mse_loss(reduction="none"))
            metric_fn_list = {
                "mse": mse_timesteps,
                "ssim": ssim_tensor,
                "mae": mae_timesteps,
            }
            # TODO initialize dictionary with torch tensors initialized to 0 and without grad with 1 dimension of size tsteps
            # OR initialize tensor of metric with shape [n_metrics, n_timesteps]
            # metric_dict = defaultdict(lambda: 0.0)
            if use_tqdm:
                valloader = tqdm(valloader, "Validation")
            with torch.no_grad():
                for idx, sample in enumerate(valloader):
                    x, grid, y, ts = sample

                    # get the rolled out validation trajectories
                    x_rollout = get_rollout(
                        problem_dim=len(active_keys),
                        n_steps=y.shape[0],
                        bundle_steps=cfg.training.bundle_seq_length,
                        predict_delta=cfg.training.predict_delta,
                    )(model, x, grid, ts0=ts)  # (n_steps, bs, d, h, w)

                    y = y.cpu()

                    if idx == 0:
                        # initialize metrics tensor (y has shape [n_steps, bs, d, h, w])
                        metrics = validation_metrics(x_rollout, y, metric_fn_list)
                        frequencies, power_pred, power_gt = get_power_spectra(
                            x_rollout, y
                        )
                        # get images of trajectories
                        traj_plots = get_val_trajectory_plots(
                            x_rollout, y, [1, 5, 10, 25, 50]
                        )

                    else:
                        # get all validation metrics for the rollout
                        # metrics tensor will have shape [number_of_metrics, n_timesteps]
                        metrics += validation_metrics(x_rollout, y, metric_fn_list)
                        _, _power_pred, _power_gt = get_power_spectra(x_rollout, y)
                        power_pred += _power_pred
                        power_gt += _power_gt

            metrics = metrics / len(valloader)
            power_pred = power_pred / len(valloader)
            power_gt = power_gt / len(valloader)

            # create mse/ssim/mae tables for logging
            metric_tables = get_wandb_tables(metrics, list(metric_fn_list.keys()))
            # subsample to keep only 3 values
            log_metric_dict = {}
            rollout_len = y.shape[0]
            for idx, metric_name in enumerate(metric_fn_list):
                vals = metrics[idx, ...]
                for t in [0, int(0.5 * rollout_len) - 1, rollout_len - 1]:
                    log_metric_dict["val/" + metric_name + f"x{t+1}"] = vals[t]

            # Save model if validation loss improves
            loss_val_min = save_model_and_config(
                model,
                optimizer,
                cfg,
                epoch,
                # TODO decide target metric
                val_loss=log_metric_dict["val/msex1"],
                loss_val_min=loss_val_min,
            )

            # scheduler.step()

            # log to wandb
            epoch_logs = train_losses_dict | log_metric_dict
            if wandb_run:
                # log epoch details
                wandb_run.log(epoch_logs, commit=False)
                # log validation trajectory images
                images = [wandb.Image(graphic) for graphic in traj_plots]
                wandb_run.log(
                    {k: images[idx] for idx, k in enumerate(valset.active_keys)},
                    commit=False,
                )
                # log metrics graphs over full validation rollouts
                wandb_run.log(
                    {
                        key: wandb.plot.line(table, "t", key, title=f"{key}test")
                        for key, table in metric_tables.items()
                    },
                    commit=False,
                )
                # log power spectra
                wandb_run.log(
                    {
                        "power spectra": wandb.plot.line_series(
                            xs=frequencies,
                            ys=[power_pred, power_gt],
                            keys=["predicted", "groundtruth"],
                            title="Power Spectrum",
                            xname="frequency",
                        )
                    }
                )

            epoch_str = str(epoch).zfill(len(str(int(cfg.training.n_epochs))))
            print(
                f"Epoch: {epoch_str}, "
                f"{', '.join([f'{k}: {v:.5f}' for k, v in epoch_logs.items()])}"
            )
        if wandb_run:
            wandb_run.finish()

    if cfg.mode == "rollout":
        os.makedirs(cfg.logging.output_dir, exist_ok=True)

        run_dir = f"{cfg.logging.output_dir}/{cfg.logging.run_name}"

        trajs = list(valset.trajectory_tags())
        n_rollout_traj = min(cfg.rollout.n_rollout_traj, len(trajs))
        window_len = cfg.model.input_seq_length
        rollout_steps = cfg.rollout.rollout_steps

        rollout_fn = get_rollout(
            problem_dim=len(active_keys),
            n_steps=rollout_steps,
            bundle_steps=cfg.training.bundle_seq_length,
            predict_delta=cfg.training.predict_delta,
        )

        for run_n in range(n_rollout_traj):
            val_traj, grid = valset.get_traj(trajs[run_n])

            traj_x = {
                k: torch.from_numpy(
                    np.array([v[i] for i in range(window_len)])  # add window of inputs
                ).unsqueeze(0)
                for k, v in val_traj.items()
            }
            sample = {"x": traj_x, "grid": grid.unsqueeze(0)}
            x, grid, _, _ = sample

            pred_rollout = rollout_fn(model, x, grid, ts0=0)  # (t, bs, c, h, w)
            pred_rollout_dict = {k: [] for k in active_keys}

            for c, k in enumerate(active_keys):
                for t in range(rollout_steps):
                    xt_c = pred_rollout.squeeze()[t, c, ...].detach().cpu().numpy()
                    pred_rollout_dict[k].append(xt_c)

            pred_rollout = {
                k: np.array(pred_rollout_dict[k]).squeeze() for k in pred_rollout_dict
            }
            gt_rollout = {k: val_traj[k][:rollout_steps] for k in val_traj}
            # TODO pickle rollouts
            get_gifs3x3(pred_rollout, gt_rollout, f"{run_dir}_{run_n}")
