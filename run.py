from datetime import datetime

import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F

from dataset import get_data
from models import get_model
from eval.plot_utils import (
    get_wandb_tables,
)
from train import get_pushforward_trick, relative_norm_mse
from eval import get_rollout, validation_metrics, ssim_tensor
from utils import load_model_and_config, save_model_and_config


def runner(cfg, writer):
    (trainset, valset), (trainloader, valloader), augmentations = get_data(cfg)

    model = get_model(cfg)

    data_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    cfg.logging.run_name = f"{cfg.model.name}_{data_and_time}"

    device = cfg.device
    active_keys = cfg.dataset.active_keys

    opt_state_dict = None
    if cfg.ckpt_path is not None:
        # TODO move config loading to here (now in main.py)
        model, opt_state_dict, _ = load_model_and_config(
            cfg.ckpt_path, model=model, device=device
        )

    model = model.to(device)

    if cfg.mode == "train":
        n_epochs = cfg.training.n_epochs

        # optimizer config
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        if opt_state_dict is not None:
            opt.load_state_dict(opt_state_dict)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, 1e-6)

        # configure loss
        predict_delta = cfg.training.predict_delta
        # and pushforward
        pf_cfg = cfg.training.pushforward
        pushforward_fn = None
        if sum(pf_cfg.unrolls) > 0:
            pushforward_fn = get_pushforward_trick(
                pf_cfg.unrolls,
                pf_cfg.probs,
                schecule=pf_cfg.epochs,
                predict_delta=predict_delta,
                dataset=trainset,
            )

        loss_val_min = torch.inf
        use_tqdm = cfg.logging.tqdm

        for epoch in range(n_epochs):
            train_mse = 0
            if use_tqdm:
                trainloader = tqdm(trainloader, "Training")
            for sample in trainloader:
                x, ts, y = (
                    sample[0].to(device),
                    sample[1].to(device),
                    sample[2].to(device),
                )

                if augmentations is not None:
                    for aug_fn in augmentations:
                        x = aug_fn(x)

                if pushforward_fn:
                    # accessory information for pf (to retreive unrolled target)
                    file_idx = sample[-1]
                    x, ts, y = pushforward_fn(model, x, ts, y, file_idx, epoch)

                pred_x = model(x, timestep=ts)

                if predict_delta:
                    pred_x = x + pred_x

                loss = relative_norm_mse(pred_x, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_mse += loss.item()

            train_mse /= len(trainloader)
            train_losses_dict = {"train/relative_norm_mse": train_mse}

            # Validation loop
            model.eval()
            # TODO configurable metric list
            metric_fn_list = {
                "mse": F.mse_loss,
                "ssim": ssim_tensor,
            }
            # TODO initialize dictionary with torch tensors initialized to 0 and without grad with 1 dimension of size tsteps
            # OR initialize tensor of metric with shape [n_metrics, n_timesteps]
            # metric_dict = defaultdict(lambda: 0.0)
            if use_tqdm:
                valloader = tqdm(valloader, "Validation")
            with torch.no_grad():
                for idx, sample in enumerate(valloader):
                    x, ts, y, file_idx = (
                        sample[0].to(device),
                        sample[1].to(device),
                        sample[2].to(device),
                        sample[3],
                    )

                    # cap eval steps
                    n_eval_steps = min(
                        [
                            min(
                                valset.num_ts(f_idx) - int(ts[i]),
                                cfg.validation.n_eval_steps,
                            )
                            for i, f_idx in enumerate(file_idx.tolist())
                        ]
                    )

                    # get the rolled out validation trajectories
                    x_rollout = get_rollout(
                        problem_dim=len(active_keys),
                        n_steps=n_eval_steps,
                        bundle_steps=cfg.model.bundle_seq_length,
                        predict_delta=cfg.training.predict_delta,
                    )(model, x, ts0=ts)

                    if idx == 0:
                        # initialize metrics tensor
                        metrics = validation_metrics(
                            x_rollout, file_idx, valset, metric_fn_list
                        )
                        # TODO reintroduce
                        # frequencies, power_pred, power_gt = get_power_spectra(
                        #     x_rollout, y
                        # )
                        # get images of trajectories
                        # traj_plots = get_val_trajectory_plots(
                        #     x_rollout, y, [1, 5, 10, 25, 50]
                        # )

                    else:
                        # get all validation metrics for the rollout
                        # metrics tensor will have shape [number_of_metrics, n_timesteps]
                        metrics += validation_metrics(
                            x_rollout, file_idx, valset, metric_fn_list
                        )
                        # _, _power_pred, _power_gt = get_power_spectra(x_rollout, y)
                        # power_pred += _power_pred
                        # power_gt += _power_gt

            metrics = metrics / len(valloader)
            # power_pred = power_pred / len(valloader)
            # power_gt = power_gt / len(valloader)

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
                opt,
                cfg,
                epoch,
                # TODO decide target metric
                val_loss=log_metric_dict["val/msex1"],
                loss_val_min=loss_val_min,
            )

            sched.step()

            # log to wandb
            epoch_logs = train_losses_dict | log_metric_dict
            if writer:
                # log epoch details
                writer.log(epoch_logs, commit=False)
                # log validation trajectory images
                # images = [wandb.Image(graphic) for graphic in traj_plots]
                # writer.log(
                #     {k: images[idx] for idx, k in enumerate(active_keys)},
                #     commit=False,
                # )
                # log metrics graphs over full validation rollouts
                writer.log(
                    {
                        key: wandb.plot.line(table, "t", key, title=f"{key}test")
                        for key, table in metric_tables.items()
                    },
                    commit=False,
                )
                # log power spectra
                # writer.log(
                #     {
                #         "power spectra": wandb.plot.line_series(
                #             xs=frequencies,
                #             ys=[power_pred, power_gt],
                #             keys=["predicted", "groundtruth"],
                #             title="Power Spectrum",
                #             xname="frequency",
                #         )
                #     }
                # )

            epoch_str = str(epoch).zfill(len(str(int(cfg.training.n_epochs))))
            print(
                f"Epoch: {epoch_str}, "
                f"{', '.join([f'{k}: {v:.5f}' for k, v in epoch_logs.items()])}"
            )
        if writer:
            writer.finish()

    if cfg.mode == "rollout":
        raise NotImplementedError("TODO")
