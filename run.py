from datetime import datetime

from functools import partial
import torch
import warnings
from tqdm import tqdm

from dataset import get_data
from models import get_model

from train import get_pushforward_trick, relative_norm_mse
from eval import get_rollout, validation_metrics, distribution_5D, plot4x4_sided
from utils import load_model_and_config, save_model_and_config


def runner(cfg, writer):
    (trainset, valset), (trainloader, valloader), augmentations = get_data(cfg)

    model = get_model(cfg)

    data_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    cfg.logging.run_name = f"{cfg.model.name}_{data_and_time}"

    device = cfg.device
    active_keys = cfg.dataset.active_keys

    opt_state_dict = None
    if cfg.load_ckp is True and cfg.ckpt_path is not None:
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

        for epoch in range(1, n_epochs + 1):
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

            log_metric_dict = {}
            val_plots = {}
            if epoch % cfg.validation.validate_every_n_epochs == 0:
                # Validation loop
                model.eval()
                # TODO configurable metric list
                metric_fn_list = {
                    "relative_norm_mse": partial(
                        relative_norm_mse, dim_to_keep=0
                    ),  # to average across all dimensions except timesteps
                    # TODO: add more useful metrics
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

                        n_eval_steps = cfg.validation.n_eval_steps

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
                            val_plots[
                                f"GT vs Pred at time {ts[0].item():.2f} (Linear Phase)"
                            ] = plot4x4_sided(
                                y[0, ...].to("cpu"),
                                x_rollout[0, 0, ...],  # first timestep and batch
                            )
                            val_plots[
                                f"Pred at time {ts[0].item():.2f} (Linear Phase)"
                            ] = distribution_5D(
                                x_rollout[0, 0, ...]  # first timestep and batch
                            )

                        elif idx == 5:
                            # TODO: make smarter (i.e. use timeindex when we output a dataclass from the dataset)
                            # metrics tensor will have shape [number_of_metrics, n_timesteps]
                            metrics += validation_metrics(
                                x_rollout, file_idx, valset, metric_fn_list
                            )
                            val_plots[
                                f"GT vs Pred at time {ts[0].item():.2f} (Saturated Phase)"
                            ] = plot4x4_sided(
                                y[0, ...].to("cpu"),
                                x_rollout[0, 0, ...],  # first timestep and batch
                            )
                            val_plots[
                                f"Pred at time {ts[0].item():.2f} (Saturated Phase)"
                            ] = distribution_5D(
                                x_rollout[0, 0, ...]  # first timestep and batch
                            )
                        else:
                            # metrics tensor will have shape [number_of_metrics, n_timesteps]
                            metrics += validation_metrics(
                                x_rollout, file_idx, valset, metric_fn_list
                            )

                metrics = metrics / len(valloader)

                for idx, metric_name in enumerate(metric_fn_list):
                    vals = metrics[idx, ...]
                    for t in range(n_eval_steps):
                        log_metric_dict["val/" + metric_name + f"x{t+1}"] = vals[t]

                if cfg.ckpt_path is not None:
                    warnings.warn(
                        "`cfg.ckpt_path` is not set: checkpoints will not be stored"
                    )
                    # Save model if validation loss improves
                    loss_val_min = save_model_and_config(
                        model,
                        opt,
                        cfg,
                        epoch,
                        # TODO decide target metric
                        val_loss=log_metric_dict["val/relative_norm_msex1"],
                        loss_val_min=loss_val_min,
                    )

            sched.step()

            # log to wandb
            epoch_logs = train_losses_dict | log_metric_dict
            if writer:
                # log epoch details
                if not val_plots:
                    writer.log(epoch_logs)
                else:
                    writer.log(epoch_logs, commit=False)
                    writer.log(val_plots)

            epoch_str = str(epoch).zfill(len(str(int(cfg.training.n_epochs))))
            print(
                f"Epoch: {epoch_str}, "
                f"{', '.join([f'{k}: {v:.5f}' for k, v in epoch_logs.items()])}"
            )
        if writer:
            writer.finish()

    if cfg.mode == "rollout":
        raise NotImplementedError("TODO")
