from typing import Dict, Callable

from einops import rearrange
import torch
from torch import nn
from torch.utils.data import Dataset


def get_rollout_fn(
    problem_dim: int,
    n_steps: int,
    bundle_steps: int,
    dataset: Dataset,
    predict_delta: bool = False,
    use_amp: bool = False,
    device: str = "cuda",
) -> Callable:
    # correct step size by adding last bundle
    # n_steps_ = n_steps + bundle_steps - 1

    def _rollout(
        model: nn.Module,
        x0: torch.Tensor,
        file_idx: torch.Tensor,
        ts_index_0: torch.Tensor,
        itg: torch.Tensor,
    ) -> torch.Tensor:
        # cap the steps depending on the current max timestep
        rollout_steps = []
        for i, f_idx in enumerate(file_idx.tolist()):
            ts_left = dataset.num_ts(int(f_idx)) - int(ts_index_0[i])
            ts_left = ts_left // bundle_steps - 1
            rollout_steps.append(min(ts_left, n_steps))
        rollout_steps = min(rollout_steps)

        tot_ts = rollout_steps * bundle_steps
        xt = x0.clone()
        if xt.ndim == 7:
            x_rollout = torch.zeros((xt.shape[0], problem_dim, tot_ts, *xt.shape[2:]))
        elif xt.ndim == 8:
            x_rollout = torch.zeros((xt.shape[0], problem_dim, tot_ts, *xt.shape[3:]))
        else:
            raise (
                "x should have 7 (b, c, v1, v2, s, x, y) "
                "or 8 (b, c, t, v1, v2, s, x, y) dimensions!"
            )

        # get corresponding timesteps
        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, ts_step))
            for ts in ts_index_0.tolist()
        ]
        tsteps = dataset.get_timesteps(file_idx, torch.tensor(ts_idxs))
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i in range(0, rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    x_p = model(xt, timestep=tsteps[:, i].to(xt.device), itg=itg)

                    if predict_delta:
                        x_p = xt + x_p
                    # update model input
                    xt = x_p.clone().float()
                # concatenate rollout
                x_rollout[:, :, i * bundle_steps : (i + 1) * bundle_steps, ...] = (
                    x_p.cpu().unsqueeze(2) if x_p.ndim == 7 else x_p.cpu()
                )

        # only return desired size
        x_rollout = rearrange(x_rollout, "b c t ... -> t b c ...")
        return x_rollout[: rollout_steps * bundle_steps, :, ...]

    return _rollout


def validation_metrics(
    rollout: torch.Tensor,
    file_idx: torch.Tensor,
    ts_index: torch.Tensor,
    bundle_steps: int,
    dataset,
    metrics_fns: Dict[str, Callable] = None,
    get_normalized: bool = False,
) -> torch.Tensor:
    assert (
        metrics_fns is not None
    ), "Pleas provide some metrics function for the validation metrics."
    n_steps = rollout.shape[0]
    # n_steps = rollout.shape[0] // bundle_steps
    # TODO: optimize: if valset is not shuffled, we can only return every second, since the next input is the previous' target (maybe handle in the dataset not sure)
    # construct target y (NOTE: can use a lot of RAM with large n_steps and takes a lot of time)
    if bundle_steps == 1:
        y = torch.stack(
            [
                dataset.get_at_time(file_idx.long(), (ts_index + t).long(), get_normalized).y
                for t in range(0, n_steps, bundle_steps)
            ],
            dim=0,
        )
    else:
        y = torch.concat(
            [
                dataset.get_at_time(file_idx.long(), (ts_index + t).long(), get_normalized).y
                for t in range(0, n_steps, bundle_steps)
            ],
            dim=2,
        )
        y = rearrange(y, "b c t ... -> t b c ...")

    metrics = torch.zeros((len(metrics_fns), n_steps))
    assert y.shape == rollout.shape
    for idx, value in enumerate(metrics_fns.values()):
        value_result = value(rollout, y)
        metrics[idx, ...] = value_result
    return metrics
