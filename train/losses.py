from typing import List, Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from .utils import roll_forward_once, unbind_time


def mse_timesteps(x, y):
    mse = F.mse_loss(x, y, reduction="none")
    # average over everything except the timesteps
    return torch.mean(mse, dim=[1, 2, 3, 4])


def relative_norm_mse(x, y):
    n_ts, bs = x.shape[0], x.shape[1]
    y = y[:n_ts]
    diff = x.reshape(n_ts, bs, -1) - y.reshape(n_ts, bs, -1)
    diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
    y_norms = torch.linalg.norm(y.reshape(n_ts, bs, -1), ord=2, dim=-1)
    diff_norms, y_norms = diff_norms**2, y_norms**2
    # sum over timesteps and mean over examples in batch
    return torch.mean(torch.sum(diff_norms / y_norms, dim=0))


def mae_timesteps(x, y):
    mae = F.l1_loss(x, y, reduction="none")
    # average over everything except the timesteps
    return torch.mean(mae, dim=[1, 2, 3, 4])


def autoencoder_loss(
    encoder: nn.Module,
    decoder: nn.Module,
    x: torch.Tensor,
    grid: torch.Tensor,
    ts: torch.Tensor,
    conditioner: Optional[nn.Module] = None,
) -> List[float]:
    if conditioner is not None:
        cond = conditioner(ts)

    grid = grid.flatten(1, 2)
    z = encoder(x, grid, condition=cond)
    pred_x = decoder(z, grid, condition=cond)

    lossz = F.mse_loss(pred_x, x)
    return lossz


def get_base_train_loss(
    predict_delta: bool, bundle_steps: Optional[int] = None
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        grid: torch.Tensor,
        ys: torch.Tensor,
        ts: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> List[float]:
        del epoch

        extra_steps = ys.shape[0] if bundle_steps is None else bundle_steps
        problem_dim = ys.shape[2]

        pred_x = model(x, grid, ts)
        if pred_x.ndim == 4:
            pred_x = unbind_time(pred_x, d=problem_dim)

        if predict_delta:
            x_ = unbind_time(x, d=problem_dim)
            tail_idx = torch.arange(x_.shape[1] - extra_steps, x_.shape[1])
            pred_x = pred_x + x_[:, tail_idx, ...]

        pred_x = pred_x.transpose(1, 0)

        loss = relative_norm_mse(pred_x, ys)

        return loss

    return _loss_fn


def get_pushforward_trick(
    unrolls: List[int],
    probs: List[float],
    schecule: List[float],
    predict_delta: bool,
    bundle_steps: Optional[int] = None,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        grid: torch.Tensor,
        ys: torch.Tensor,
        ts: torch.Tensor,
        epoch: int,
    ) -> List[float]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(schecule)).sum()
        # sample number of steps
        curr_probs = [p / sum(probs[:idx]) for p in probs[:idx]]
        unroll_steps = np.random.choice(unrolls[:idx], p=curr_probs)
        extra_steps = ys.shape[0] if bundle_steps is None else bundle_steps
        problem_dim = ys.shape[2]

        if unroll_steps < 2:
            return x, grid, ys, ts

        with torch.no_grad():
            xt = x
            for i in range(unroll_steps - 1):
                _, xt = roll_forward_once(
                    model, xt, grid, ts + i, problem_dim, extra_steps, predict_delta
                )

        ys_slice = ys[(unroll_steps - 1) * bundle_steps : unroll_steps * bundle_steps]
        # have to clone xt to avoid view mode grad runtime error
        return xt.clone(), grid, ys_slice, ts + unroll_steps

    return _loss_fn
