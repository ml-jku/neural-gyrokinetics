from typing import List, Callable

import torch
from torch import nn

import numpy as np

from dataset.cyclone import CycloneDataset


def relative_norm_mse(x, y, dim_to_keep=None):
    if dim_to_keep is None:
        y = y.flatten(1)
        diff = x.flatten(1) - y
        diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        diff_norms, y_norms = diff_norms**2, y_norms**2
        # sum over timesteps and mean over examples in batch
        return torch.mean(diff_norms / y_norms)
    else:
        # TODO: Check if this is necessary
        y = y.flatten(2)
        diff = x.flatten(2) - y
        diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        diff_norms, y_norms = diff_norms**2, y_norms**2
        dims = [i for i in range(len(y_norms.shape))][dim_to_keep + 1:]
        return torch.mean(diff_norms / y_norms, dim=dims)


def get_pushforward_trick(
    unrolls: List[int],
    probs: List[float],
    schecule: List[float],
    predict_delta: bool,
    dataset: CycloneDataset,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        x: torch.Tensor,
        ts: torch.Tensor,
        y: torch.Tensor,
        file_idx: torch.Tensor,
        epoch: int,
    ) -> List[float]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(schecule)).sum()
        # sample number of steps
        curr_probs = [p / sum(probs[:idx]) for p in probs[:idx]]
        unroll_steps = np.random.choice(unrolls[:idx], p=curr_probs)

        # cap the unroll steps depending on the current max timestep
        unroll_steps = min(
            [
                min(dataset.num_ts(f_idx) - int(ts[i]), unroll_steps)
                for i, f_idx in enumerate(file_idx.tolist())
            ]
        )

        if unroll_steps < 2:
            return x, ts, y

        with torch.no_grad():
            xt = x
            for i in range(unroll_steps - 1):
                x_p = model(x, timestep=(ts + i))

                if predict_delta:
                    x_p = x + x_p
            ts_unrolled = ts + unroll_steps - 1

            # TODO check if fetching correct target!
            # get unrolled y lazily (too large to load otherwise)
            _, _, y_unrolled, _ = dataset.get_at_time(file_idx, ts_unrolled.cpu())

        # have to clone xt to avoid view mode grad runtime error
        return xt.clone(), ts_unrolled, y_unrolled.to(x.device)

    return _loss_fn
