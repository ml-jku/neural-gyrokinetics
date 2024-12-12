from typing import List, Callable

import torch
from torch import nn
import numpy as np

from concurrent.futures import ThreadPoolExecutor

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
        dims = [i for i in range(len(y_norms.shape))][dim_to_keep + 1 :]
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
        ts_idx: torch.Tensor,
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
                min(dataset.num_ts(f_idx) - int(ts_idx[i]), unroll_steps)
                for i, f_idx in enumerate(file_idx.tolist())
            ]
        )

        if unroll_steps < 2:
            return x, ts, y

        # get timesteps for unrolling
        ts_idxs = [
            [i for i in range(int(ts_idx_start), int(ts_idx_start) + unroll_steps - 1)]
            for ts_idx_start in ts_idx.tolist()
        ]
        tsteps = dataset.get_timesteps_only(file_idx, torch.tensor(ts_idxs))

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(file_idx.cpu(), ts_unrolled.cpu())

        executor = ThreadPoolExecutor(max_workers=1)

        with torch.no_grad():
            ts_unrolled = ts_idx + unroll_steps - 1
            future = executor.submit(fetch_target, dataset, file_idx, ts_unrolled)

            xt = x
            for i in range(unroll_steps - 1):
                # TODO: currenlty only integer conditioning. Remove that line if floats are possible
                x_p = model(xt, timestep=torch.ceil(tsteps[:, i]).to(xt.device))
                if predict_delta:
                    x_p = xt + x_p
                xt = x_p.clone()
            # Get the result when needed
            unrolled = future.result()

        # have to clone xt to avoid view mode grad runtime error
        return xt.clone(), unrolled.timestep.to(x.device), unrolled.y.to(x.device)

    return _loss_fn
