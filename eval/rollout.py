from typing import Dict, List, Tuple, Callable, Union

import einops
import torch
from torch import nn


def get_rollout(
    problem_dim: int,
    n_steps: int,
    bundle_steps: int,
    predict_delta: bool = False,
) -> Callable:
    # correct step size by adding last bundle
    n_steps_ = n_steps + bundle_steps

    def _rollout(
        model: nn.Module, x0: torch.Tensor, grid: torch.Tensor, ts0: torch.Tensor
    ) -> torch.Tensor:
        xt = x0.clone()
        bs, _, h, w = xt.shape
        x_rollout = torch.zeros((bs, n_steps_, problem_dim, h, w))

        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i, ti in enumerate(range(0, n_steps, bundle_steps)):
                x_p = model(xt, grid, ts0 + ti)

                if x_p.ndim == 4:
                    x_p = einops.rearrange(
                        x_p, "bs (d t) h w -> bs t d h w", d=problem_dim
                    )

                xt = einops.rearrange(xt, "bs (d t) h w -> bs t d h w", d=problem_dim)

                # shift input window forward
                head_idx = torch.arange(0, xt.shape[1] - bundle_steps)
                tail_idx = torch.arange(bundle_steps, xt.shape[1])
                tgt_idx = torch.arange(xt.shape[1] - bundle_steps, xt.shape[1])
                xt[:, head_idx, ...] = xt[:, tail_idx, ...]
                if predict_delta:
                    # add to last bundle if predicting deltas
                    x_p = x_p + xt[:, tgt_idx, ...]
                xt[:, tgt_idx, ...] = x_p
                # concatenate rollout
                x_rollout[:, i * bundle_steps : (i + 1) * bundle_steps, ...] = x_p
                # flatten for next model call
                xt = xt.flatten(1, 2)

        # only return desired size
        x_rollout = x_rollout.permute(1, 0, 2, 3, 4)
        return x_rollout[:n_steps, :, ...]

    return _rollout


def validation_metrics(
    rollout: torch.Tensor,
    ys: torch.Tensor,
    metrics_fns: Dict[str, Callable] = None,
) -> Union[
    Dict[str, List[float]],
    Tuple[Dict[str, List[float]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
]:
    assert (
        metrics_fns is not None
    ), "Pleas provide some metrics function for the validation metrics."

    metrics = torch.zeros((len(metrics_fns), ys.shape[0]))
    for idx, value in enumerate(metrics_fns.values()):
        value_result = value(rollout, ys)
        metrics[idx, ...] = value_result

    return metrics
