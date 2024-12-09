from typing import Dict, List, Tuple, Callable, Union

from einops import rearrange
import torch
from torch import nn


def get_rollout(
    problem_dim: int,
    n_steps: int,
    bundle_steps: int,
    predict_delta: bool = False,
) -> Callable:
    # correct step size by adding last bundle
    n_steps_ = n_steps + bundle_steps - 1

    def _rollout(model: nn.Module, x0: torch.Tensor, ts0: torch.Tensor) -> torch.Tensor:
        xt = x0.clone()
        x_rollout = torch.zeros((xt.shape[0], problem_dim, n_steps_, *xt.shape[2:]))

        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i, ti in enumerate(range(0, n_steps, bundle_steps)):
                x_p = model(xt, timestep=(ts0 + ti))

                if predict_delta:
                    x_p = xt + x_p

                # concatenate rollout
                x_rollout[
                    :, :, i * bundle_steps : (i + 1) * bundle_steps, ...
                ] = x_p.cpu().unsqueeze(
                    2
                )  # TODO bundling?

        # only return desired size
        x_rollout = rearrange(x_rollout, "b c t ... -> t b c ...")
        return x_rollout[:n_steps, :, ...]

    return _rollout


def validation_metrics(
    rollout: torch.Tensor,
    file_idx: torch.Tensor,
    dataset,
    metrics_fns: Dict[str, Callable] = None,
) -> Union[
    Dict[str, List[float]],
    Tuple[Dict[str, List[float]], Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
]:
    assert (
        metrics_fns is not None
    ), "Pleas provide some metrics function for the validation metrics."

    n_steps = rollout.shape[0]

    # construct target y (NOTE: can use a lot of RAM with large n_steps)
    y = torch.stack(
        [dataset.get_at_time(file_idx, t)[2] for t in range(n_steps)], dim=0
    )
    metrics = torch.zeros((len(metrics_fns), n_steps))
    for idx, value in enumerate(metrics_fns.values()):
        value_result = value(rollout, y)
        metrics[idx, ...] = value_result

    return metrics
