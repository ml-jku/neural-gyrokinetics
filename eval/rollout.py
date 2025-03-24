from typing import Dict, Callable, Optional

from einops import rearrange
import torch
from torch import nn
from torch.utils.data import Dataset

from dataset.cyclone import CycloneSample


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
        phi: Optional[torch.Tensor] = None,
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
        if phi is not None:
            phit = phi.clone()
        if xt.ndim == 7:
            x_rollout = torch.zeros((xt.shape[0], problem_dim, tot_ts, *xt.shape[2:]))
            if phi is not None:
                phi_rollout = torch.zeros((xt.shape[0], 1, tot_ts, *phit.shape[2:]))
        elif xt.ndim == 8:
            x_rollout = torch.zeros((xt.shape[0], problem_dim, tot_ts, *xt.shape[3:]))
            if phi is not None:
                phi_rollout = torch.zeros((xt.shape[0], 1, tot_ts, *phit.shape[3:]))
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
        fluxes = []
        tsteps = dataset.get_timesteps(file_idx, torch.tensor(ts_idxs))
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i in range(0, rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    if phi is not None:
                        x_p, phi_p, flux = model(
                            xt, phit, timestep=tsteps[:, i].to(xt.device), itg=itg
                        )
                        fluxes.append(flux.cpu())
                    else:
                        x_p = model(xt, timestep=tsteps[:, i].to(xt.device), itg=itg)
                    if predict_delta:
                        x_p = xt + x_p
                    # update model input
                    xt = x_p.clone().float()
                    if phi is not None:
                        phit = phi_p.clone().float()
                # concatenate rollout
                x_rollout[:, :, i * bundle_steps : (i + 1) * bundle_steps, ...] = (
                    x_p.cpu().unsqueeze(2) if x_p.ndim == 7 else x_p.cpu()
                )
                if phi is not None:
                    phi_rollout[
                        :, :, i * bundle_steps : (i + 1) * bundle_steps, ...
                    ] = (phi_p.cpu().unsqueeze(2) if x_p.ndim == 7 else phi_p.cpu())

        # only return desired size
        x_rollout = rearrange(x_rollout, "b c t ... -> t b c ...")
        x_rollout = x_rollout[: rollout_steps * bundle_steps, :, ...]
        if phi is None:
            return x_rollout
        if phi is not None:
            phi_rollout = rearrange(phi_rollout, "b c t ... -> t b c ...")
            phi_rollout = phi_rollout[: rollout_steps * bundle_steps, :, ...]
            flux_rollout = rearrange(torch.cat(fluxes, dim=-1), "b t -> t 1 b")
            return x_rollout, phi_rollout, flux_rollout

    return _rollout


def validation_metrics(
    rollout: torch.Tensor,
    file_idx: torch.Tensor,
    ts_index: torch.Tensor,
    bundle_steps: int,
    dataset,
    metrics_fns: Dict[str, Callable] = None,
    phi_rollout: Optional[torch.Tensor] = None,
    flux_rollout: Optional[torch.Tensor] = None,
    get_normalized: bool = False,
) -> torch.Tensor:
    assert (
        metrics_fns is not None
    ), "Pleas provide some metrics function for the validation metrics."
    n_steps = rollout.shape[0]
    # n_steps = rollout.shape[0] // bundle_steps
    # TODO: optimize: if valset is not shuffled, we can only return every second, since the next input is the previous' target (maybe handle in the dataset not sure)
    # construct target y (NOTE: can use a lot of RAM with large n_steps and takes a lot of time)
    ys = []
    phis = []
    fluxes = []
    for t in range(0, n_steps, bundle_steps):
        sample: CycloneSample = dataset.get_at_time(
            file_idx.long(), (ts_index + t).long(), get_normalized
        )
        ys.append(sample.y)
        phis.append(sample.y_poten)
        fluxes.append(sample.y_flux)
    if bundle_steps == 1:
        y = torch.stack(ys, dim=0)
        phi = torch.stack(phis, dim=0)
        flux = torch.stack(fluxes, dim=0)
    else:
        # TODO??
        y = torch.cat(ys, dim=2)
        phi = torch.stack(phis, dim=2)
        flux = torch.stack(fluxes, dim=1)
        y = rearrange(y, "b c t ... -> t b c ...")
        phi = rearrange(phi, "b c t ... -> t b c ...")
        flux = rearrange(flux, "b t -> t b")

    metrics = torch.zeros((len(metrics_fns), n_steps))
    assert y.shape == rollout.shape
    for idx, (name, fn) in enumerate(metrics_fns.items()):
        if "phi" in name and phi_rollout is not None:
            value_result = fn(phi_rollout, phi)
        elif "flux" in name and flux_rollout is not None:
            value_result = fn(flux_rollout, flux.unsqueeze(1))
        else:
            value_result = fn(rollout, y)
        metrics[idx, ...] = value_result
    return metrics
