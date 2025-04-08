from typing import Dict, Callable, Optional

from einops import rearrange
import torch
from torch import nn
from torch.utils.data import Dataset
from collections import defaultdict

from dataset.cyclone import CycloneSample


def get_rollout_fn(
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
        inputs: Dict,
        idx_data: Dict,
        conds: Dict,
    ) -> torch.Tensor:
        # cap the steps depending on the current max timestep
        rollout_steps = []
        for i, f_idx in enumerate(idx_data["file_index"].tolist()):
            ts_left = dataset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i])
            ts_left = ts_left // bundle_steps - 1
            rollout_steps.append(min(ts_left, n_steps))
        rollout_steps = min(rollout_steps)

        tot_ts = rollout_steps * bundle_steps
        problem_dim = model.module.problem_dim if hasattr(model, "module") else model.problem_dim
        inputs_t = inputs.copy()
        outputs = {}
        for key in inputs_t.keys():
            shape = inputs_t[key].shape
            dim = problem_dim if key != "phi" else 1
            if inputs_t[key].ndim in [5,7]:
                outputs[key] = torch.zeros((shape[0], dim, tot_ts, *shape[2:]))
            elif inputs_t[key].ndim in [6,8]:
                outputs[key] = torch.zeros((shape[0], dim, tot_ts, *shape[3:]))
            else:
                raise Exception(
                    "x should have 7 (b, c, v1, v2, s, x, y) "
                    "or 8 (b, c, t, v1, v2, s, x, y) dimensions!"
                )

        # get corresponding timesteps
        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, ts_step))
            for ts in idx_data["timestep_index"].tolist()
        ]
        fluxes = []
        tsteps = dataset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))
        use_bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i in range(0, rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    conds["timestep"] = tsteps[:, i].to(device)
                    output = model(**inputs_t, **conds)
                    if "flux" in output:
                        fluxes.append(output["flux"].cpu())
                        del output["flux"]

                    if predict_delta:
                        for key in output.keys():
                            output[key] = output[key] + inputs_t[key]

                    for key in output.keys():
                        inputs_t[key] = output[key].clone().float()

                for key in outputs.keys():
                    outputs[key][:, :, i * bundle_steps : (i + 1) * bundle_steps, ...] = (
                        output[key].cpu().unsqueeze(2) if output[key].ndim in [5,7] else output[key].cpu())

        for key in outputs.keys():
            # only return desired size
            outputs[key] = rearrange(outputs[key], "b c t ... -> t b c ...")
            outputs[key] = outputs[key][: rollout_steps * bundle_steps, :, ...]
        if len(fluxes) > 0:
            outputs["flux"] = rearrange(torch.cat(fluxes, dim=-1), "b t -> t 1 b")
        return outputs

    return _rollout


def validation_metrics(
    rollout: Dict,
    idx_data: Dict,
    bundle_steps: int,
    dataset,
    metrics_fns: Dict[str, Callable] = None,
    get_normalized: bool = False,
) -> Dict:
    assert (
        metrics_fns is not None
    ), "Pleas provide some metrics function for the validation metrics."
    assert_key = list(rollout.keys())[0]
    n_steps = rollout[assert_key].shape[0]
    # n_steps = rollout.shape[0] // bundle_steps
    # TODO: optimize: if valset is not shuffled, we can only return every second, since the next input is the previous' target (maybe handle in the dataset not sure)
    # construct target y (NOTE: can use a lot of RAM with large n_steps and takes a lot of time)
    gts = defaultdict(list)
    for t in range(0, n_steps, bundle_steps):
        sample: CycloneSample = dataset.get_at_time(
            idx_data["file_index"].long(), (idx_data["timestep_index"] + t).long(), get_normalized
        )
        for key in rollout.keys():
            gts[key].append(getattr(sample, f"y_{key}"))

    for key in gts.keys():
        if bundle_steps == 1:
            gts[key] = torch.stack(gts[key], dim=0)
            if gts[key].ndim <= 2:
                gts[key] = gts[key].unsqueeze(1)
        else:
            if gts[key].ndim > 2:
                gts[key] = torch.stack(gts[key], dim=2)
                gts[key] = rearrange(gts[key], "b c t ... -> t b c ...")
            else:
                gts[key] = torch.stack(gts[key], dim=1)
                gts[key] = rearrange(gts[key], "b t -> b 1 t")

    metrics = {k: torch.zeros((len(metrics_fns), n_steps)) for k in rollout.keys()}
    for idx, (name, fn) in enumerate(metrics_fns.items()):
        for key in rollout.keys():
            assert gts[key].shape == rollout[key].shape, f"Mismatch in shapes for {key}"
            value_result = fn(rollout[key], gts[key])
            metrics[key][idx, ...] = value_result
    return metrics
