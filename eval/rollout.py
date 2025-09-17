from typing import Dict, Callable, Sequence

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
    use_bf16: bool = False,
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
        inputs_t = inputs.copy()
        preds = defaultdict(list)
        # get corresponding timesteps
        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + tot_ts, ts_step))
            for ts in idx_data["timestep_index"].tolist()
        ]
        fluxes = []
        tsteps = dataset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            # move bundles forward, rollout in blocks
            for i in range(0, rollout_steps):
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    conds["timestep"] = tsteps[:, i].to(device)
                    pred = model(**inputs_t, **conds)
                    if "flux" in pred:
                        fluxes.append(pred["flux"].cpu())
                        del pred["flux"]

                    if predict_delta:
                        for key in pred.keys():
                            pred[key] = pred[key] + inputs_t[key]

                    for key in inputs_t.keys():
                        if key in pred:
                            # Position is not in pred, and is constant
                            inputs_t[key] = pred[key].clone().float()

                for key in pred:
                    # add time dim if not there
                    preds[key].append(
                        pred[key].cpu().unsqueeze(2)
                        if pred[key].ndim in [4, 5, 7]
                        else pred[key].cpu()
                    )
        
        for key in preds.keys():
            # only return desired size
            if "position" in inputs_t:
                preds[key] = torch.cat([p.unsqueeze(2) for p in preds[key]], 2)
            else:
                preds[key] = torch.cat(preds[key], 2)
            preds[key] = rearrange(preds[key], "b c t ... -> t b c ...")
            preds[key] = preds[key][: rollout_steps * bundle_steps, :, ...]
        if len(fluxes) > 0:
            preds["flux"] = rearrange(torch.stack(fluxes, dim=-1), "b t -> t b")
        # to float32 for integrals etc
        # TODO is this the best approach?
        preds = {k: p.to(dtype=torch.float32) for k, p in preds.items()}
        return preds

    return _rollout


def validation_metrics(
    rollout: Dict,
    idx_data: Dict,
    bundle_steps: int,
    dataset,
    output_fields: Sequence[str],
    loss_wrap: nn.Module,
    get_normalized: bool = False,
    eval_integrals: bool = True,
) -> Dict:
    assert_key = list(rollout.keys())[0]
    n_steps = rollout[assert_key].shape[0]
    # n_steps = rollout.shape[0] // bundle_steps
    # TODO: optimize: if valset is not shuffled, we can only return every second, since
    #  the next input is the previous' target (maybe handle in the dataset not sure)
    # target y (NOTE: can use a lot of RAM with large n_steps and takes a lot of time)
    gts = defaultdict(list)
    for t in range(0, n_steps, bundle_steps):
        sample: CycloneSample = dataset.get_at_time(
            idx_data["file_index"].long(),
            (idx_data["timestep_index"] + t).long(),
            get_normalized,
        )
        for key in output_fields:
            gts[key].append(getattr(sample, f"y_{key}"))
        geometry = sample.geometry

    for key in gts.keys():
        if bundle_steps == 1:
            gts[key] = torch.stack(gts[key], 0)
        else:
            if gts[key].ndim > 2:
                gts[key] = rearrange(torch.stack(gts[key], 2), "b c t ... -> t b c ...")
            else:
                gts[key] = rearrange(torch.stack(gts[key], 1), "b t -> t b")

    for k in rollout:
        assert gts[k].shape == rollout[k].shape, f"Mismatch in shapes for {key}"
    metrics = {k: torch.zeros(n_steps) for k in loss_wrap.all_losses}
    integrated = []
    for n in range(n_steps):
        # one timestep at a time
        nth_rollout = {k: rollout[k][n] for k in rollout.keys()}
        nth_gts = {k: gts[k][n] for k in output_fields}
        # TODO proper flux plots?
        _, nth_losses, nth_integrated = loss_wrap(
            preds=nth_rollout,
            tgts=nth_gts,
            geometry=geometry,
            compute_integrals=eval_integrals,
        )
        integrated.append(nth_integrated)
        for k in nth_losses:
            metrics[k][n] = nth_losses[k]
    return metrics, integrated
