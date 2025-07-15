"""Loss wrapper, gradient balancer for multitask and pushforward loss."""

from typing import List, Callable, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.utils import clip_grad_norm_

from concurrent.futures import ThreadPoolExecutor

from conflictfree.grad_operator import ConFIGOperator
from conflictfree.momentum_operator import PseudoMomentumOperator
from conflictfree.utils import get_gradient_vector, OrderedSliceSelector

from neugk.dataset.cyclone import CycloneDataset, CycloneSample
from neugk.train.integrals import FluxIntegral


def relative_norm_mse(x, y, dim_to_keep=None, squared=True):
    assert x.shape == y.shape, "Mismatch in dimensions for computing loss"
    if x.ndim > 1:
        if dim_to_keep is None:
            x = x.flatten(1)
            y = y.flatten(1)
        else:
            # inference mode
            x = x.flatten(2)
            y = y.flatten(2)
    diff = x - y
    diff_norms = torch.linalg.norm(diff, ord=2, dim=-1)
    y_norms = torch.linalg.norm(y, ord=2, dim=-1)
    eps = 1e-8
    if squared:
        diff_norms, y_norms = diff_norms**2, y_norms**2
    if dim_to_keep is None:
        # sum over timesteps and mean over examples in batch
        return torch.mean(diff_norms / (y_norms + eps))
    else:
        dims = [i for i in range(len(y_norms.shape))][dim_to_keep + 1 :]
        return torch.mean(diff_norms / (y_norms + eps), dim=dims)


class LossWrapper(nn.Module):
    def __init__(
        self,
        weights: Dict,
        schedulers: Dict,
        denormalize_fn: Optional[Callable] = None,
        separate_zf: bool = False,
        real_potens: bool = False,
    ):
        super().__init__()
        self.weights = weights
        self._data_losses = ["df", "phi", "flux"]
        self._int_losses = ["flux_int", "phi_int", "flux_cross", "phi_cross"]
        self.integrator = FluxIntegral(real_potens=real_potens)
        self.denormalize_fn = denormalize_fn
        self.separate_zf = separate_zf
        self.schedulers = schedulers

    def integral_loss(
        self,
        geometry: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
        tgts: Dict[str, torch.Tensor],
        idx_data: Optional[Dict[str, torch.Tensor]] = None,
    ):
        assert self.denormalize_fn is not None
        assert geometry is not None
        if self.training:
            pred_df = []
            pred_phi = []
            tgt_phi = []
            tgt_eflux = []
            for b, f in enumerate(idx_data["file_index"].tolist()):
                assert "df" in preds, "Integral losses requires df (5D)."
                pred_df.append(self.denormalize_fn(f, df=preds["df"][b]))
                if "phi" in preds:
                    if preds["phi"].ndim == 3:
                        preds["phi"] = preds["phi"].unsqueeze(0)
                    pred_phi.append(self.denormalize_fn(f, phi=preds["phi"][b]))
                if tgts["phi"].ndim == 3:
                    tgts["phi"] = tgts["phi"].unsqueeze(0)
                tgt_phi.append(self.denormalize_fn(f, phi=tgts["phi"][b]))
                tgt_eflux.append(self.denormalize_fn(f, flux=tgts["flux"][b]))
            pred_df = torch.stack(pred_df)
            if len(pred_phi) > 0:
                pred_phi = torch.stack(pred_phi)
            else:
                pred_phi = None
            tgt_phi = torch.stack(tgt_phi)
            tgt_eflux = torch.stack(tgt_eflux)
        else:
            # already denormalized for evaluation
            pred_df = preds["df"]
            pred_phi = preds["phi"] if "phi" in preds else None
            tgt_phi = tgts["phi"]
            tgt_eflux = tgts["flux"]

        if self.separate_zf:
            # recompose zf
            pred_df = torch.cat(
                [pred_df[:, 0::2].sum(1, True), pred_df[:, 1::2].sum(1, True)], dim=1
            )

        pphi_int, (pflux, eflux, _) = self.integrator(geometry, pred_df, pred_phi)
        int_losses = {}
        # NOTE: these losses are in unnormalized space
        int_losses["phi_int"] = F.mse_loss(pphi_int, tgt_phi)
        # pflux -> 0, eflux -> heat flux
        int_losses["flux_int"] = (pflux**2).mean() + F.mse_loss(eflux, tgt_eflux)
        # mimicry / cross terms in the loss (between prediction heads and integrals)
        if "phi" in preds:
            int_losses["phi_cross"] = F.mse_loss(pred_phi, pphi_int)
        else:
            int_losses["phi_cross"] = 0.0
        if "flux" in preds:
            pred_eflux = preds["flux"] if "flux" in preds else None
            int_losses["flux_cross"] = F.mse_loss(pred_eflux, eflux)
        else:
            int_losses["flux_cross"] = 0.0

        return int_losses, {"phi": pphi_int, "pflux": pflux, "eflux": eflux}

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        tgts: Dict[str, torch.Tensor],
        idx_data: Optional[Dict[str, torch.Tensor]] = None,
        geometry: Optional[Dict[str, torch.Tensor]] = None,
        compute_integrals: bool = True,
        progress_remaining: float = 1.0,
        separate_zf: bool = False,
    ):
        losses = {}
        int_losses = {}
        # reset weight if scheduler is defined
        for key in self.schedulers.keys():
            if key in self.weights:
                self.weights[key] = self.schedulers[key](progress_remaining)
        # NOTE: network predicts phi -> weight["phi_int"] = 0 (otherwise summed twice)
        # only compute integrals if requested by weights or in eval
        do_ints = not self.training and compute_integrals
        if sum([self.weights.get(k, 0.0) for k in self._int_losses]) > 0 or do_ints:
            int_losses, integrated = self.integral_loss(geometry, preds, tgts, idx_data)
        loss_keys = (
            [k for k, w in self.weights.items() if w > 0.0]
            if self.training
            else list(set(self.weights.keys()).union(set(int_losses.keys())))
        )
        int_keys = [k for k in loss_keys if "int" in k]
        cross_keys = [k for k in loss_keys if "cross" in k]
        data_keys = list(set(loss_keys) - set(int_keys) - set(cross_keys))
        if not all([k in preds for k in data_keys]):
            # warnings.warn("Prediction - DATA loss weight key mismatch.")
            missing_keys = [k for k in data_keys if k not in preds]
            for k in missing_keys:
                preds[k] = torch.zeros_like(tgts[k]).to(tgts[k].device)
        if not all([k.replace("_cross", "") in preds for k in cross_keys]):
            raise ValueError("Prediction - CROSS loss weight key mismatch.")
        # compute losses
        for k in data_keys:
            if k in ["df", "phi"]:
                if k == "df" and separate_zf:
                    zf_loss = F.mse_loss(preds[k][:, :2], tgts[k][:, :2])
                    other_loss = relative_norm_mse(preds[k][:, 2:], tgts[k][:, 2:])
                    losses[k] = zf_loss + other_loss
                else:
                    if preds[k].shape != tgts[k].shape and k == "phi":
                        preds[k] = preds[k].unsqueeze(0)
                    losses[k] = relative_norm_mse(preds[k], tgts[k])
            else:
                losses[k] = F.mse_loss(preds[k], tgts[k])
        for k in int_keys + cross_keys:
            losses[k] = int_losses[k]
        if self.training:
            # reweight and accumulate
            loss = sum([self.weights[k] * losses[k] for k in loss_keys])
            # filter active losses
            losses = {k: losses[k] for k, w in self.weights.items() if w > 0.0}
            return loss, losses
        else:
            # no reweight in validation
            loss = sum([losses[k] for k in loss_keys])
            return loss, losses, integrated

    @property
    def active_losses(self):
        return [k for k in self.weights if self.weights[k] > 0.0]

    @property
    def all_losses(self):
        return list(self._data_losses) + self._int_losses

    def __len__(self):
        return len(self.all_losses)


class GradientBalancer(nn.Module):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str,
        scaler: torch.amp.GradScaler,
        clip_grad: bool = True,
        clip_to: float = 1.0,
        n_tasks: Optional[int] = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.mode = mode
        self.clip_grad = clip_grad
        self.scaler = scaler
        self.clip_to = clip_to
        if mode in [None, "none"]:
            pass
        # conflict free gradnorm
        if mode == "pseudo":
            self.operator = PseudoMomentumOperator(n_tasks)
            self.loss_selector = OrderedSliceSelector()
        if mode == "full":
            self.operator = ConFIGOperator()

    def forward(
        self, model: nn.Module, weighted_loss: torch.Tensor, losses: List[torch.Tensor]
    ):
        """Balances multitask gradients with conflict-free IG."""

        grads = []
        if self.mode in [None, "none"]:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(weighted_loss).backward()
        if self.mode == "pseudo":
            self.optimizer.zero_grad(set_to_none=True)
            idx, loss_i = self.loss_selector.select(1, losses)
            self.scaler.scale(loss_i).backward()
            self.operator.update_gradient(model, idx, grads=get_gradient_vector(model))
        if self.mode == "full":
            for loss_i in losses:
                self.optimizer.zero_grad(set_to_none=True)
                # retain graph for multiple backward passes
                self.scaler.scale(loss_i).backward(retain_graph=True)
                grads.append(get_gradient_vector(model, none_grad_mode="zero"))
            # apply conflict-free gradient directions
            self.operator.update_gradient(model, grads)

        # clipping
        if self.clip_grad:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(model.parameters(), self.clip_to)
        # gradient step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return model


def get_pushforward_fn(
    n_unrolls_schedule: List[int],
    probs_schedule: List[float],
    epoch_schedule: List[float],
    predict_delta: bool,
    dataset: CycloneDataset,
    bundle_steps: int,
    use_amp: bool = False,
    use_bf16: bool = False,
    device: str = None,
) -> Callable:
    def _loss_fn(
        model: nn.Module,
        inputs: Dict,
        gts: Dict,
        conds: Dict,
        idx_data: Dict,
        epoch: int,
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        # pushforward scheduler with epoch
        idx = (epoch > np.array(epoch_schedule)).sum()
        if not idx:
            return inputs, gts, conds

        # sample number of steps
        curr_probs = [p / sum(probs_schedule[:idx]) for p in probs_schedule[:idx]]
        pf_n_unrolls = np.random.choice(n_unrolls_schedule[:idx], p=curr_probs)

        # cap the unroll steps depending on the current max timestep
        n_unrolls = []
        for i, f_idx in enumerate(idx_data["file_index"].tolist()):
            sleft = (
                dataset.num_ts(int(f_idx)) - int(idx_data["timestep_index"][i])
            ) // bundle_steps - 1
            n_unrolls.append(min(sleft, pf_n_unrolls))
        n_unrolls = min(n_unrolls)

        if n_unrolls < 2:
            return inputs, gts, conds

        ts_step = bundle_steps
        ts_idxs = [
            list(range(int(ts), int(ts) + n_unrolls * ts_step, ts_step))
            for ts in idx_data["timestep_index"].tolist()
        ]
        tsteps = dataset.get_timesteps(idx_data["file_index"], torch.tensor(ts_idxs))

        # get unrolled target in a non-blocking way
        def fetch_target(dataset, file_idx, ts_unrolled):
            return dataset.get_at_time(
                file_idx.cpu(),
                ts_unrolled.cpu(),
            )

        executor = ThreadPoolExecutor(max_workers=1)
        with torch.no_grad():
            ts_unrolled = idx_data["timestep_index"] + (n_unrolls - 1) * ts_step
            future = executor.submit(
                fetch_target, dataset, idx_data["file_index"], ts_unrolled
            )

            inputs_t = inputs.copy()
            for i in range(n_unrolls - 1):
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    conds["timestep"] = tsteps[:, i].to(device)
                    outputs = model(**inputs_t, **conds)
                    if predict_delta:
                        for key in dataset.input_fields:
                            outputs[key] = outputs[key] + inputs[key]

                    for key in dataset.input_fields:
                        inputs_t[key] = outputs[key]

            # Get the result when needed
            unrolled: CycloneSample = future.result()

        gts = {
            k: getattr(unrolled, k).to(device, non_blocking=True)
            for k in gts.keys()
            if k is not None
        }
        conds = {
            k: getattr(unrolled, k).to(device, non_blocking=True)
            for k in conds.keys()
            if k is not None
        }
        return inputs_t, gts, conds

    return _loss_fn
