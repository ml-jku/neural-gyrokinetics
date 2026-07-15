from typing import Optional, Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from contextlib import nullcontext
from tqdm import tqdm
from math import log10

from neugk.pinc.neural_fields import (
    integral_losses,
    spectra_losses,
    CycloneNFDataset,
    CycloneNFDataLoader,
    sample_field,
)
from neugk.plot_utils import plot_nd, plot_diag
from neugk.physics.integrals import get_integrals
from neugk.pinc.eval.metrics import ml_eval


@torch.no_grad()
def nf_eval(
    model: nn.Module,
    data: CycloneNFDataset,
    device: torch.device,
    use_flux_fields: bool = False,
    spectral_loss_type: str = "l1",
):
    if data.ndim == 6:
        timesteps = list(range(data.grid.shape[0]))
    else:
        timesteps = [None]
    losses, ml = [], []
    for t in timesteps:
        pred_df = sample_field(model, data, device, timestep=t).to(device)
        gt_df = data.full_df[:, t] if t is not None else data.full_df
        int_losses, (pred_phi, gt_phi), (pred_eflux, gt_eflux) = integral_losses(
            pred_df,
            gt_df,
            geom=data.geom,
            device=device,
            use_flux_fields=use_flux_fields,
            timestep=t,
            return_fields=True,
        )
        spec_losses, _ = spectra_losses(
            pred_df=pred_df,
            pred_phi=pred_phi,
            pred_eflux=pred_eflux,
            gt_df=gt_df,
            gt_phi=gt_phi,
            gt_eflux=gt_eflux,
            ds=data.ds,
            spectral_loss_type=spectral_loss_type,
            mode_stds=getattr(data, "spectral_stds", None),
        )

        losses.append(int_losses | spec_losses)
        # psnr/eflux through the canonical eval definition (neugk.pinc.eval.metrics.ml_eval)
        # so the logged numbers match the eval pipeline: physical phi, per-snapshot max
        g_phi, (_, g_ef, _) = get_integrals(gt_df.to(device), data.geom, flux_fields=True)
        p_phi, (_, p_ef, _) = get_integrals(pred_df, data.geom, flux_fields=True)
        ml.append(ml_eval(pred_df, gt_df, p_phi, g_phi, p_ef, g_ef))
    losses = {k: sum([v[k] for v in losses]).item() / len(losses) for k in losses[0]}
    # mirror the eval pipeline: average per-snapshot psnr (each snapshot uses its own max)
    losses["df psnr"] = sum(m["psnr"] for m in ml) / len(ml)
    losses["phi psnr"] = sum(m["phi_psnr"] for m in ml) / len(ml)
    losses["eflux l1"] = sum(m["eflux_l1"] for m in ml) / len(ml)
    return losses


def train_density(
    model: nn.Module,
    n_epochs: int,
    data: CycloneNFDataset,
    loader: CycloneNFDataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    field_subsamples: Optional[Sequence[float]] = None,
    use_tqdm: bool = True,
    use_print: bool = True,
    eval_every: int = 2,
    use_compile: bool = True,
    use_amp: bool = False,  # bf16 autocast slows the small NF MLPs; on for large nets
):
    torch.set_float32_matmul_precision("high")
    if use_compile:
        model = torch.compile(model)
        eval_model = model._orig_mod  # unwrap for eval (bessel/abs Blackwell-safe)
    else:
        eval_model = model
    best_loss, best_model, best_e, train_losses = -torch.inf, None, 0, []
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    data.to(device)
    loader.to(device)
    model.to(device)
    for e in range(n_epochs):
        model.train()
        losses = {}
        if field_subsamples is not None:
            loader.subsample = field_subsamples[e]
        ploader = tqdm(loader, desc=f"Loss: {0.0:.6f}") if use_tqdm else loader
        # accumulate on-device; one host sync per epoch instead of per batch
        run_loss, n_batches = torch.zeros((), device=device), 0
        optim.zero_grad()
        for f, coords in ploader:
            with amp():
                pred_f = model(coords)
                loss = F.mse_loss(pred_f, f)
            loss.backward()
            optim.step()
            optim.zero_grad()
            run_loss += loss.detach()
            n_batches += 1
            if use_tqdm and n_batches % 50 == 0:
                ploader.set_description(f"Loss: {float(run_loss) / n_batches:.6f}")
        losses["train/loss"] = float(run_loss) / max(n_batches, 1)
        if sched is not None:
            sched.step()
        # eval (sample_field + flux/spectral losses) is ~the cost of a train
        # step, so only run it every eval_every epochs and on the last one.
        # eval_every <= 0 skips it entirely (density is just an MSE warmup; the
        # PINC phase does the physics-aware model selection). Sample with the
        # compiled model (~1.7x); the bessel/Blackwell issue is in FluxIntegral,
        # which nf_eval runs separately on the sampled tensors.
        if eval_every > 0 and (e % eval_every == 0 or e == n_epochs - 1):
            eval_losses = nf_eval(model, data, device=device, use_flux_fields=False)
            losses.update({f"val/{k}": v for k, v in eval_losses.items()})
            if losses["val/df psnr"] > best_loss:
                best_loss, best_e = losses["val/df psnr"], e
                best_model = deepcopy(eval_model)  # save uncompiled copy
        train_losses.append(losses)
        if use_print:
            str_losses = ", ".join([f"{k}: {float(v):.6f}" for k, v in losses.items()])
            print(f"[{e}] {str_losses}")
    if best_model is None:  # eval was skipped -> the final model is "best"
        best_model, best_e = deepcopy(eval_model), n_epochs - 1
    return eval_model, best_model, train_losses, best_e


def _make_fixed_weight():
    from conflictfree.weight_model import WeightModel

    class _FixedWeight(WeightModel):
        """Per-objective direction weights (vs ConFIG's default EqualWeight).

        Lets a flat-landscape objective (the integrated heat-flux scalar) bias
        the conflict-free direction more strongly than the dense field losses.
        """

        def __init__(self, weights):
            super().__init__()
            self.weights = weights

        def get_weights(self, gradients=None, losses=None, device=None):
            return self.weights.to(device)

    return _FixedWeight


_FixedWeight = None  # lazily set on first ConFIG build (needs conflictfree import)


def _build_config_op(
    op_name: str, length_name: str, use_lstsq: bool, dir_weights=None
):
    """Build a (operator, get_grad, apply_grad, dir_weights) ConFIG state.

    op_name: "config" | "pcgrad" | "imtlg". length_name (ConFIG only):
    "projection" | "min" | "max" | "harmonic" | "arithmetic" | "geometric".
    dir_weights: optional {objective_name: weight} biasing the conflict-free
    direction (ConFIG only; ignored by pcgrad/imtlg which have no weight model).
    """
    global _FixedWeight
    from conflictfree.grad_operator import (
        ConFIGOperator, PCGradOperator, IMTLGOperator)
    from conflictfree.length_model import (
        ProjectionLength, TrackMinimum, TrackMaximum,
        TrackHarmonicAverage, TrackArithmeticAverage, TrackGeometricAverage)
    from conflictfree.utils import get_gradient_vector, apply_gradient_vector

    if _FixedWeight is None:
        _FixedWeight = _make_fixed_weight()

    if op_name == "pcgrad":
        op = PCGradOperator()
    elif op_name == "imtlg":
        op = IMTLGOperator()
    else:
        lengths = {
            "projection": ProjectionLength, "min": TrackMinimum,
            "max": TrackMaximum, "harmonic": TrackHarmonicAverage,
            "arithmetic": TrackArithmeticAverage, "geometric": TrackGeometricAverage,
        }
        op = ConFIGOperator(
            length_model=lengths[length_name](), use_least_square=use_lstsq,
            # the simplified 2-grad path bypasses use_least_square and the
            # custom weight model wiring below; force the general path.
            allow_simplified_model=(dir_weights is None))
    return (op, get_gradient_vector, apply_gradient_vector, dir_weights)


def train_pinc(
    model: nn.Module,
    n_epochs: int,
    data: CycloneNFDataset,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    use_flux_fields: bool = False,
    pinc_loss_weight: Optional[Dict[str, float]] = None,
    use_print: bool = True,
    skip_eval: bool = False,
    eval_every: int = 2,
    use_config: bool = False,
    config_op: str = "config",
    config_length: str = "projection",
    config_lstsq: bool = True,
    config_weights: Optional[Dict[str, float]] = None,
    config_clip: Optional[float] = None,
    select: str = "phi",
    spectral_loss_type: str = "l1",
):
    if pinc_loss_weight is None:
        print("`pinc_loss_weight` not specified. Skipping.")
        return model, model, [], -1

    torch.set_float32_matmul_precision("high")
    best_loss, best_model, best_e, train_losses = -torch.inf, None, 0, []
    # model-selection score. "phi": legacy, max phi psnr only. "balanced":
    # minimise the summed relative degradation of df/phi/flux vs the warmup
    # state, so the conflict-free optimiser is not scored on phi alone (ConFIG
    # trajectories oscillate; the best-phi epoch can have a bad integrated flux).
    sel_ref = {}
    # ConFIG (tum-pbs): combine the objective gradients conflict-free instead of
    # summing weighted losses, so no manual loss-weight tuning is needed.
    config_state = None
    if use_config:
        config_state = _build_config_op(
            config_op, config_length, config_lstsq, config_weights)
    data.to(device)
    model.to(device)
    for e in range(n_epochs):
        model.train()
        losses = {}
        if data.ndim == 6:
            timesteps = list(range(data.grid.shape[0]))
        else:
            timesteps = [None]
        for t in timesteps:
            pred_df = sample_field(model, data, device, timestep=t).to(device)
            gt_df = data.full_df[:, t] if t is not None else data.full_df

            int_losses, (pred_phi, gt_phi), (pred_eflux, gt_eflux) = integral_losses(
                pred_df, gt_df, geom=data.geom, device=device,
                use_flux_fields=use_flux_fields, timestep=t, return_fields=True)
            int_losses = {
                f"{k} loss": pinc_loss_weight[k] * int_losses[f"{k} loss"]
                for k in pinc_loss_weight
                if pinc_loss_weight[k] != 0 and f"{k} loss" in int_losses}

            spec_losses, _ = spectra_losses(
                pred_df=pred_df, pred_phi=pred_phi, pred_eflux=pred_eflux,
                gt_df=gt_df, gt_phi=gt_phi, gt_eflux=gt_eflux, ds=data.ds,
                spectral_loss_type=spectral_loss_type,
                mode_stds=getattr(data, "spectral_stds", None))
            spec_losses = {
                f"{k} loss": pinc_loss_weight[k] * spec_losses[f"{k} loss"]
                for k in pinc_loss_weight
                if pinc_loss_weight[k] != 0 and f"{k} loss" in spec_losses}

            named = list(int_losses.items()) + list(spec_losses.items())
            components = [v for _, v in named]
            if config_state is not None and len(components) > 1:
                # per-objective gradients -> conflict-free combined direction.
                # none_grad_mode="zero" keeps the vectors equal-length (each loss
                # touches a different param subset); drop zero/non-finite grads
                # (e.g. an already-satisfied monotonicity term) so the unit-vector
                # normalization inside ConFIG does not divide by zero -> NaN.
                op, get_grad, apply_grad, dir_w = config_state
                grads, keys = [], []
                for i, (name_i, loss_i) in enumerate(named):
                    optim.zero_grad()
                    loss_i.backward(retain_graph=(i < len(named) - 1))
                    gv = get_grad(model, none_grad_mode="zero")
                    if torch.isfinite(gv).all() and gv.norm() > 0:
                        grads.append(gv)
                        keys.append(name_i.replace(" loss", ""))
                if len(grads) > 1:
                    # direction weights: how strongly each objective biases the
                    # conflict-free direction. Defaults to equal; `dir_w` lets a
                    # term (e.g. the flat integrated-flux scalar) pull harder.
                    if dir_w:
                        op.weight_model = _FixedWeight(
                            torch.tensor([dir_w.get(k, 1.0) for k in keys]))
                    g = op.calculate_gradient(grads)
                    if not torch.isfinite(g).all():
                        # numerical blow-up in lstsq/pinv -> fall back to sum
                        g = torch.stack(grads).sum(0)
                    elif config_clip is not None and g.norm() > 0:
                        # ConFIG's ProjectionLength inflates |g| to the SUM of
                        # projected gradient norms, which is dominated by the
                        # largest objective and gives steps far bigger than the
                        # weighted sum -> the flat integrated-flux scalar over-
                        # shoots and df forgets. Rescale the conflict-free
                        # DIRECTION to a baseline-scale magnitude (a multiple of
                        # the largest input gradient norm) so Adam steps match.
                        target = config_clip * max(gi.norm() for gi in grads)
                        g = g * (target / g.norm())
                    apply_grad(model, g)
                elif grads:
                    apply_grad(model, grads[0])
            else:
                optim.zero_grad()
                sum(components).backward()
            optim.step()
            if sched is not None:
                sched.step()

            losses.update({f"train/{k}": v.item() for k, v in int_losses.items()})
            losses.update({f"train/{k}": v.item() for k, v in spec_losses.items()})

        # evaluation (coarsened: ~as costly as a train step, see train_density)
        if not skip_eval and (e % eval_every == 0 or e == n_epochs - 1):
            eval_losses = nf_eval(
                model, data, device=device, use_flux_fields=False,
                spectral_loss_type=spectral_loss_type)
            losses.update({f"val/{k}": v for k, v in eval_losses.items()})
            if select == "balanced":
                # higher is better: negative summed relative degradation of the
                # three reported metrics (df recon, phi, integrated flux) vs the
                # first PINC eval. eps guards already-near-zero flux references.
                if not sel_ref:
                    sel_ref = {k: max(abs(eval_losses[k]), 1e-6)
                               for k in ("df loss", "phi loss", "flux loss")}
                score = -sum(abs(eval_losses[k]) / sel_ref[k] for k in sel_ref)
            else:
                score = losses["val/phi psnr"]
            if score > best_loss:
                best_loss, best_e = score, e
                best_model = deepcopy(model)

        train_losses.append(losses)
        if use_print:
            str_losses = ", ".join([f"{k}: {float(v):.6f}" for k, v in losses.items()])
            print(f"[{e}] {str_losses}")

    if best_model is None:
        best_model = model

    # TODO losses per epoch
    return model, best_model, train_losses, best_e


def train_nf(
    model: nn.Module,
    n_density_epochs: int,
    n_pinc_epochs: int,
    data: CycloneNFDataset,
    loader: CycloneNFDataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    aux_opt: torch.optim.Optimizer,
    aux_sched: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    field_subsamples: Optional[Sequence[float]] = None,
    use_flux_fields: bool = False,
    use_tqdm: bool = True,
    pinc_loss_weight: Optional[Dict[str, float]] = None,
    use_print: bool = True,
    skip_eval: bool = False,
    eval_every: int = 1,
):
    # density function training
    model_density, model_density_best, density_losses = train_density(
        model,
        optim=optim,
        sched=sched,
        n_epochs=n_density_epochs,
        data=data,
        loader=loader,
        device=device,
        field_subsamples=field_subsamples,
        use_tqdm=use_tqdm,
        use_print=use_print,
        eval_every=eval_every,
    )
    model_pinc = deepcopy(model_density)
    # update tracked params
    # finetune
    opt_kwargs = {
        k: v
        for k, v in aux_opt.defaults.items()
        if k in {"lr", "betas", "eps", "weight_decay", "amsgrad"}
    }
    optim = type(aux_opt)(model_pinc.parameters(), **opt_kwargs)
    sched = type(aux_sched)(aux_opt, **aux_sched.state_dict()["_hyperparam_defaults"])
    model_pinc, model_pinc_best, pinc_losses = train_pinc(
        model_pinc,
        n_epochs=n_pinc_epochs,
        data=data,
        optim=optim,
        sched=sched,
        device=device,
        use_flux_fields=use_flux_fields,
        pinc_loss_weight=pinc_loss_weight,
        use_print=use_print,
        skip_eval=skip_eval,
        eval_every=eval_every,
    )
    return (
        (model_density, model_pinc),
        (model_density_best, model_pinc_best),
        {"density": density_losses, "pinc": pinc_losses},
    )


@torch.no_grad()
def eval_diagnose(
    data: CycloneNFDataset,
    device: torch.device,
    model: Optional[nn.Module] = None,
    pred_df: Optional[torch.Tensor] = None,
    T: Optional[int] = None,
    use_spectral: bool = False,
    metrics_only: bool = False,
):
    if model is not None:
        model.to(device)
        pred_df = sample_field(model, data, device, timestep=T)
    pred_df = pred_df.to(device)
    gt_df = data.full_df.clone()
    if T is not None:
        gt_df = gt_df[:, T]
    gt_phi, (_, gt_eflux, _) = get_integrals(
        gt_df.to(device), data.geom, flux_fields=True, spectral_df=use_spectral
    )
    pred_phi, (pred_pflux, pred_eflux, _) = get_integrals(
        pred_df,
        data.geom,
        flux_fields=True,
        spectral_df=use_spectral,
    )
    # diagnostics
    spec_losses, (gt_diag, pred_diag) = spectra_losses(
        pred_df.cpu(),
        pred_phi.cpu(),
        pred_eflux.cpu(),
        gt_df.cpu(),
        gt_phi.cpu(),
        gt_eflux.cpu(),
        data.ds,
    )

    mse = F.mse_loss(pred_df.cpu(), gt_df.cpu())
    psnr = 10 * torch.log10(gt_df.max() ** 2 / mse)       # PSNR = 10 log10(max^2/MSE)
    phi_mse = F.mse_loss(pred_phi, gt_phi)
    phi_psnr = 10 * torch.log10(gt_phi.max() ** 2 / phi_mse)
    print(
        f"df nmse: {mse / (gt_df.cpu() ** 2).mean():.2f}, "
        f"df psnr: {psnr.item():.2f}\n"
        f"pflux: {pred_pflux.sum():.2f}, "
        f"eflux: {pred_eflux.sum():.2f}, gt eflux {gt_eflux.sum():.2f}\n"
        f"phi nmse: {phi_mse / (gt_phi ** 2).mean():.2f}, "
        f"phi psnr: {phi_psnr:.2f}\n"
        f"kyspec L1: {spec_losses['kyspec loss']:.2f}, "
        f"kyspec mono: {spec_losses['kyspec monotonicity loss']:.2f}\n"
        f"qspec L1: {spec_losses['qspec loss']:.2f}, "
        f"qspec mono: {spec_losses['qspec monotonicity loss']:.2f}\n"
    )
    # plots
    if not metrics_only:
        fig_df = plot_nd(pred_df.cpu().numpy(), gt_df.cpu().numpy())
        fig_eflux = plot_nd(pred_eflux.cpu().numpy(), gt_eflux.cpu().numpy())
        fig_potens = plot_nd(
            pred_phi.cpu().numpy(),
            gt_phi.cpu().numpy(),
            cmap="plasma",
            aspect=2,
            aggregate="slice",
        )
        fig_diag = plot_diag([gt_diag], [pred_diag], loglog=True)
        return fig_df, fig_eflux, fig_potens, fig_diag
