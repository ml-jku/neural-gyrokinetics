from typing import Optional, Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm
from math import log10

from neugk.pinc.neural_fields import (
    get_integrals,
    integral_losses,
    spectra_losses,
    CycloneNFDataset,
    CycloneNFDataLoader,
    sample_field,
)
from neugk.pinc.neural_fields.nf_utils import plotND, plot_diag


@torch.no_grad()
def nf_eval(
    model: nn.Module,
    data: CycloneNFDataset,
    device: torch.device,
    use_flux_fields: bool = False,
):
    if data.ndim == 6:
        timesteps = list(range(data.grid.shape[0]))
    else:
        timesteps = [None]
    losses = []
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
        )

        losses.append(int_losses | spec_losses)
    losses = {k: sum([v[k] for v in losses]).item() / len(losses) for k in losses[0]}
    losses["df psnr"] = 10 * log10(gt_df.max().item() ** 2 / losses["df loss"] ** 2)
    losses["phi psnr"] = 10 * log10(gt_phi.max().item() ** 2 / losses["phi mse"] ** 2)
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
):
    torch.set_float32_matmul_precision("high")
    i = 0
    best_loss, best_model, train_losses = -torch.inf, None, []
    data.to(device)
    loader.to(device)
    model.to(device)
    for e in range(n_epochs):
        model.train()
        losses = {}

        if field_subsamples is not None:
            loader.subsample = field_subsamples[e]

        ll = []
        ploader = tqdm(loader, desc=f"Loss: {0.0:.6f}") if use_tqdm else loader
        for f, coords in ploader:
            pred_f = model(coords)

            # neural field loss
            loss = F.mse_loss(pred_f, f)

            optim.zero_grad()
            loss.backward()
            optim.step()
            ll.append(loss.item())
            i += 1
            if i > 50 and use_tqdm:
                ploader.set_description(f"Loss: {sum(ll) / len(ll):.6f}")
                i = 0

        losses["train/loss"] = sum(ll) / len(ll)

        if sched is not None:
            sched.step()

        # evaluation
        eval_losses = nf_eval(model, data, device=device, use_flux_fields=False)
        losses.update({f"val/{k}": v for k, v in eval_losses.items()})

        curr_loss = losses["val/df psnr"]

        if curr_loss > best_loss:
            best_loss, best_e = curr_loss, e
            best_model = deepcopy(model)

        train_losses.append(losses)
        if use_print:
            str_losses = ", ".join([f"{k}: {float(v):.6f}" for k, v in losses.items()])
            print(f"[{e}] {str_losses}")

    return model, best_model, train_losses, best_e


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
):
    if pinc_loss_weight is None:
        print("`pinc_loss_weight` not specified. Skipping.")
        return model, model, [], -1

    torch.set_float32_matmul_precision("high")
    best_loss, best_model, train_losses = -torch.inf, None, []
    data.to(device)
    model.to(device)
    for e in range(n_epochs):
        model.train()
        losses = {}
        # pinc training
        if data.ndim == 6:
            timesteps = list(range(data.grid.shape[0]))
        else:
            timesteps = [None]
        for t in timesteps:
            pred_df = sample_field(model, data, device, timestep=t).to(device)
            gt_df = data.full_df[:, t] if t is not None else data.full_df

            # spatial integral losses
            int_losses, (pred_phi, gt_phi), (pred_eflux, gt_eflux) = integral_losses(
                pred_df,
                gt_df,
                geom=data.geom,
                device=device,
                use_flux_fields=use_flux_fields,
                timestep=t,
                return_fields=True,
            )
            int_losses = {
                f"{k} loss": pinc_loss_weight[k] * int_losses[f"{k} loss"]
                for k in pinc_loss_weight
                if pinc_loss_weight[k] != 0 and f"{k} loss" in int_losses
            }

            # spectral and diagnostics losses
            spec_losses, _ = spectra_losses(
                pred_df=pred_df,
                pred_phi=pred_phi,
                pred_eflux=pred_eflux,
                gt_df=gt_df,
                gt_phi=gt_phi,
                gt_eflux=gt_eflux,
                ds=data.ds,
            )
            spec_losses = {
                f"{k} loss": pinc_loss_weight[k] * spec_losses[f"{k} loss"]
                for k in pinc_loss_weight
                if pinc_loss_weight[k] != 0 and f"{k} loss" in spec_losses
            }

            aux_loss = sum(int_losses.values()) + sum(spec_losses.values())

            optim.zero_grad()
            aux_loss.backward()
            optim.step()
            if sched is not None:
                sched.step()

            losses.update({f"train/{k}": v.item() for k, v in int_losses.items()})
            losses.update({f"train/{k}": v.item() for k, v in spec_losses.items()})

        # evaluation
        if not skip_eval:
            eval_losses = nf_eval(model, data, device=device, use_flux_fields=False)
            losses.update({f"val/{k}": v for k, v in eval_losses.items()})

            # TODO different scales, phi dominates...
            curr_loss = losses["val/phi psnr"]  # - 0.2 * losses["val/flux loss"]

            if curr_loss > best_loss:
                best_loss, best_e = curr_loss, e
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
    compile: bool = False,
    field_subsamples: Optional[Sequence[float]] = None,
    use_flux_fields: bool = False,
    use_tqdm: bool = True,
    pinc_loss_weight: Optional[Dict[str, float]] = None,
    use_print: bool = True,
    skip_eval: bool = False,
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
        torch.compile(model_pinc) if compile else model_pinc,
        n_epochs=n_pinc_epochs,
        data=data,
        optim=optim,
        sched=sched,
        device=device,
        use_flux_fields=use_flux_fields,
        pinc_loss_weight=pinc_loss_weight,
        use_print=use_print,
        skip_eval=skip_eval,
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
    psnr = 10 * torch.log10(gt_df.max() ** 2 / mse**2)
    phi_mse = F.mse_loss(pred_phi, gt_phi)
    phi_psnr = 10 * torch.log10(gt_phi.max() ** 2 / phi_mse**2)
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
        fig_df = plotND(pred_df.cpu().numpy(), gt_df.cpu().numpy())
        fig_eflux = plotND(pred_eflux.cpu().numpy(), gt_eflux.cpu().numpy())
        fig_potens = plotND(
            pred_phi.cpu().numpy(),
            gt_phi.cpu().numpy(),
            n=3,
            cmap="plasma",
            aspect=2,
            aggregate="slice",
        )
        fig_diag = plot_diag([gt_diag], [pred_diag], loglog=True)
        return fig_df, fig_eflux, fig_potens, fig_diag
