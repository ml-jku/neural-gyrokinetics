import os
import sys
import time
import itertools
from typing import Dict, Sequence, Optional
from collections import defaultdict
from copy import deepcopy
from queue import Queue

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.multiprocessing as mp
from tqdm import tqdm
from omegaconf import OmegaConf, ListConfig, DictConfig
from transformers.optimization import get_scheduler

sys.path.extend([".", ".."])

from neugk.pinc.neural_fields.nf_train import train_density, train_pinc
from neugk.pinc.neural_fields import CycloneNFDataset, CycloneNFDataLoader
from neugk.pinc.neural_fields.models import MLPNF, SIREN, WIRE
from neugk.pinc.neural_fields.nf_utils import ACTS

KY_MODES = {
    "base": None,
    "zfout": [0],
    "first2": [0, 1],
    "first5": [0, 1, 2, 3, 4, 5],
    "fancy1": [0, 1, 2, [3, 4, 5]],
    "fancy2": [0, 1, 2, [3, 4], [5, 6, 7, 8]],
}


def get_model(cfg: DictConfig, data: CycloneNFDataset):
    if cfg.name == "siren":
        return SIREN(
            data.ndim,
            data.nchannels,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            first_w0=cfg.first_w0,
            hidden_w0=cfg.hidden_w0,
            readout_w0=cfg.hidden_w0,
            skips=cfg.skips,
            embed_type=cfg.embed_type,
            clip_out=False,
        )
    if cfg.name == "wire":
        return WIRE(
            data.ndim,
            data.nchannels // 2,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            first_w0=cfg.first_w0,
            hidden_w0=cfg.hidden_w0,
            readout_w0=cfg.hidden_w0,
            complex_out=False,
            skips=cfg.skips,
            learnable_w0_s0=True,
        )
    if cfg.name == "mlp":
        return MLPNF(
            data.ndim,
            data.nchannels,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            act_fn=ACTS[cfg.act_fn],
            use_checkpoint=False,
            skips=cfg.skips,
            embed_type=cfg.embed_type,
        )
    raise ValueError(f"unknown model: {cfg.name}")


def run(
    cfg: DictConfig,
    trajectory: str,
    timestep: int,
    is_grid: bool = False,
    verbose: bool = True,
    shared_init: Optional[str] = None,
):
    timestep = timestep[0] if isinstance(timestep, Sequence) else timestep
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    data = CycloneNFDataset(
        trajectory,
        timesteps=timestep,
        normalize=cfg.normalization,
        normalize_coords="discrete" not in cfg.embed_type,
        separate_ky_modes=KY_MODES[cfg.ky_filter],
        flux_fields=cfg.use_flux_fields,
        realpotens=True,
    )
    loader = CycloneNFDataLoader(data, cfg.batch_size, preload=True, shuffle=True)

    model = get_model(cfg, data)
    if shared_init:
        model.load_state_dict(torch.load(shared_init))

    compression = data.full_df.nbytes / sum(p.nbytes for p in model.parameters())

    opt = optim.AdamW(model.parameters(), cfg.lr, weight_decay=1e-8)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, 1e-12)

    model, best_model, pre_losses, best_pre_epoch = train_density(
        model,
        n_epochs=cfg.epochs,
        data=data,
        loader=loader,
        device=device,
        field_subsamples=np.linspace(0.2, 1.0, cfg.epochs),
        opt=opt,
        sched=sched,
        use_tqdm=False,
        use_print=verbose,
    )
    model_pre, best_model_pre = deepcopy(model), deepcopy(best_model)

    pinc_epochs = cfg.pinc_epochs
    pinc_loss_weight = {
        "df": 1.0,
        "flux": 1.0,
        "phi": 1.0,
        "kyspec": 1.0,
        "qspec": 1.0,
        "kyspec monotonicity": 1.0,
        "qspec monotonicity": 1.0,
    }

    if hasattr(cfg, "physical_losses"):
        if not cfg.physical_losses:
            pinc_epochs = 0
        if "df" not in cfg.physical_losses:
            pinc_loss_weight.pop("df", None)
        if "int" not in cfg.physical_losses:
            pinc_loss_weight.pop("flux", None)
            pinc_loss_weight.pop("phi", None)
        if "diag" not in cfg.physical_losses:
            pinc_loss_weight.pop("kyspec", None)
            pinc_loss_weight.pop("qspec", None)
        if "mono" not in cfg.physical_losses:
            pinc_loss_weight.pop("kyspec monotonicity", None)
            pinc_loss_weight.pop("qspec monotonicity", None)

    model_pinc, best_model_pinc, pinc_losses, best_pinc_epoch = None, None, [], -1
    if pinc_epochs > 0:
        model_pinc = deepcopy(best_model)
        pinc_opt = torch.optim.AdamW(
            model_pinc.parameters(), cfg.pinc_lr, weight_decay=1e-12
        )
        pinc_sched = (
            get_scheduler(
                "cosine_with_min_lr",
                optimizer=pinc_opt,
                num_warmup_steps=pinc_epochs // 5,
                num_training_steps=pinc_epochs,
                scheduler_specific_kwargs={"min_lr": getattr(cfg, "min_lr", 1e-8)},
            )
            if cfg.pinc_lr_sched
            else None
        )

        model_pinc, best_model_pinc, pinc_losses, best_pinc_epoch = train_pinc(
            torch.compile(model_pinc),
            n_epochs=pinc_epochs,
            data=data,
            loader=loader,
            device=device,
            use_flux_fields=cfg.use_flux_fields,
            optim=pinc_opt,
            sched=pinc_sched,
            use_tqdm=False,
            use_print=verbose,
            pinc_loss_weight=pinc_loss_weight,
        )

    if not is_grid:
        os.makedirs(cfg.ckp_path, exist_ok=True)
        fname = trajectory.replace("_ifft", "").replace("_realpotens", "").split(".")[0]
        model_name = f"{cfg.name.lower()}_{fname}_t{timestep}_x{int(compression)}"

        torch.save(
            {"state_dict": model_pre.state_dict(), "cfg": cfg},
            f"{cfg.ckp_path}/{model_name}.pt",
        )
        torch.save(
            {"state_dict": best_model_pre.state_dict(), "cfg": cfg},
            f"{cfg.ckp_path}/best_{model_name}.pt",
        )
        if model_pinc:
            torch.save(
                {"state_dict": model_pinc.state_dict(), "cfg": cfg},
                f"{cfg.ckp_path}/int_{model_name}.pt",
            )
            torch.save(
                {"state_dict": best_model_pinc.state_dict(), "cfg": cfg},
                f"{cfg.ckp_path}/best_int_{model_name}.pt",
            )

    best_pre = pre_losses[best_pre_epoch]
    best_pre["CR"] = compression
    best_pinc = pinc_losses[best_pinc_epoch] if pinc_losses else {}
    if best_pinc:
        best_pinc["CR"] = compression

    return best_pre, best_pinc


def grid_worker(
    combo_cfg: DictConfig,
    trajectories: Sequence[str],
    timesteps: Sequence[int],
    gpu: int,
    return_dict: Dict,
    key: int,
):
    torch.cuda.set_device(int(gpu))
    acc_pre, acc_pinc = defaultdict(float), defaultdict(float)

    for traj in trajectories:
        for t in timesteps:
            m_pre, m_pinc = run(combo_cfg, traj, t, is_grid=True, verbose=False)
            for k, v in m_pre.items():
                acc_pre[k] += float(v)
            if m_pinc:
                for k, v in m_pinc.items():
                    acc_pinc[k] += float(v)

    nums = len(trajectories) * len(timesteps)
    avg_metrics = {f"pre_{k}": v / nums for k, v in acc_pre.items()}
    avg_metrics.update({f"pinc_{k}": v / nums for k, v in acc_pinc.items()})
    return_dict[key] = avg_metrics


def grid(cfg: DictConfig):
    grid_params = {
        k: v
        for k, v in cfg.items()
        if isinstance(v, ListConfig) and k not in ["timesteps", "trajectory", "gpus"]
    }
    fixed_params = {
        k: v
        for k, v in cfg.items()
        if not isinstance(v, ListConfig) or k in ["timesteps", "trajectory", "gpus"]
    }

    combinations = list(itertools.product(*grid_params.values()))
    timesteps = (
        cfg.timesteps
        if hasattr(cfg, "timesteps")
        else list(range(100, 100 + cfg.timeframe * cfg.coarse, cfg.coarse))
    )

    manager = mp.get_context("spawn").Manager()
    return_dict = manager.dict()

    gpu_queue = Queue()
    for gpu in cfg.gpus:
        gpu_queue.put(gpu)

    active, results = [], []
    pbar = tqdm(total=len(combinations), desc="grid search")

    for job_id, combo in enumerate(combinations):
        combo_cfg = OmegaConf.create(
            dict(fixed_params, **dict(zip(grid_params.keys(), combo)))
        )

        while len(active) >= len(cfg.gpus) * cfg.throttling:
            for p, gpu in active:
                if not p.is_alive():
                    p.join()
                    pbar.update()
                    gpu_queue.put(gpu)
            active = [(p, gpu) for p, gpu in active if p.is_alive()]
            time.sleep(1.0)

        gpu = gpu_queue.get()
        p = mp.get_context("spawn").Process(
            target=grid_worker,
            args=(combo_cfg, cfg.trajectory, timesteps, gpu, return_dict, job_id),
        )
        p.start()
        active.append((p, gpu))

    for p, _ in active:
        p.join()
        pbar.update()

    for combo_id, combo in enumerate(combinations):
        if combo_id in return_dict:
            results.append(
                {**dict(zip(grid_params.keys(), combo)), **return_dict[combo_id]}
            )

    pbar.close()

    grid_df = pd.DataFrame(results)
    print(grid_df)

    tag = (
        "_pinc"
        if hasattr(cfg, "physical_losses") and len(cfg.physical_losses) > 1
        else ""
    )
    tag += "_lora" if hasattr(cfg, "use_lora") and len(cfg.use_lora) > 1 else ""
    grid_df.to_csv(f"grid_search_{cfg.name}{tag}.csv", index=False)


def worker(cfg: DictConfig, traj: str, timesteps: Sequence, gpu: int):
    torch.cuda.set_device(int(gpu))
    shared_init = (
        f"nf_shared_init/{traj.replace('.h5', '')}.pth"
        if getattr(cfg, "use_shared_init", False)
        else None
    )
    for t in timesteps:
        run(cfg, traj, [int(t)], is_grid=False, verbose=False, shared_init=shared_init)


def main(cfg: DictConfig):
    timesteps = (
        cfg.timesteps
        if hasattr(cfg, "timesteps")
        else list(range(100, 100 + cfg.timeframe * cfg.coarse, cfg.coarse))
    )
    ctx = mp.get_context("spawn")

    gpu_queue = Queue()
    for gpu in cfg.gpus:
        gpu_queue.put(gpu)

    active = []
    pbar = tqdm(total=len(cfg.trajectory) * len(timesteps), desc="parallel evaluation")

    for traj in cfg.trajectory:
        for t_chunk in np.array_split(timesteps, cfg.throttling):
            while len(active) >= len(cfg.gpus) * cfg.throttling:
                for p, gpu in active:
                    if not p.is_alive():
                        p.join()
                        pbar.update()
                        gpu_queue.put(gpu)
                active = [(p, gpu) for p, gpu in active if p.is_alive()]
                time.sleep(1.0)

            gpu = gpu_queue.get()
            p = ctx.Process(target=worker, args=(cfg, traj, t_chunk, gpu))
            p.start()
            active.append((p, gpu))

    for p, _ in active:
        p.join()
        pbar.update()

    pbar.close()


if __name__ == "__main__":
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(
        OmegaConf.load(cli_cfg.get("config", "nf/eval.yaml")), cli_cfg
    )
    print("#" * 88)
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88)

    if cfg.mode == "default":
        main(cfg)
    elif cfg.mode == "grid":
        grid(cfg)
    else:
        raise NotImplementedError(cfg.mode)
