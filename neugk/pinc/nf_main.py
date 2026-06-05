"""Neural-field training entry point.

Trains per-snapshot neural fields (density phase + optional PINC physics phase)
and writes checkpoints. Evaluation/metrics are NOT done here; that is handled by
the scalable eval package (``neugk.pinc.eval``). The only metrics produced are
the per-epoch training losses, used for `grid` hyperparameter ranking.

Modes:
- ``default``: train every (trajectory, timestep) across the GPU pool, save checkpoints.
- ``grid``:    train each hyperparameter combination, rank by training loss -> csv.
"""

import os
import sys
import time
import itertools
from typing import Dict, Optional, Sequence, Tuple
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

# PINC physics-loss terms, grouped by the `cfg.physical_losses` toggles.
PINC_LOSS_GROUPS = {
    "df": ["df"],
    "int": ["flux", "phi"],
    "diag": ["kyspec", "qspec"],
    "mono": ["kyspec monotonicity", "qspec monotonicity"],
}


def get_model(cfg: DictConfig, data: CycloneNFDataset):
    if cfg.name == "siren":
        return SIREN(
            data.ndim, data.nchannels, n_layers=cfg.n_layers, dim=cfg.dim,
            first_w0=cfg.first_w0, hidden_w0=cfg.hidden_w0, readout_w0=cfg.hidden_w0,
            skips=cfg.skips, embed_type=cfg.embed_type, clip_out=False,
            grid_size=data.grid_size,
        )
    if cfg.name == "wire":
        return WIRE(
            data.ndim, data.nchannels // 2, n_layers=cfg.n_layers, dim=cfg.dim,
            first_w0=cfg.first_w0, hidden_w0=cfg.hidden_w0, readout_w0=cfg.hidden_w0,
            complex_out=False, skips=cfg.skips, learnable_w0_s0=True,
            grid_size=data.grid_size,
        )
    if cfg.name == "mlp":
        return MLPNF(
            data.ndim, data.nchannels, n_layers=cfg.n_layers, dim=cfg.dim,
            act_fn=ACTS[cfg.act_fn], use_checkpoint=False, skips=cfg.skips,
            embed_type=cfg.embed_type, grid_size=data.grid_size,
        )
    raise ValueError(f"unknown model: {cfg.name}")


def build_data(
    cfg: DictConfig, trajectory: str, timestep: int
) -> Tuple[CycloneNFDataset, CycloneNFDataLoader]:
    kwargs = {}
    if hasattr(cfg, "path"):
        kwargs["path"] = cfg.path
    if hasattr(cfg, "backend"):
        kwargs["backend"] = cfg.backend
    data = CycloneNFDataset(
        trajectory,
        timesteps=timestep,
        normalize=cfg.normalization,
        normalize_coords="discrete" not in cfg.embed_type,
        norm_axes=tuple(getattr(cfg, "norm_axes", (-4,))),
        flux_fields=cfg.use_flux_fields,
        realpotens=True,
        **kwargs,
    )
    loader = CycloneNFDataLoader(data, cfg.batch_size, preload=True, shuffle=True)
    return data, loader


def pinc_loss_weights(cfg: DictConfig) -> Optional[Dict[str, float]]:
    """Active PINC loss weights, or None when the physics phase is disabled.

    No `physical_losses` key -> all terms active. Empty -> disabled (None).
    Otherwise keep the terms whose group is listed in `cfg.physical_losses`.
    """
    if not hasattr(cfg, "physical_losses"):
        return {k: 1.0 for g in PINC_LOSS_GROUPS.values() for k in g}
    if not cfg.physical_losses:
        return None
    keep = [k for g, ks in PINC_LOSS_GROUPS.items() if g in cfg.physical_losses for k in ks]
    return {k: 1.0 for k in keep} or None


def density_phase(cfg, model, data, loader, device, verbose):
    opt = optim.AdamW(model.parameters(), cfg.lr, weight_decay=1e-8)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, 1e-12)
    return train_density(
        model, n_epochs=cfg.epochs, data=data, loader=loader, optim=opt, sched=sched,
        device=device, field_subsamples=np.linspace(0.2, 1.0, cfg.epochs),
        use_tqdm=False, use_print=verbose,
    )


def pinc_phase(cfg, model, data, device, weights, verbose):
    opt = optim.AdamW(model.parameters(), cfg.pinc_lr, weight_decay=1e-12)
    sched = (
        get_scheduler(
            "cosine_with_min_lr", optimizer=opt,
            num_warmup_steps=cfg.pinc_epochs // 5, num_training_steps=cfg.pinc_epochs,
            scheduler_specific_kwargs={"min_lr": getattr(cfg, "min_lr", 1e-8)},
        )
        if cfg.pinc_lr_sched
        else None
    )
    # NB: no torch.compile here — the PINC loss path (complex FluxIntegral ops)
    # does not compile on Blackwell GPUs (nvrtc arch error); it runs eager.
    return train_pinc(
        model, n_epochs=cfg.pinc_epochs, data=data, optim=opt, sched=sched,
        device=device, use_flux_fields=cfg.use_flux_fields,
        pinc_loss_weight=weights, use_print=verbose,
    )


def save_checkpoints(cfg, trajectory, timestep, compression, state_dicts: Dict[str, dict]):
    os.makedirs(cfg.ckp_path, exist_ok=True)
    fname = trajectory.replace("_ifft", "").replace("_realpotens", "").split(".")[0]
    base = f"{cfg.name.lower()}_{fname}_t{timestep}_x{int(compression)}"
    for prefix, sd in state_dicts.items():
        torch.save({"state_dict": sd, "cfg": cfg}, f"{cfg.ckp_path}/{prefix}{base}.pt")


def train_run(
    cfg: DictConfig,
    trajectory: str,
    timestep: int,
    device: torch.device,
    save: bool = True,
    verbose: bool = True,
    shared_init: Optional[str] = None,
) -> Dict[str, float]:
    """Train one (trajectory, timestep): density phase then optional PINC phase.

    Saves `{,best_,int_,best_int_}<name>.pt` checkpoints when ``save``. Returns a
    flat dict of best-epoch training losses + compression ratio (for grid ranking).
    """
    timestep = timestep[0] if isinstance(timestep, Sequence) else timestep
    data, loader = build_data(cfg, trajectory, timestep)

    model = get_model(cfg, data)
    if shared_init:
        model.load_state_dict(torch.load(shared_init))
    compression = data.full_df.nbytes / sum(p.nbytes for p in model.parameters())

    model, best_model, density_losses, best_de = density_phase(
        cfg, model, data, loader, device, verbose
    )
    ckpts = {"": deepcopy(model).state_dict(), "best_": deepcopy(best_model).state_dict()}
    summary = {f"pre_{k}": float(v) for k, v in density_losses[best_de].items()}
    summary["CR"] = compression

    weights = pinc_loss_weights(cfg)
    if weights and cfg.pinc_epochs > 0:
        pinc_model = deepcopy(best_model)
        pinc_model, pinc_best, pinc_losses, best_pe = pinc_phase(
            cfg, pinc_model, data, device, weights, verbose
        )
        ckpts["int_"] = pinc_model.state_dict()
        ckpts["best_int_"] = pinc_best.state_dict()
        if pinc_losses:
            summary.update({f"pinc_{k}": float(v) for k, v in pinc_losses[best_pe].items()})

    if save:
        save_checkpoints(cfg, trajectory, timestep, compression, ckpts)
    return summary


# --------------------------------------------------------------------------- #
# multi-GPU orchestration                                                      #
# --------------------------------------------------------------------------- #

def _timesteps(cfg: DictConfig) -> Sequence[int]:
    if hasattr(cfg, "timesteps"):
        return cfg.timesteps
    return list(range(100, 100 + cfg.timeframe * cfg.coarse, cfg.coarse))


def worker(cfg: DictConfig, traj: str, timesteps: Sequence, gpu: int):
    torch.cuda.set_device(int(gpu))
    shared_init = (
        f"nf_shared_init/{traj.replace('.h5', '')}.pth"
        if getattr(cfg, "use_shared_init", False)
        else None
    )
    for t in timesteps:
        train_run(cfg, traj, [int(t)], torch.device(f"cuda:{gpu}"),
                  save=True, verbose=False, shared_init=shared_init)


def main(cfg: DictConfig):
    timesteps = _timesteps(cfg)
    ctx = mp.get_context("spawn")
    gpu_queue: Queue = Queue()
    for gpu in cfg.gpus:
        gpu_queue.put(gpu)

    active = []
    pbar = tqdm(total=len(cfg.trajectory) * len(timesteps), desc="training")
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


# --------------------------------------------------------------------------- #
# grid hyperparameter search (ranked by training loss)                         #
# --------------------------------------------------------------------------- #

def grid_worker(
    combo_cfg: DictConfig, trajectories: Sequence[str], timesteps: Sequence[int],
    gpu: int, return_dict: Dict, key: int,
):
    torch.cuda.set_device(int(gpu))
    device = torch.device(f"cuda:{gpu}")
    acc, n = defaultdict(float), 0
    for traj in trajectories:
        for t in timesteps:
            for k, v in train_run(combo_cfg, traj, t, device, save=False, verbose=False).items():
                acc[k] += v
            n += 1
    return_dict[key] = {k: v / max(n, 1) for k, v in acc.items()}


def grid(cfg: DictConfig):
    grid_params = {
        k: v for k, v in cfg.items()
        if isinstance(v, ListConfig) and k not in ["timesteps", "trajectory", "gpus"]
    }
    fixed = {
        k: v for k, v in cfg.items()
        if not isinstance(v, ListConfig) or k in ["timesteps", "trajectory", "gpus"]
    }
    combinations = list(itertools.product(*grid_params.values()))
    timesteps = _timesteps(cfg)

    return_dict = mp.get_context("spawn").Manager().dict()
    gpu_queue: Queue = Queue()
    for gpu in cfg.gpus:
        gpu_queue.put(gpu)

    active = []
    pbar = tqdm(total=len(combinations), desc="grid search")
    for job_id, combo in enumerate(combinations):
        combo_cfg = OmegaConf.create(dict(fixed, **dict(zip(grid_params.keys(), combo))))
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
    pbar.close()

    results = [
        {**dict(zip(grid_params.keys(), combo)), **return_dict[i]}
        for i, combo in enumerate(combinations)
        if i in return_dict
    ]
    grid_df = pd.DataFrame(results)
    print(grid_df)
    tag = "_pinc" if len(getattr(cfg, "physical_losses", [])) > 1 else ""
    tag += "_lora" if len(getattr(cfg, "use_lora", [])) > 1 else ""
    grid_df.to_csv(f"grid_search_{cfg.name}{tag}.csv", index=False)


if __name__ == "__main__":
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(OmegaConf.load(cli_cfg.get("config", "nf/eval.yaml")), cli_cfg)
    print("#" * 88)
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88)

    if cfg.mode == "default":
        main(cfg)
    elif cfg.mode == "grid":
        grid(cfg)
    else:
        raise NotImplementedError(cfg.mode)
