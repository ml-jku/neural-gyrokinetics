"""Utilities for running model inference from Hugging Face Hub."""

import sys
import os
import yaml
from argparse import ArgumentParser
import pickle
import time
from collections import defaultdict
from functools import partial

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from huggingface_hub import login, hf_hub_download

sys.path.append("../../..")
from neugk.gyroswin.models import get_model
from neugk.utils import expand_as, recombine_zf, separate_zf


def create_parser():
    """Configure command line argument parser for inference."""
    parser = ArgumentParser()
    parser.add_argument("--ckpt", default="ml-jku/gyroswin_small")
    parser.add_argument("--data_path", default="cache")
    parser.add_argument(
        "--eval_sim",
        default="iteration_262.h5",
        choices=[
            "iteration_262.h5",
            "iteration_135.h5",
            "iteration_8.h5",
            "iteration_232.h5",
            "iteration_148.h5",
            "iteration_115.h5",
            "ood_iteration_0.h5",
            "ood_iteration_1.h5",
            "ood_iteration_2.h5",
            "ood_iteration_3.h5",
            "ood_iteration_4.h5",
        ],
    )
    return parser.parse_args()


def invert_ifft(x):
    """Invert FFT operation on spatial dimensions to recover original representation."""
    knth = np.moveaxis(x, 0, -1).copy()
    knth = knth.view(dtype=np.complex64).squeeze()
    knth = np.fft.fftn(knth, axes=(3, 4), norm="forward")
    knth = np.fft.ifftshift(knth, axes=(3,))
    return np.stack([knth.real, knth.imag]).squeeze().astype("float32")


def invert_df(b_xt, cfg, parser):
    """Undo dataframe transformations and scaling."""
    if cfg.dataset.separate_zf:
        b_xt = recombine_zf(b_xt, dim=1)
    b_xt = b_xt.squeeze(0).cpu().numpy()
    if cfg.dataset.spatial_ifft:
        b_xt = invert_ifft(b_xt)
    return b_xt


def main():
    """Main execution block for Hugging Face Hub inference."""
    # login to hf
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            login(f.read().strip())

    parser = create_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = parser.ckpt.split("/")[-1]

    # download assets
    snapshot_dir = hf_hub_download(
        repo_id="ml-jku/gyroswin_cbc_id_ood",
        filename=os.path.join("preprocessed", parser.eval_sim),
        repo_type="dataset",
        token=True,
        cache_dir=parser.data_path,
    )

    norm_stats_dir = hf_hub_download(
        repo_id="ml-jku/gyroswin_cbc_id_ood",
        filename="normalization_stats.pkl",
        repo_type="dataset",
        token=True,
        cache_dir=parser.data_path,
    )

    # load normalization info
    with open(norm_stats_dir, "rb") as f:
        norm_stats = pickle.load(f)
    if "flux" not in norm_stats:
        norm_stats["flux"] = norm_stats.pop("fluxavg", None)

    # setup output
    cyclone_name = parser.eval_sim.replace(".h5", "")
    out_dir = f"predictions/{model_name}/{cyclone_name}"
    os.makedirs(out_dir, exist_ok=True)

    # download weights and config
    sd_path = hf_hub_download(
        repo_id=parser.ckpt, filename="pytorch_model.bin", cache_dir=parser.data_path
    )
    state_dict = torch.load(sd_path, map_location=device)

    with open(f"configs/checkpoints/{model_name}/config.yaml", "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # identify required fields
    input_fields = set(
        cfg.dataset.input_fields
        + [
            k
            for k, w in cfg.model.loss_weights.items()
            if w > 0.0 or cfg.model.loss_scheduler[k]
        ]
    )

    model_inputs = ["df"]
    idx_end = 263
    params = {}

    # load initial state
    with h5py.File(snapshot_dir, "r") as infile:
        k = infile["data/timestep_00000"][:]
        if cfg.dataset.separate_zf:
            k = separate_zf(k, dim=0)

        params["itg"] = infile["metadata/ion_temp_grad"][:]
        params["dg"] = infile["metadata/density_grad"][:]
        params["s_hat"] = infile["metadata/s_hat"][:]
        params["q"] = infile["metadata/q"][:]

        start_tstep = infile["metadata/timesteps"][:].item()
        resolution = infile["metadata/resolution"][:]

    # temporal grid setup
    delta = 1.2
    steps = idx_end - cfg.dataset.offset
    timesteps = (
        torch.arange(start_tstep, start_tstep + steps * delta, delta)
        .unsqueeze(0)
        .to(device)
    )

    # prepare tensors
    params["timestep"] = timesteps[:, 0].cpu().numpy()
    inputs = {"df": torch.tensor(k).unsqueeze(0).to(device, non_blocking=True)}
    conds = {
        k: torch.tensor(v).float().to(device, non_blocking=True)
        for k, v in params.items()
    }

    cfg.dataset.resolution = tuple(i.item() for i in resolution)
    cfg.dataset.phi_resolution = (
        resolution[3].item(),
        resolution[2].item(),
        resolution[4].item(),
    )

    # instantiate model
    model = get_model(cfg, dataset=cfg.dataset).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # prepare loop context
    fwd_time = []
    shift_scale_dict = defaultdict(dict)
    invert_fns = {
        "df": partial(invert_df, cfg=cfg, parser=parser),
        "phi": None,
        "flux": None,
        "fluxavg": None,
    }

    # compute normalization factors
    for key in input_fields:
        if cfg.dataset.normalization == "zscore":
            shift = torch.tensor(
                norm_stats[key]["full"]["mean"], device=device
            ).unsqueeze(0)
            scale = torch.tensor(
                norm_stats[key]["full"]["std"], device=device
            ).unsqueeze(0)
        elif cfg.dataset.normalization == "minmax":
            x_min = torch.tensor(norm_stats[key]["full"]["min"], device=device)
            x_max = torch.tensor(norm_stats[key]["full"]["max"], device=device)
            scale = (x_max - x_min) / cfg.dataset.beta1
            shift = x_min + scale * cfg.dataset.beta2
        shift_scale_dict[key] = {"shift": shift, "scale": scale}

    # normalize inputs
    for key in model_inputs:
        inputs[key] = (inputs[key] - shift_scale_dict[key]["shift"]) / shift_scale_dict[
            key
        ]["scale"]

    # iterative inference
    with torch.no_grad():
        for idx in range(steps + 1):
            conds["timestep"] = timesteps[:, idx].to(device)

            t0 = time.time()
            outputs = model(**inputs, **conds)

            # update loop state
            for key in model_inputs:
                inputs[key] = outputs[key].clone()

            fwd_time.append(time.time() - t0)

            # process and save results
            for key, val in outputs.items():
                scale = shift_scale_dict[key]["scale"]
                shift = shift_scale_dict[key]["shift"]
                if scale.ndim != val.ndim:
                    scale = expand_as(scale, val.squeeze()).unsqueeze(0)
                    shift = expand_as(shift, val.squeeze()).unsqueeze(0)

                b_xt = val * scale + shift
                if invert_fns[key]:
                    b_xt = invert_fns[key](b_xt)
                if isinstance(b_xt, torch.Tensor):
                    b_xt = b_xt.cpu().numpy()

                b_xt = b_xt.astype("float64").reshape(-1, order="F")

                dir_tgt = os.path.join(
                    out_dir, f"K{str(idx + 1 + cfg.dataset.offset).zfill(2)}"
                )
                os.makedirs(dir_tgt, exist_ok=True)

                f_tgt = os.path.join(
                    dir_tgt,
                    "FDS" if key == "df" else "Poten" if key == "phi" else "flux",
                )
                print(f"writing file {f_tgt}")

                if key in ["flux", "fluxavg"]:
                    np.savetxt(f_tgt, b_xt)
                else:
                    with open(f_tgt, "wb") as f:
                        f.write(b_xt)

    # log completion
    print(
        f"took {np.mean(fwd_time):.4f} +/- {np.std(fwd_time):.4f} seconds per forward pass"
    )


if __name__ == "__main__":
    main()
