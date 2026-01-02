import sys

sys.path.append("../../..")
import os
import yaml
from argparse import ArgumentParser
import pickle
import torch
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from functools import partial
import time
import h5py

from huggingface_hub import login, hf_hub_download

auth_token = open(os.path.expanduser("~/.cache/huggingface/token"), "r").read()
login(auth_token)

from neugk.gyroswin.models import get_model
from neugk.utils import expand_as


def create_parser():
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
    # invert fft on spatial
    knth = np.moveaxis(x, 0, -1).copy()
    knth = knth.view(dtype=np.complex64).squeeze()
    # shift freqs to correct range
    knth = np.fft.fftn(knth, axes=(3, 4), norm="forward")
    knth = np.fft.ifftshift(knth, axes=(3,))
    knth = np.stack([knth.real, knth.imag]).squeeze().astype("float32")
    return knth


def invert_df(b_xt, cfg, parser):
    if cfg.dataset.separate_zf:
        zf = b_xt[:, :2]
        no_zf = b_xt[:, 2:]
        b_xt = zf + no_zf

    b_xt = b_xt.squeeze(0).cpu().numpy()
    if cfg.dataset.spatial_ifft:
        # apply fft if not done so already
        b_xt = invert_ifft(b_xt)

    return b_xt


parser = create_parser()
device = "cuda"
model_name = parser.ckpt.split("/")[-1]

# load inference snapshots from huggingface
snapshot_dir = hf_hub_download(
    repo_id="ml-jku/gyroswin_cbc_id_ood",
    filename=os.path.join("preprocessed", parser.eval_sim),
    repo_type="dataset",  # or "model" if that’s where you uploaded it
    token=True,
    cache_dir=parser.data_path,
)

# load normalization stats
norm_stats_dir = hf_hub_download(
    repo_id="ml-jku/gyroswin_cbc_id_ood",
    filename="normalization_stats.pkl",
    repo_type="dataset",
    token=True,
    cache_dir=parser.data_path,
)

norm_stats = pickle.load(open(norm_stats_dir, "rb"))
if "flux" not in norm_stats:
    norm_stats["flux"] = norm_stats["fluxavg"]
    del norm_stats["fluxavg"]
cyclone_name = parser.eval_sim.replace(".h5", "")
OUT_DIR = f"predictions/{model_name}/{cyclone_name}"
os.makedirs(OUT_DIR, exist_ok=True)

sd_path = hf_hub_download(
    repo_id=parser.ckpt, filename="pytorch_model.bin", cache_dir=parser.data_path
)
state_dict = torch.load(sd_path)

# instantiate dummy dataset required for model instantiation
cfg = OmegaConf.create(
    yaml.safe_load(open(f"configs/checkpoints/{model_name}/config.yaml", "r"))
)
input_fields = set(
    cfg.dataset.input_fields
    + [
        k
        for k in cfg.model.loss_weights.keys()
        if cfg.model.loss_weights[k] > 0.0 or cfg.model.loss_scheduler[k]
    ]
)

model_inputs = ["df"]
IDX_END = 263
params = {}
with h5py.File(snapshot_dir) as infile:
    k_name = "timestep_" + str(0).zfill(5)
    k = infile[f"data/{k_name}"][:]
    if cfg.dataset.separate_zf:
        nky = k.shape[-1]
        zf = np.repeat(k.mean(axis=-1, keepdims=True), repeats=nky, axis=-1)
        k = np.concatenate([zf, k - zf], axis=0)
    params["itg"] = infile["metadata/ion_temp_grad"][:]
    params["dg"] = infile["metadata/density_grad"][:]
    params["s_hat"] = infile["metadata/s_hat"][:]
    params["q"] = infile["metadata/q"][:]
    start_tstep = infile["metadata/timesteps"][:].item()
    delta = 1.2
    steps = IDX_END - cfg.dataset.offset
    end_tstep = start_tstep + steps * delta
    timesteps = torch.arange(start_tstep, end_tstep, delta).unsqueeze(0).to(device)
    resolution = infile["metadata/resolution"][:]
    phi_resolution = (
        resolution[3],
        resolution[2],
        resolution[4],
    )

params["timestep"] = timesteps[:, 0].cpu().numpy()
inputs = {"df": torch.tensor(k).unsqueeze(0).to(device, non_blocking=True)}
conds = {
    k: torch.tensor(params[k]).float().to(device, non_blocking=True)
    for k in params.keys()
}

cfg.dataset.resolution = tuple(i.item() for i in resolution)
cfg.dataset.phi_resolution = tuple(i.item() for i in phi_resolution)
model = get_model(cfg, dataset=cfg.dataset)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model = model.eval()

# load model checkpoint from huggingface
# doesnt work just yet due to missing keys in config.json
# model = GyroSwinMultitask.from_pretrained("ml-jku/gyroswin_small", cache_dir=parser.data_path)

fwd_time = []
shift_scale_dict = defaultdict(dict)
invert_fns = {
    "df": partial(invert_df, cfg=cfg, parser=parser),
    "phi": None,
    "flux": None,
    "fluxavg": None,
}

for key in input_fields:
    if cfg.dataset.normalization == "zscore":
        shift = torch.tensor(norm_stats[key]["full"]["mean"]).unsqueeze(0).to(device)
        scale = torch.tensor(norm_stats[key]["full"]["std"]).unsqueeze(0).to(device)
    elif cfg.dataset.normalization == "minmax":
        x_min = torch.tensor(norm_stats[key]["full"]["min"]).to(device)
        x_max = torch.tensor(norm_stats[key]["full"]["max"]).to(device)
        scale = (x_max - x_min) / cfg.dataset.beta1
        shift = x_min + scale * cfg.dataset.beta2
    shift_scale_dict[key]["shift"] = shift
    shift_scale_dict[key]["scale"] = scale

# Normalize the data
for key in model_inputs:
    shift = shift_scale_dict[key]["shift"]
    scale = shift_scale_dict[key]["scale"]
    inputs[key] = (inputs[key] - shift) / scale

with torch.no_grad():
    for idx in range(steps + 1):

        ts = timesteps[:, idx].to(device)
        conds["timestep"] = ts

        fwd_start = time.time()
        outputs = model(**inputs, **conds)

        # replace inputs with outputs for next timestep
        for key in model_inputs:
            inputs[key] = outputs[key].clone()

        fwd_end = time.time()
        fwd_time.append(fwd_end - fwd_start)

        for key in outputs.keys():
            scale = shift_scale_dict[key]["scale"]
            shift = shift_scale_dict[key]["shift"]
            if scale.ndim != outputs[key].ndim:
                scale = expand_as(scale, outputs[key].squeeze()).unsqueeze(0)
                shift = expand_as(shift, outputs[key].squeeze()).unsqueeze(0)
            b_xt = outputs[key] * scale + shift

            invert_fn = invert_fns[key]
            if invert_fn is not None:
                b_xt = invert_fn(b_xt)

            if isinstance(b_xt, torch.Tensor):
                b_xt = b_xt.cpu().numpy()

            b_xt = b_xt.astype("float64").reshape(-1, order="F")
            # dump to file
            if OUT_DIR:
                dirtarget = os.path.join(
                    OUT_DIR,
                    f"K{str((int(idx) + 1 + cfg.dataset.offset)).zfill(2)}",
                )
                os.makedirs(dirtarget, exist_ok=True)
                if key == "df":
                    ftarget = os.path.join(dirtarget, "FDS")
                elif key == "phi":
                    ftarget = os.path.join(dirtarget, "Poten")
                else:
                    # Predicted flux
                    ftarget = os.path.join(dirtarget, "flux")

                print(f"Writing file {ftarget}")
                if key in ["flux", "fluxavg"]:
                    np.savetxt(ftarget, b_xt)
                else:
                    with open(ftarget, "wb") as f:
                        f.write(b_xt)

print(f"Took {np.mean(fwd_time)}+/-{np.std(fwd_time)} seconds per forward pass")
