import sys
sys.path.append("..")
import os
from argparse import ArgumentParser
import pickle
import yaml
from omegaconf import OmegaConf
import torch
import numpy as np
from collections import defaultdict
from functools import partial
import time
import re
import h5py

from utils import load_model_and_config, expand_as
from models import get_model
from dataset.cyclone import CycloneDataset


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--data_path", default="/restricteddata/ukaea/gyrokinetics/preprocessed"
    )
    parser.add_argument("--eval_sim", default="iteration_13.h5")
    parser.add_argument("--onestep", action="store_true")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--ifft_merge", action="store_true")
    parser.add_argument("--predict_on_different", action="store_true")
    parser.add_argument("--start", type=int, default=0)
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

def invert_fluxfield(fluxfield, norm: str = "forward"):
    fluxfield = np.moveaxis(fluxfield.squeeze().cpu().numpy(), 0, -1).copy()
    fluxfield = fluxfield.view(dtype=np.complex64).squeeze()
    fluxfield = np.fft.fftn(fluxfield, axes=(-1,-2), norm=norm)
    fluxfield = np.fft.fftshift(fluxfield, axes=(-2,))
    return np.stack([fluxfield.real, fluxfield.imag]).astype("float32")

def modify_fds_dat(path):
    with open(path, "r") as infile:
        content = infile.read()
        content = content.replace("DTIM    =  1.000000000000000E-002", "DTIM    =  0.0")
        content = content.replace(
            "NT_REMAIN       =           0", "NT_REMAIN       =           1"
        )
        content = content.replace("TIME    =   319.999999999854     ", "TIME    =   0")

    with open(path, "w") as outfile:
        outfile.write(content)


def modify_input_dat(path):
    with open(path, "r") as infile:
        content = infile.read()
        content = content.replace("read_file = .false.", "read_file = .true.")
        content = content.replace("naverage = 40", "naverage = 1")
        content = content.replace("dtim = 0.01", "dtim = 0.0")
        content = content.replace("ntime = 800", "ntime = 1")
        content = content.replace("out3d_interval = 3", "out3d_interval = 1")
        content = content.replace("keep_dumps = .true.", "! keep_dumps = .true.")
        content = content.replace("ndump_ts = 3", "! ndump_ts = 3")

    with open(path, "w") as outfile:
        outfile.write(content)


def compute_pearson_correlation(x, y):
    # shape of [c, ...]
    n_channels = x.shape[0]
    x = x.reshape(n_channels, -1)
    y = y.reshape(n_channels, -1)
    x = x - torch.mean(x, dim=1, keepdim=True)
    y = y - torch.mean(y, dim=1, keepdim=True)
    cov = torch.sum(x * y, dim=1)
    std_x = torch.linalg.norm(x, dim=1)
    std_y = torch.linalg.norm(y, dim=1)
    return torch.mean(cov / (std_x * std_y))

def invert_df(b_xt, cfg, parser):
    if cfg.dataset.separate_zf:
        if parser.ifft_merge:
            zf = invert_ifft(b_xt.cpu().numpy().squeeze()[:2, ...])
            no_zf = invert_ifft(b_xt.cpu().numpy().squeeze()[2:, ...])
            b_xt = np.zeros_like(zf)
            b_xt[..., 0] = zf[..., 0]
            b_xt[..., 1:] = no_zf[..., :-1]
            b_xt = torch.tensor(np.expand_dims(b_xt, axis=0)).to(b_xt.device)
        else:
            zf = b_xt[:, :2]
            no_zf = b_xt[:, 2:]
            b_xt = zf + no_zf

    b_xt = b_xt.squeeze(0).cpu().numpy()
    if cfg.dataset.spatial_ifft and not parser.ifft_merge:
        # apply fft if not done so already
        b_xt = invert_ifft(b_xt)

    return b_xt

def invert_phi(b_xt):
    # invert ifft on spatial
    b_xt = np.moveaxis(b_xt.squeeze().cpu().numpy(), 0, -1).copy()
    b_xt = b_xt.view(dtype=np.complex64)
    spc = np.fft.fftn(b_xt, axes=(0, 2), norm="forward")
    spc = np.fft.fftshift(spc, axes=(0,))
    spc = np.stack([spc.real, spc.imag]).squeeze().astype("float32")
    return spc

def parse_input_dat(file_path):
    parsed_data = {}
    with open(file_path, "r") as file:
        content = file.read()
    # split the content by section headers (e.g., &SPECIES, &SPCGENERAL, etc.)
    sections = re.split(r"&\w+", content)
    # get all the headers by finding the section names
    section_headers = re.findall(r"&(\w+)", content)
    # remove comments
    sections = [
        section.strip() for section in sections if len(section) and section[0] != "!" and section.strip()
    ]
    for header, section in zip(section_headers, sections):
        section_dict = {}
        params = re.findall(r"(\w+)\s*=\s*([-\d\.e\w]+)", section)
        for param, value in params:
            try:
                section_dict[param] = (
                    float(value) if "e" in value or "." in value else int(value)
                )
            except ValueError:
                section_dict[param] = value.strip()
        while header in parsed_data:
            header = f"{header}0"
        parsed_data[header] = section_dict

    return parsed_data

parser = create_parser()
CKP = parser.ckpt
device = "cuda"

cfg = OmegaConf.create(yaml.safe_load(open(f"{CKP}/config.yaml", "r")))
train_losses = [k for k, v in cfg.model.loss_weights.items() if v > 0.0]
input_fields = set(cfg.dataset.input_fields + [k for k in cfg.model.loss_weights.keys()
                   if cfg.model.loss_weights[k] > 0.0 or cfg.model.loss_scheduler[k]])
traindata = CycloneDataset(
    path=parser.data_path,
    active_keys=cfg.dataset.active_keys,
    input_fields=input_fields,
    split="train",
    random_seed=cfg.seed,
    normalization=cfg.dataset.normalization,
    normalization_scope=cfg.dataset.normalization_scope,
    spatial_ifft=cfg.dataset.spatial_ifft,
    bundle_seq_length=cfg.model.bundle_seq_length,
    trajectories=cfg.dataset.training_trajectories,
    partial_holdouts=cfg.dataset.partial_holdouts,
    cond_filters=cfg.dataset.training_cond_filters,
    subsample=cfg.dataset.subsample,
    log_transform=cfg.dataset.log_transform,
    split_into_bands=cfg.dataset.split_into_bands,
    minmax_beta1=cfg.dataset.minmax_beta1,
    minmax_beta2=cfg.dataset.minmax_beta2,
    offset=cfg.dataset.offset,
    separate_zf=cfg.dataset.separate_zf,
    num_workers=cfg.dataset.num_workers,
    real_potens=cfg.dataset.real_potens,
)

# TODO: hardcoded eval_sim for iteration_13 now
eval_sim = ["iteration_13.h5"] if parser.predict_on_different else [parser.eval_sim]
data = CycloneDataset(
    active_keys=cfg.dataset.active_keys,
    input_fields=input_fields,
    path=parser.data_path,
    split="val",
    random_seed=cfg.seed,
    normalization=cfg.dataset.normalization,
    normalization_scope=cfg.dataset.normalization_scope,
    normalization_stats=traindata.norm_stats,
    spatial_ifft=cfg.dataset.spatial_ifft,
    bundle_seq_length=cfg.model.bundle_seq_length,
    trajectories=eval_sim,
    cond_filters=cfg.dataset.eval_cond_filters,
    subsample=cfg.dataset.subsample,
    log_transform=cfg.dataset.log_transform,
    split_into_bands=cfg.dataset.split_into_bands,
    minmax_beta1=cfg.dataset.minmax_beta1,
    minmax_beta2=cfg.dataset.minmax_beta2,
    offset=cfg.dataset.offset,
    separate_zf=cfg.dataset.separate_zf,
    num_workers=cfg.dataset.num_workers,
    real_potens=cfg.dataset.real_potens,
)

cyclone_name = parser.eval_sim.replace(".h5", "")
last = parser.last
IDX_0 = parser.start
ONESTEP = parser.onestep
ifft_merge = "_ifft_merge" if parser.ifft_merge else ""
OUT_DIR = f"{CKP}/{'onestep{}'.format(ifft_merge) if ONESTEP else 'autoreg_t{}{}'.format(IDX_0, ifft_merge)}/{cyclone_name}/{'best' if not last else 'ckp'}"
os.makedirs(OUT_DIR, exist_ok=True)
if "ood" in parser.eval_sim:
    raw_path = f"/restricteddata/ukaea/gyrokinetics/raw/ood/{parser.eval_sim.replace('.h5', '').replace("ood_", "").replace("_ifft", "")}"
else:
    raw_path = f"/restricteddata/ukaea/gyrokinetics/raw/{parser.eval_sim.replace('.h5', '')}"

assert traindata.norm_stats == data.norm_stats, "Normalization stats mismatch"
model = get_model(cfg, dataset=data)
path = f"{CKP}/best.pth" if not last else f"{CKP}/ckp.pth"
model, *_ = load_model_and_config(path, model, device)
model = model.to(device)
model = model.eval()

# TODO: make universal
model_inputs = ["df"]
if "ood" in parser.eval_sim:
    IDX_END = 263
    params = {}
    with h5py.File(f"/restricteddata/ukaea/gyrokinetics/preprocessed/{parser.eval_sim.replace('.h5', '_ifft_realpotens')}.h5") as infile:
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
    timesteps = traindata.get_timesteps(torch.tensor([0], dtype=torch.long), offset=cfg.dataset.offset)
    params["timestep"] = timesteps[:, 0].numpy()
    inputs = { "df": torch.tensor(k).unsqueeze(0).to(device, non_blocking=True) }
    conds = {k: torch.tensor(params[k]).float().to(device, non_blocking=True) for k in params.keys()}
else:
    IDX_END = len(data) - 2
    sample = data[IDX_0]
    conditioning = cfg.model.conditioning
    inputs = {k: getattr(sample, k).unsqueeze(0).to(device, non_blocking=True) for k in model_inputs if
              getattr(sample, k) is not None}
    conds = {k: getattr(sample, k).unsqueeze(0).to(device, non_blocking=True) for k in conditioning if
             getattr(sample, k) is not None}
    timesteps = data.get_timesteps(torch.tensor([0], dtype=torch.long), offset=cfg.dataset.offset)

if parser.predict_on_different:
    # we only use starting condition of iteration_13
    other_config = parse_input_dat(os.path.join(raw_path, "input.dat"))
    new_params = {
        "q": other_config["geom"]["q"],
        "s_hat": other_config["geom"]["shat"],
        "itg": other_config["species"]["rlt"],
        "dg": other_config["species"]["rln"],
    }
    conds = {k: torch.tensor([new_params[k]]).to(device, non_blocking=True) for k in new_params.keys()}

if not "ood" in parser.eval_sim:
    delta = (timesteps[:, 1:].squeeze() - timesteps[:, :-1].squeeze()).squeeze()[-1]
    if IDX_END > len(data) - 2:
        print("Extrapolating in time...")
        # add future timesteps, not observed during training
        timesteps = torch.cat(
            [
                timesteps,
                torch.arange(timesteps[:, -1].item() + delta, IDX_END, delta).unsqueeze(0),
            ],
            dim=1,
        )

files = []
gt_corr = {}
model_corr = {}
fwd_time = []
gt_corr = defaultdict(dict)
model_corr = defaultdict(dict)
shift_scale_dict = defaultdict(dict)
invert_fns = {
    "df": partial(invert_df, cfg=cfg, parser=parser),
    "phi": invert_phi if not cfg.dataset.real_potens else None,
    "fluxfield": invert_fluxfield,
    "flux": None,
    "fluxavg": None
}

for key in input_fields:
    if cfg.dataset.normalization == "zscore":
        shift = torch.tensor(traindata.norm_stats[key]["full"]["mean"]).unsqueeze(0).to(device)
        scale = torch.tensor(traindata.norm_stats[key]["full"]["std"]).unsqueeze(0).to(device)
    elif cfg.dataset.normalization == "minmax":
        x_min = torch.tensor(traindata.norm_stats[key]["full"]["min"]).to(device)
        x_max = torch.tensor(traindata.norm_stats[key]["full"]["max"]).to(device)
        scale = (x_max - x_min) / cfg.dataset.beta1
        shift = x_min + scale * cfg.dataset.beta2
    shift_scale_dict[key]["shift"] = shift
    shift_scale_dict[key]["scale"] = scale

if "ood" in  parser.eval_sim:
    # Normalize the data
    for key in model_inputs:
        inputs[key] = (inputs[key] - shift_scale_dict[key]["shift"]) / shift_scale_dict[key]["scale"]

with torch.no_grad():
    for idx in range(IDX_0, IDX_END + 1):

        ts = timesteps[:, idx].to(device)
        conds["timestep"] = ts

        if not "ood" in parser.eval_sim:
            # len(data) could be smaller than IDX_END
            sample = data[idx]
            gts_t = {k: getattr(sample, f"y_{k}").unsqueeze(0).to(device, non_blocking=True) for k in input_fields if
                     getattr(sample, f"y_{k}") is not None}

            if idx <= IDX_END or ONESTEP:
                sample = data[idx]
                inputs_t = {k: getattr(sample, k).unsqueeze(0).to(device, non_blocking=True) for k in model_inputs if
                          getattr(sample, k) is not None}

                for key in input_fields:
                    if key not in inputs_t:
                        continue
                    xt_gt = inputs_t[key]
                    yt = gts_t[key].squeeze()
                    if key == "df" and cfg.dataset.separate_zf:
                        xt_gt = xt_gt.squeeze()[:2] + xt_gt.squeeze()[2:]
                        yt = yt.squeeze()[:2] + yt.squeeze()[2:]
                    gt_corr[key][ts] = compute_pearson_correlation(xt_gt.squeeze(), yt)

        fwd_start = time.time()
        if ONESTEP:
            outputs = model(**inputs_t, **conds)
        else:
            outputs = model(**inputs, **conds)
            # replace inputs with outputs for next timestep
            for key in model_inputs:
                inputs[key] = outputs[key].clone()

        fwd_end = time.time()
        fwd_time.append(fwd_end - fwd_start)

        if idx <= IDX_END and "ood" not in parser.eval_sim:
            for key in outputs.keys():
                if key not in inputs:
                    continue
                yt = gts_t[key].squeeze()
                if key == "df" and cfg.dataset.separate_zf:
                    xt_merged = outputs[key].squeeze()[:2] + outputs[key].squeeze()[2:]
                    yt = yt.squeeze()[:2] + yt.squeeze()[2:]
                else:
                    xt_merged = outputs[key].squeeze()
                model_corr[key][ts] = compute_pearson_correlation(xt_merged, yt)

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
                dirtarget = os.path.join(OUT_DIR,
                                         f"K{str((int(idx) + 1 + cfg.dataset.offset) * cfg.dataset.subsample).zfill(2)}")
                os.makedirs(dirtarget, exist_ok=True)
                if key == "df":
                    ftarget = os.path.join(dirtarget, "FDS")
                elif key == "phi":
                    ftarget = os.path.join(dirtarget, "Poten")
                else:
                    # Predicted flux
                    ftarget = os.path.join(dirtarget, "flux")

                os.system(
                    f"cp {raw_path}/input.dat {dirtarget}")
                os.system(
                    f"cp {raw_path}/FDS.dat {dirtarget}")
                modify_fds_dat(f"{dirtarget}/FDS.dat")
                modify_input_dat(f"{dirtarget}/input.dat")
                write_mode = "wb" if key in ["df", "phi", "fluxfield"] else "w"
                with open(ftarget, write_mode) as f:
                    print(f"Writing file {ftarget}")
                    if key in ["flux", "fluxavg"]:
                        f.write(str(b_xt.item()))
                    else:
                        f.write(b_xt)

pickle.dump(model_corr, open(f"{OUT_DIR}/model_corr.pkl", "wb"))
pickle.dump(gt_corr, open(f"{OUT_DIR}/gt_corr.pkl", "wb"))
print(f"Took {np.mean(fwd_time)}+/-{np.std(fwd_time)} seconds per forward pass")
