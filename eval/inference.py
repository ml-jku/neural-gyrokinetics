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

from utils import load_model_and_config
from models import get_model
from dataset.cyclone import CycloneDataset


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--data_path", default="/restricteddata/ukaea/gyrokinetics/preprocessed"
    )
    parser.add_argument("--eval_sim", default="cyclone4_2_2.h5")
    parser.add_argument("--onestep", action="store_true")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--ifft_merge", action="store_true")
    parser.add_argument("--start_idx", default=0, type=int)
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


def modify_fds_dat(path):
    with open(path, "r") as infile:
        content = infile.read()
        content = content.replace("DTIM    =  2.000000000000000E-002", "DTIM    =  0.0")
        content = content.replace(
            "NT_REMAIN       =           0", "NT_REMAIN       =           1"
        )
        content = content.replace("TIME    =   192.753733197446     ", "TIME    =   0")

    with open(path, "w") as outfile:
        outfile.write(content)


def modify_input_dat(path):
    with open(path, "r") as infile:
        content = infile.read()
        content = content.replace("READ_FILE  = .false.", "READ_FILE  = .true.")
        content = content.replace("DTIM   = 0.02", "DTIM   = 0.0")
        content = content.replace("out3d_interval = 3", "out3d_interval = 1")
        content = content.replace("keep_dumps = .true.", "! keep_dumps = .true.")
        content = content.replace("ndump_ts = 3", "! ndump_ts = 3")

    with open(path, "w") as outfile:
        outfile.write(content)


def compute_pearson_correlation(x, y):
    # shape of [c, ...]
    n_channels = x.shape[0]
    x = x.view(n_channels, -1)
    y = y.view(n_channels, -1)
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
            b_xt = torch.tensor(np.expand_dims(b_xt, axis=0)).to(xt.device)
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
    b_xt = np.moveaxis(b_xt.squeeze(), 0, -1).copy()
    b_xt = b_xt.view(dtype=np.complex64)
    spc = np.fft.fftn(b_xt, axes=(0, 2), norm="forward")
    spc = np.fft.fftshift(spc, axes=(0,))
    return spc

parser = create_parser()
CKP = parser.ckpt
device = "cuda"

cfg = OmegaConf.create(yaml.safe_load(open(f"{CKP}/config.yaml", "r")))

if cfg.dataset.offset > 0 and os.path.exists(f"{parser.ckpt}/normalization_stats.pkl"):
    with open(f"{parser.ckpt}/normalization_stats.pkl", "rb") as infile:
        normalization_stats = pickle.load(infile)
else:
    normalization_stats = None

input_fields = np.unique(cfg.dataset.input_fields + cfg.model.losses)
traindata = CycloneDataset(
    active_keys=cfg.dataset.active_keys,
    input_fields=input_fields,
    path=parser.data_path,
    random_seed=cfg.seed,
    normalization=cfg.dataset.normalization,
    normalization_scope=cfg.dataset.normalization_scope,
    spatial_ifft=cfg.dataset.spatial_ifft,
    bundle_seq_length=cfg.model.bundle_seq_length,
    trajectories=cfg.dataset.training_trajectories,
    subsample=cfg.dataset.subsample,
    log_transform=cfg.dataset.log_transform,
    split_into_bands=cfg.dataset.split_into_bands,
    minmax_beta1=cfg.dataset.minmax_beta1,
    minmax_beta2=cfg.dataset.minmax_beta2,
    offset=cfg.dataset.offset,
    separate_zf=cfg.dataset.separate_zf,
)

data = CycloneDataset(
    trajectories=[parser.eval_sim],
    active_keys=cfg.dataset.active_keys,
    split="val",
    random_seed=cfg.seed,
    path=parser.data_path,
    normalization=cfg.dataset.normalization,
    normalization_scope=cfg.dataset.normalization_scope,
    normalization_stats=traindata.norm_stats,
    spatial_ifft=cfg.dataset.spatial_ifft,
    bundle_seq_length=cfg.model.bundle_seq_length,
    subsample=cfg.dataset.subsample,
    minmax_beta1=cfg.dataset.minmax_beta1,
    minmax_beta2=cfg.dataset.minmax_beta2,
    offset=cfg.dataset.offset,
    separate_zf=cfg.dataset.separate_zf,
)
raw_path = f"/restricteddata/ukaea/gyrokinetics/raw/{parser.eval_sim.replace('.h5', '')}"
print(f"Val: {len(data)}")

assert traindata.norm_stats == data.norm_stats, "Normalization stats mismatch"
if cfg.dataset.offset > 0:
    # dump normalization stats so we only need to compute them once
    with open(f"{parser.ckpt}/normalization_stats.pkl", "wb") as out:
        pickle.dump(traindata.norm_stats, out)

model = get_model(cfg, dataset=data)
last = parser.last
path = f"{CKP}/best.pth" if not last else f"{CKP}/ckp.pth"
model, _, _ = load_model_and_config(path, model, device)
model = model.to(device)
model = model.eval()

ONESTEP = parser.onestep
cyclone_name = "_".join(data.files[0].split("/")[-1].split(".")[0].split("_")[:-1])
IDX_0 = parser.start_idx
IDX_END = len(data) - 2
norm_output = "_rescaled" if parser.rescale else ""
ifft_merge = "_ifft_merge" if parser.ifft_merge else ""
OUT_DIR = f"{CKP}/{'onestep{}'.format(ifft_merge) if ONESTEP else 'autoreg_t{}{}'.format(IDX_0, ifft_merge)}/{cyclone_name}/{'best' if not last else 'ckp'}"
os.makedirs(OUT_DIR, exist_ok=True)

losses = []
sample = data[IDX_0]
input_fields = data.input_fields
outputs = cfg.model.losses
conditioning = cfg.model.conditioning
inputs = { k: getattr(sample, k).unsqueeze(0).to(device, non_blocking=True) for k in input_fields if getattr(sample, k) is not None }
conds = { k: getattr(sample, k).unsqueeze(0).to(device, non_blocking=True) for k in conditioning if getattr(sample, k) is not None }
f_idx = sample.file_index.item()
timesteps = data.get_timesteps(torch.tensor([0], dtype=torch.long))

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

gt_corr = defaultdict(dict)
model_corr = defaultdict(dict)
shift_scale_dict = defaultdict(dict)
invert_fns = {
    "df": partial(invert_df, cfg=cfg, parser=parser),
    "phi": invert_phi,
    "flux": None
}

for key in input_fields:
    if cfg.dataset.normalization == "zscore":
        shift = torch.tensor(traindata.norm_stats[key]["full"]["mean"]).unsqueeze(0).to(device)
        scale = torch.tensor(traindata.norm_stats[key]["full"]["std"]).unsqueeze(0).to(device)
    elif cfg.dataset.normalization == "minmax":
        x_min = torch.tensor(traindata.norm_stats[key]["full"]["min"]).unsqueeze(0).to(device)
        x_max = torch.tensor(traindata.norm_stats[key]["full"]["max"]).unsqueeze(0).to(device)
        scale = (x_max - x_min) / cfg.dataset.beta1
        shift = x_min + scale * cfg.dataset.beta2
    shift_scale_dict[key]["shift"] = shift
    shift_scale_dict[key]["scale"] = scale

with torch.no_grad():
    for idx in range(IDX_0, IDX_END + 1):

        ts = timesteps[:, idx].to(device)
        conds["timestep"] = ts

        if idx <= IDX_END or ONESTEP:
            sample = data[idx]
            inputs_t = {k: getattr(sample, k).unsqueeze(0).to(device, non_blocking=True) for k in input_fields if
                      getattr(sample, k) is not None}
            gts_t = {k: getattr(sample, f"y_{k}").unsqueeze(0).to(device, non_blocking=True) for k in input_fields if
                   getattr(sample, k) is not None}

            for key in input_fields:
                xt_gt = inputs_t[key]
                yt = gts_t[key].squeeze()
                if key == "df" and cfg.dataset.separate_zf:
                    xt_gt = xt_gt.squeeze()[:2] + xt_gt.squeeze()[2:]
                    yt = yt.squeeze()[:2] + yt.squeeze()[2:]
                gt_corr[key][ts] = compute_pearson_correlation(xt_gt.squeeze(), yt)

        if ONESTEP:
            outputs = model(**inputs_t, **conds)
        else:
            outputs = model(**inputs, **conds)
            # replace inputs with outputs for next timestep
            for key in input_fields:
                inputs[key] = outputs[key].clone()

        if idx <= IDX_END:
            for key in outputs.keys():
                if key == "df" and cfg.dataset.separate_zf:
                    xt_merged = outputs[key].squeeze()[:2] + outputs[key].squeeze()[2:]
                else:
                    xt_merged = outputs[key].squeeze()
                model_corr[key][ts] = compute_pearson_correlation(xt_merged, yt)

        for key in outputs.keys():
            scale = shift_scale_dict[key]["scale"]
            shift = shift_scale_dict[key]["shift"]
            # denormalize
            b_xt = gts_t[key] * scale + shift

            invert_fn = invert_fns[key]
            if invert_fn is not None:
                b_xt = invert_fn(b_xt)

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
                with open(ftarget, "wb") as f:
                    print(f"Writing file {ftarget}")
                    f.write(b_xt)

pickle.dump(model_corr, open(f"{OUT_DIR}/model_corr.pkl", "wb"))
pickle.dump(gt_corr, open(f"{OUT_DIR}/gt_corr.pkl", "wb"))
