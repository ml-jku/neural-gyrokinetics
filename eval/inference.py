import sys
sys.path.append("..")
import os
from argparse import ArgumentParser
import pickle
import yaml
from omegaconf import OmegaConf
import torch
import numpy as np

from utils import load_model_and_config, expand_as
from models import get_model
from dataset.cyclone import CycloneDataset

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_path", default="/restricteddata/ukaea/gyrokinetics/preprocessed")
    parser.add_argument("--eval_sim", default="cyclone4_2_2.h5")
    parser.add_argument("--onestep", action="store_true")
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--ifft_merge", action="store_true")
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--zf_last", action="store_true")
    return parser.parse_args()

def invert_ifft(x):
    # invert fft on spatial
    knth = np.moveaxis(x, 0, -1).copy()
    knth = knth.view(dtype=np.complex64)
    # shift freqs to correct range
    knth = np.fft.fftn(knth, axes=(3, 4))
    knth = np.fft.ifftshift(knth, axes=(3,))
    knth = np.stack([knth.real, knth.imag]).squeeze().astype("float32")
    return knth

def modify_fds_dat(path):
    with open(path, 'r') as infile:
        content = infile.read()
        content = content.replace("DTIM    =  2.000000000000000E-002", "DTIM    =  0.0")
        content = content.replace("NT_REMAIN       =           0", "NT_REMAIN       =           1")
        content = content.replace("TIME    =   192.753733197446     ", "TIME    =   0")

    with open(path, 'w') as outfile:
        outfile.write(content)


def modify_input_dat(path):
    with open(path, 'r') as infile:
        content = infile.read()
        content = content.replace("READ_FILE  = .false.", "READ_FILE  = .true.")
        content = content.replace("DTIM   = 0.02", "DTIM   = 0.0")
        content = content.replace("out3d_interval = 3", "out3d_interval = 1")
        content = content.replace("keep_dumps = .true.", "! keep_dumps = .true.")
        content = content.replace("ndump_ts = 3", "! ndump_ts = 3")

    with open(path, 'w') as outfile:
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

parser = create_parser()
CKP = parser.ckpt
device = "cuda"

cfg = OmegaConf.create(yaml.safe_load(open(f"{CKP}/config.yaml", "r")))

if cfg.dataset.offset > 0 and os.path.exists(f"{parser.ckpt}/normalization_stats.pkl"):
    with open(f"{parser.ckpt}/normalization_stats.pkl", "rb") as infile:
        normalization_stats = pickle.load(infile)
else:
    normalization_stats = None

traindata = CycloneDataset(
    trajectories=cfg.dataset.training_trajectories,
    active_keys=cfg.dataset.active_keys,
    path=parser.data_path,
    split="train",
    random_seed=cfg.seed,
    normalization=cfg.dataset.normalization,
    normalization_scope=cfg.dataset.normalization_scope,
    normalization_stats=normalization_stats,
    spatial_ifft=cfg.dataset.spatial_ifft,
    bundle_seq_length=cfg.model.bundle_seq_length,
    subsample=cfg.dataset.subsample,
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
    normalization_stats=traindata.dataset_stats,
    spatial_ifft=cfg.dataset.spatial_ifft,
    bundle_seq_length=cfg.model.bundle_seq_length,
    subsample=cfg.dataset.subsample,
    minmax_beta1=cfg.dataset.minmax_beta1,
    minmax_beta2=cfg.dataset.minmax_beta2,
    offset=cfg.dataset.offset,
    separate_zf=cfg.dataset.separate_zf,
)

print(f"Val: {len(data)}")

assert (traindata.dataset_stats == data.dataset_stats), "Normalization stats mismatch"
if cfg.dataset.offset > 0:
    # dump normalization stats so we only need to compute them once
    with open(f"{parser.ckpt}/normalization_stats.pkl", "wb") as out:
        pickle.dump(traindata.dataset_stats, out)

model = get_model(cfg, dataset=data)
last = parser.last
path = f"{CKP}/best.pth" if not last else f"{CKP}/ckp.pth"

model, _, _ = load_model_and_config(path, model, device)

model = model.to(device)
model = model.eval()

ONESTEP = parser.onestep
cyclone_name = '_'.join(data.files[0].split('/')[-1].split('.')[0].split('_')[:-1])
IDX_0 = parser.start_idx
IDX_END = len(data)-2
norm_output = "_rescaled" if parser.rescale else ""
ifft_merge = "_ifft_merge" if parser.ifft_merge else ""
OUT_DIR = f"{CKP}/{'onestep{}{}'.format(norm_output, ifft_merge) if ONESTEP else 'autoreg_t{}{}{}'.format(IDX_0, norm_output, ifft_merge)}/{cyclone_name}/{'best' if not last else 'ckp'}"
os.makedirs(OUT_DIR, exist_ok=True)

losses = []
sample = data[IDX_0]
xt = sample.x.to(device).unsqueeze(0)
itg = sample.itg.to(device).unsqueeze(0)
f_idx = sample.file_index.item()
timesteps = data.get_timesteps(torch.tensor([0], dtype=torch.long))
delta = (timesteps[:, 1:].squeeze() - timesteps[:, :-1].squeeze()).squeeze()[-1]
if IDX_END > len(data) - 2:
    # add future timesteps, not observed during training
    timesteps = torch.cat([timesteps, torch.arange(timesteps[:, -1].item() + delta, IDX_END, delta).unsqueeze(0)],
                          dim=1)
files = []
gt_corr = {}
model_corr = {}

if cfg.dataset.normalization == "zscore":
    shift = torch.tensor(traindata.dataset_stats["full"]["mean"]).unsqueeze(0).to(device)
    scale = torch.tensor(traindata.dataset_stats["full"]["std"]).unsqueeze(0).to(device)
elif cfg.dataset.normalization == "minmax":
    x_min = torch.tensor(traindata.dataset_stats["full"]["min"]).unsqueeze(0).to(device)
    x_max = torch.tensor(traindata.dataset_stats["full"]["max"]).unsqueeze(0).to(device)
    scale = (x_max - x_min) / cfg.dataset.beta1
    shift = x_min + scale * cfg.dataset.beta2

with torch.no_grad():
    for idx in range(IDX_0, IDX_END + 1):

        ts = timesteps[:, idx].to(device)
        if idx <= IDX_END or ONESTEP:
            sample = data[idx]
            xt_gt = sample.x.to(device).unsqueeze(0)
            yt = sample.y.to(device).unsqueeze(0)
            if cfg.dataset.separate_zf:
                xt_gt_merged = xt_gt.squeeze()[:2] + xt_gt.squeeze()[2:]
                yt = yt.squeeze()[:2] + yt.squeeze()[2:]
            gt_corr[ts] = compute_pearson_correlation(xt_gt_merged, yt)

        if ONESTEP:
            xt = model(xt_gt, timestep=ts, itg=itg)
        else:
            xt = model(xt, timestep=ts, itg=itg)

        if idx <= IDX_END:
            if cfg.dataset.separate_zf:
                xt_merged = xt.squeeze()[:2] + xt.squeeze()[2:]
            model_corr[ts] = compute_pearson_correlation(xt_merged, yt)

        if parser.rescale:
            if cfg.dataset.separate_zf:
                print("Rescaling for ifft_merge not supported, will lead to invalid heat flux...")
            # re-scale output to unit variance
            xt = xt / xt.std((2, 3, 4, 5, 6), keepdims=True)

        # denormalize
        b_xt = xt * scale + shift
        if cfg.dataset.separate_zf:
            if parser.ifft_merge:
                assert shift.shape == xt.mean((2, 3, 4, 5, 6), keepdims=True).shape, "Normalization stats mismatch"
                assert scale.shape == xt.std((2, 3, 4, 5, 6), keepdims=True).shape, "Normalization stats mismatch"

                if parser.zf_last:
                    zf = invert_ifft(b_xt.cpu().numpy().squeeze()[2:, ...])
                    no_zf = invert_ifft(b_xt.cpu().numpy().squeeze()[:2, ...])
                else:
                    zf = invert_ifft(b_xt.cpu().numpy().squeeze()[:2, ...])
                    no_zf = invert_ifft(b_xt.cpu().numpy().squeeze()[2:, ...])
                b_xt = np.zeros_like(zf)
                b_xt[..., 0] = zf[..., 0]
                b_xt[..., 1:] = no_zf[..., :-1]
                b_xt = torch.tensor(np.expand_dims(b_xt, axis=0)).to(xt.device)
            else:
                zf = b_xt[:, 2:]
                no_zf = b_xt[:, :2]
                b_xt = zf + no_zf

        b_xt = b_xt.squeeze(0).cpu().numpy()
        if cfg.dataset.spatial_ifft and not parser.ifft_merge:
            # apply fft if not done so already
            b_xt = invert_ifft(b_xt)

        b_xt = b_xt.astype("float64").reshape(-1, order="F")
        # dump to file
        if OUT_DIR:
            dirtarget = os.path.join(OUT_DIR, f"K{str((int(idx) + 1 + cfg.dataset.offset) * cfg.dataset.subsample).zfill(2)}")
            os.makedirs(dirtarget, exist_ok=True)
            ftarget = os.path.join(dirtarget, "FDS")
            os.system(
                f"cp {data.files[0].replace("preprocessed", "raw").replace("_ifft", "").replace("_separate_zf", "").replace(".h5", "")}/input.dat {dirtarget}")
            os.system(
                f"cp {data.files[0].replace("preprocessed", "raw").replace("_ifft", "").replace("_separate_zf", "").replace(".h5", "")}/FDS.dat {dirtarget}")
            modify_fds_dat(f"{dirtarget}/FDS.dat")
            modify_input_dat(f"{dirtarget}/input.dat")
            with open(ftarget, "wb") as f:
                files.append(ftarget)
                print(f"Writing file {dirtarget}")
                f.write(b_xt)

            os.system(f"chmod -R 777 {dirtarget}/*")

pickle.dump(model_corr, open(f"{OUT_DIR}/model_corr.pkl", "wb"))
pickle.dump(gt_corr, open(f"{OUT_DIR}/gt_corr.pkl", "wb"))
