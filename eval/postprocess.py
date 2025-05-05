import sys
sys.path.append("..")
import os
from argparse import ArgumentParser
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import pev_flux_df_phi, phi_integral, load_geometry

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--gkw", action="store_true")
    return parser.parse_args()

def correct_time(gt_path, flux_dict):
    pred_fluxes = {}
    for k_fid, pred_flux in flux_dict.items():
        gt_time = os.path.join(gt_path, f"{k_fid}.dat")
        if not os.path.exists(gt_time):
            gt_time = os.path.join(gt_path, f"{k_fid.replace('K', '')}.dat")
        with open(gt_time, "r") as file:
            for line in file:
                line_split = line.split("=")
                if line_split[0].strip() == "TIME":
                    t = float(line_split[1].strip().strip(",").strip())
                    break
        pred_fluxes[t] = max(pred_flux, 0)
    return pred_fluxes

def plot_flux_comp(gt_fluxes, pred_fluxes, dump_dir, gkw=False):
    gkw_flag = "_gkw" if gkw else ""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(np.array(list(gt_fluxes.keys())), list(gt_fluxes.values()), label="GT", c="tab:blue")
    ax.plot(np.array(list(pred_fluxes.keys())), list(pred_fluxes.values()), c="tab:green",
            label="Pred")
    plt.legend()
    plt.tight_layout()
    plt.gcf().savefig(os.path.join(dump_dir, "plots", f"flux_comp_{gkw_flag}.png"))
    plt.close()

def correlation_plot(model_corr, gt_corr, dump_dir):
    for key in model_corr.keys():
        mc = model_corr[key]
        gt_c = gt_corr[key]
        plt.plot([val.cpu().item() for val in mc.keys()], [val.cpu().item() for val in mc.values()],
                 c="tab:blue", label="Pred")
        plt.plot([val.cpu().item() for val in gt_c.keys()], [val.cpu().item() for val in gt_c.values()],
                 c="tab:red", label="Identity")
        plt.legend()
        plt.xlabel("Timestep")
        plt.ylabel("Correlation to ground-truth")
        plt.tight_layout()
        plt.savefig(os.path.join(dump_dir, "plots", f"{key}_corr_time.png"))
        plt.close()

args = parse_args()
DUMP_DIR = args.pred_path
sim = DUMP_DIR.split('/')[-2]
GT_DIR = os.path.join("/restricteddata/ukaea/gyrokinetics/raw", sim)
gt_fluxes = np.loadtxt(os.path.join(GT_DIR, "fluxes.dat"))
gt_time = np.loadtxt(os.path.join(GT_DIR, "time.dat"))
gt_fluxes = dict(zip(gt_time, gt_fluxes[:, 1]))

if args.gkw:
    # GKW used for flux computation, simply load fluxes
    assert os.path.exists(f"{DUMP_DIR}/fluxes_dict"), "GKW fluxes not found..."
    with open(f"{DUMP_DIR}/fluxes_dict", "rb") as f:
        pred_fluxes_dict = pickle.load(f)
    pred_fluxes = correct_time(GT_DIR, pred_fluxes_dict)
else:
    pred_fluxes = {}
    k_files = sorted([f for f in os.listdir(GT_DIR) if f.startswith("K") and not f.endswith(".dat")])
    if os.path.exists(os.path.join(DUMP_DIR, k_files[0], "flux")):
        # For XNet simply load fluxes from each K-file
        print("Loading flux predictions...")
        for k_file in k_files:
            flux = np.loadtxt(os.path.join(DUMP_DIR, k_file, "flux"))
            pred_fluxes[k_file] = flux.item()
    else:
        # compute flux integral
        print("Computing flux integral for dumps...")
        # load data for resolution and geometry
        # read helper vars
        sgrid = np.loadtxt(f"{GT_DIR}/sgrid")
        krho = np.loadtxt(f"{GT_DIR}/krho")
        vpgr = np.loadtxt(f"{GT_DIR}/vpgr.dat")
        # number of parallel direction grid points
        ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
        # number of modes in x and y direction
        nkx, nky = krho.shape[1], krho.shape[0]
        # get velocity space resolutions
        nvpar, nmu = vpgr.shape[1], vpgr.shape[0]
        resolution = (nvpar, nmu, ns, nkx, nky)
        geometry = load_geometry(GT_DIR)
        for k_file in tqdm(k_files, total=len(k_files), desc="Computing integral..."):
            assert os.path.exists(os.path.join(DUMP_DIR, k_file, "FDS")), f"FDS file for {k_file} missing..."
            # Load the full distribution function data
            with open(f"{DUMP_DIR}/{k_file}/FDS", "rb") as fid:
                df = np.fromfile(fid, dtype=np.float64)

            # Reshape the distribution function (copy for speeed in stat computation)
            df = np.reshape(df, (2, *resolution), order="F").copy()
            df = torch.from_numpy(df)
            phi = phi_integral(df, geometry)
            df = df.movedim(0, -1).contiguous()
            df = torch.view_as_complex(df)
            _, flux, _ = pev_flux_df_phi(df, phi, geometry, )
            pred_fluxes[k_file] = flux.item()
        pred_fluxes = correct_time(GT_DIR, pred_fluxes)

model_corr = pickle.load(open(f"{DUMP_DIR}/model_corr.pkl", "rb"))
gt_corr = pickle.load(open(f"{DUMP_DIR}/gt_corr.pkl", "rb"))

os.makedirs(os.path.join(DUMP_DIR, "plots"), exist_ok=True)
plot_flux_comp(gt_fluxes, pred_fluxes, DUMP_DIR)
correlation_plot(model_corr, gt_corr, DUMP_DIR)

