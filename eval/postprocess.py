import sys
sys.path.append("..")
import os
from argparse import ArgumentParser
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct

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

def K_files(directory):
    files = os.listdir(directory)
    digit_files = sorted(
        [file for file in files if file.isdigit()], key=lambda x: int(x)
    )
    k_files = sorted(
        [file for file in files if file.startswith("K") and not file.endswith(".dat")]
    )
    return k_files + digit_files

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
if "iteration" in DUMP_DIR:
    if DUMP_DIR.endswith('/'):
        DUMP_DIR = DUMP_DIR[:-1]
    if "best" in DUMP_DIR or "ckp" in DUMP_DIR:
        sim = DUMP_DIR.split('/')[-2]
    else:
        sim = DUMP_DIR.split('/')[-1]
    GT_DIR = os.path.join("/restricteddata/ukaea/gyrokinetics/raw/", sim)
    if "param_scans" in DUMP_DIR:
        GT_FLUX_DIR = DUMP_DIR
    else:
        GT_FLUX_DIR = GT_DIR
    sims_to_eval = [sim]
else:
    sims_to_eval = [dir for dir in os.listdir(DUMP_DIR) if not dir.endswith("Lin") and not "alpha" in dir]
    # just set GT_DIR to some fixed value, can be arbitrary sim, will only be used for timestep loading
    GT_DIR = "/restricteddata/ukaea/gyrokinetics/raw/iteration_0"
    GT_FLUX_DIR = None

for sim in sims_to_eval:
    print(f"Postprocessing {sim}...")
    if GT_FLUX_DIR is None or ("param_scans" in args.pred_path and not "iteration" in args.pred_path):
        GT_FLUX_DIR = os.path.join(args.pred_path, sim)
    if "ood" in args.pred_path:
        GT_FLUX_DIR = GT_FLUX_DIR.replace("iteration", "ood/iteration")
    print(f"Loading GT fluxes from: {'_'.join(GT_FLUX_DIR.split('_')[:2])}")
    gt_fluxes = np.loadtxt(os.path.join('_'.join(GT_FLUX_DIR.split('_')[:2]), "fluxes.dat"))
    gt_time = np.loadtxt(os.path.join('_'.join(GT_FLUX_DIR.split('_')[:2]), "time.dat"))
    gt_fluxes = dict(zip(gt_time, gt_fluxes[:, 1][:800]))
    if "param_scans" in args.pred_path:
        # we are doing whole parameter scan
        if "iteration" in args.pred_path:
            DUMP_DIR = os.path.join(args.pred_path, "autoreg_t0")
        else:
            DUMP_DIR = os.path.join(args.pred_path, sim, "autoreg_t0")

    if args.gkw:
        # GKW used for flux computation, simply load fluxes
        assert os.path.exists(f"{DUMP_DIR}/fluxes_dict"), "GKW fluxes not found..."
        with open(f"{DUMP_DIR}/fluxes_dict", "rb") as f:
            pred_fluxes_dict = pickle.load(f)
        pred_fluxes = correct_time(GT_DIR, pred_fluxes_dict)
    else:
        pred_fluxes = {}
        k_dir = K_files('_'.join(GT_DIR.split('_')[:2]))[80:]
        if os.path.isfile(os.path.join(DUMP_DIR, k_dir[0], "flux")):
            # For XNet simply load fluxes from each K-file
            print("Loading flux predictions...")
            for k_file in k_dir[:-2]:
                if not os.path.isfile(os.path.join(DUMP_DIR, k_file, "flux")):
                    try:
                        flux = np.loadtxt(os.path.join(DUMP_DIR, f"K{k_file}", "flux"))
                    except:
                        try:
                            with open(os.path.join(DUMP_DIR, f"K{k_file}", "flux"), "rb") as fid:
                                ff = np.fromfile(fid, dtype=np.float64)
                            knth = np.reshape(ff, (2, 85, 32), order="F").astype("float32").copy()
                            flux = knth.sum()
                        except:
                            flux = open(os.path.join(DUMP_DIR, f"K{k_file}", "flux"), "rb").read()
                            flux = struct.unpack("d", flux)[0]
                else:
                    try:
                        flux = np.loadtxt(os.path.join(DUMP_DIR, k_file, "flux"))
                    except:
                        try:
                            with open(os.path.join(DUMP_DIR, k_file, "flux"), "rb") as fid:
                                ff = np.fromfile(fid, dtype=np.float64)
                            knth = np.reshape(ff, (2, 85, 32), order="F").astype("float32").copy()
                            flux = knth[1].sum()
                        except:
                            flux = open(os.path.join(DUMP_DIR, k_file, "flux"), "rb").read()
                            flux = struct.unpack("d", flux)[0]

                # correct for time shift
                if k_file.startswith("K"):
                    gt_k_file_time = f"K{str(int(k_file[1:])+1).zfill(2)}"
                    if not os.path.isfile(os.path.join('_'.join(GT_DIR.split('_')[:2]), gt_k_file_time)):
                        gt_k_file_time = gt_k_file_time[1:]
                else:
                    gt_k_file_time = f"{str(int(k_file) + 1).zfill(2)}"

                with open(f"{'_'.join(GT_DIR.split('_')[:2])}/{gt_k_file_time}.dat", "r") as file:
                    for line in file:
                        line_split = line.split("=")
                        if line_split[0].strip() == "TIME":
                            time = float(line_split[1].strip().strip(",").strip())
                if isinstance(flux, np.ndarray):
                    flux = flux.item()
                pred_fluxes[time] = flux
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

    os.makedirs(os.path.join(DUMP_DIR, "plots"), exist_ok=True)
    if os.path.exists(f"{DUMP_DIR}/model_corr.pkl"):
        model_corr = pickle.load(open(f"{DUMP_DIR}/model_corr.pkl", "rb"))
        gt_corr = pickle.load(open(f"{DUMP_DIR}/gt_corr.pkl", "rb"))
        correlation_plot(model_corr, gt_corr, DUMP_DIR)
    plot_flux_comp(gt_fluxes, pred_fluxes, DUMP_DIR)
    with open(f"{DUMP_DIR}/pred_fluxes.pkl", "wb") as fid:
        pickle.dump(pred_fluxes, fid)

