"""Heat flux evaluation of the GyroSwin hub checkpoints on id/ood trajectories (run from the repo root)."""

import glob
import os
import sys
import tempfile
from argparse import ArgumentParser, Namespace

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from neugk.dataset.backend import load_meta
from neugk.gyroswin.eval.inference_from_hf import main as rollout

SUFFIX = "_ifft_realpotens"
GYROSWIN_CASES = [f"iteration_{i}" for i in (8, 115, 131, 148, 235, 262)] + [
    f"ood_iteration_{i}" for i in range(5)
]
COND_KEYS = {"itg": "ion_temp_grad", "dg": "density_grad", "s_hat": "s_hat", "q": "q"}
START = 80


def find(root, name):
    for split in ("id", "ood"):
        d = os.path.join(root, split, name + SUFFIX)
        if os.path.isdir(d):
            return split, d
    raise FileNotFoundError(f"{name} not under {root}/{{id,ood}}")


def cases_for(root, split):
    if split == "gyroswin":
        return GYROSWIN_CASES
    splits = ["id", "ood"] if split == "all" else [split]
    dirs = [d for s in splits for d in sorted(glob.glob(os.path.join(root, s, "*" + SUFFIX)))]
    return [os.path.basename(d)[: -len(SUFFIX)] for d in dirs]


def local_snapshot(traj_dir, meta, path):
    """Write frame START of a dataset trajectory in the hub snapshot format."""
    raw = np.fromfile(os.path.join(traj_dir, "data", f"timestep_{START:05d}.bf16.bin"), np.int16)
    df = torch.from_numpy(raw).view(torch.bfloat16).float().numpy()
    res = np.asarray(meta["resolution"], np.int64)
    with h5py.File(path, "w") as f:
        f["data/timestep_00000"] = df.reshape(2, *res)
        f["metadata/resolution"] = res
        f["metadata/timesteps"] = np.asarray(meta["timesteps"][START : START + 1], np.float64)
        f["metadata/fluxes"] = np.asarray(meta["flux"][START : START + 1], np.float64)
        for m in COND_KEYS.values():
            f[f"metadata/{m}"] = np.atleast_1d(np.asarray(meta[m], np.float64))


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", default="ml-jku/gyroswin_small")
    parser.add_argument("--data-root", required=True, help="dataset root with id/ and ood/")
    parser.add_argument(
        "--split",
        default="gyroswin",
        choices=["gyroswin", "id", "ood", "all"],
        help="gyroswin: the 6 id + 5 ood cases of the GyroSwin paper",
    )
    parser.add_argument("--cases", nargs="+", help="explicit trajectory names, overrides --split")
    parser.add_argument("--data_path", default="cache")
    parser.add_argument("--last", type=int, default=80)
    parser.add_argument("--save-fds", action="store_true")
    parser.add_argument(
        "--local-snapshots",
        action="store_true",
        help="start the gyroswin cases from the dataset frames instead of the hub snapshots",
    )
    args = parser.parse_args()

    errs = {"id": [], "ood": []}
    tmp = tempfile.mkdtemp()
    print(f"{'case':<26}{'split':<6}{'pred':>10}{'gt':>10}{'err':>10}")
    for name in args.cases or cases_for(args.data_root, args.split):
        split, traj_dir = find(args.data_root, name)
        meta = load_meta(os.path.join(traj_dir, "metadata_light"))
        snapshot = None
        if args.local_snapshots or name not in GYROSWIN_CASES:
            snapshot = os.path.join(tmp, f"{name}.h5")
            local_snapshot(traj_dir, meta, snapshot)
        # conditioning from the dataset metadata
        cond = {k: np.atleast_1d(np.asarray(meta[m], np.float64)) for k, m in COND_KEYS.items()}
        run_args = Namespace(ckpt=args.ckpt, data_path=args.data_path, eval_sim=f"{name}.h5")
        pred = rollout(run_args, save_fds=args.save_fds, cond_override=cond, snapshot=snapshot)
        q_pred = float(pred[-args.last :].mean())
        q_gt = float(np.asarray(meta["flux"])[-args.last :].mean())
        errs[split].append(q_pred - q_gt)
        print(f"{name:<26}{split:<6}{q_pred:>10.2f}{q_gt:>10.2f}{q_pred - q_gt:>10.2f}", flush=True)

    for split, e in errs.items():
        if e:
            print(f"{split} Qbar RMSE: {np.sqrt(np.mean(np.square(e))):.2f} (n={len(e)})")


if __name__ == "__main__":
    main()
