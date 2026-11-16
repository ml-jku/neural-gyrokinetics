"""Heat flux evaluation of the GyroSwin hub checkpoints on the id/ood cases (run from the repo root)."""

import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from neugk.dataset.backend import load_meta
from neugk.gyroswin.eval.inference_from_hf import main as rollout

ID_CASES = [f"iteration_{i}" for i in (8, 115, 131, 148, 235, 262)]
OOD_CASES = [f"ood_iteration_{i}" for i in range(5)]
COND_KEYS = {"itg": "ion_temp_grad", "dg": "density_grad", "s_hat": "s_hat", "q": "q"}


def find_meta(root, name):
    for split in ("id", "ood"):
        base = os.path.join(root, split, f"{name}_ifft_realpotens", "metadata_light")
        if os.path.exists(base + ".npz") or os.path.exists(base + ".pkl"):
            return split, load_meta(base)
    raise FileNotFoundError(f"no metadata for {name} under {root}/{{id,ood}}")


def rmse(errs):
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", default="ml-jku/gyroswin_small")
    parser.add_argument("--cases", nargs="+", default=ID_CASES + OOD_CASES)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data_path", default="cache")
    parser.add_argument("--last", type=int, default=80)
    parser.add_argument("--save-fds", action="store_true")
    parser.add_argument(
        "--gt-conditioning",
        action="store_true",
        help="condition on the operating parameters from --data-root metadata",
    )
    args = parser.parse_args()

    errs = {"id": [], "ood": []}
    print(f"{'case':<22}{'split':<6}{'pred':>10}{'gt':>10}{'err':>10}")
    for name in args.cases:
        split, meta = find_meta(args.data_root, name)
        cond = None
        if args.gt_conditioning:
            cond = {
                k: np.atleast_1d(np.asarray(meta[m], dtype=np.float64))
                for k, m in COND_KEYS.items()
            }
        run_args = Namespace(ckpt=args.ckpt, data_path=args.data_path, eval_sim=f"{name}.h5")
        pred = rollout(run_args, save_fds=args.save_fds, cond_override=cond)
        q_pred = float(pred[-args.last :].mean())
        q_gt = float(np.asarray(meta["flux"])[-args.last :].mean())
        errs[split].append(q_pred - q_gt)
        print(f"{name:<22}{split:<6}{q_pred:>10.2f}{q_gt:>10.2f}{q_pred - q_gt:>10.2f}", flush=True)

    for split, e in errs.items():
        print(f"{split} Qbar RMSE: {rmse(e):.2f} (n={len(e)})")


if __name__ == "__main__":
    main()
