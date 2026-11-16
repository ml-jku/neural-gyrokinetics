"""Sample GyroFlow on the id/ and ood/ trajectory sets and report the time-averaged heat-flux error."""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from generate_samples import generate  # noqa: E402

SUFFIX = "_ifft_realpotens"


def discover(data_path, split):
    dirs = sorted(glob.glob(os.path.join(data_path, split, f"*{SUFFIX}")))
    return [f"{split}/{os.path.basename(d)[: -len(SUFFIX)]}.h5" for d in dirs]


def score(out_dir, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows, err, boot = [], [], []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.h5"))):
        with h5py.File(path, "r") as f:
            pred, gt = f["pred_flux"][:], float(f.attrs["gt_avg_flux"])
            name = f.attrs["traj_name"]
        rows.append(
            {
                "trajectory": name,
                "gt": gt,
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
            }
        )
        err.append(pred.mean() - gt)
        boot.append(rng.choice(pred, (n_boot, pred.size)).mean(1) - gt)
    rmse = float(np.sqrt(np.mean(np.square(err))))
    lo, hi = np.percentile(np.sqrt(np.mean(np.square(boot), 0)), [2.5, 97.5])
    return {"qbar_rmse": rmse, "ci95": [float(lo), float(hi)], "n": len(rows), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="sampling config (see configs/generation/generate_samples.yaml)")
    ap.add_argument("--splits", nargs="+", default=["id", "ood"])
    ap.add_argument("--save-df", action="store_true", help="also store the generated fields")
    ap.add_argument("--score-only", action="store_true", help="score existing outputs")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    root = cfg.generation.output_dir
    results = {}
    for split in args.splits:
        out_dir = os.path.join(root, split)
        if not args.score_only:
            run = cfg.copy()
            run.generation.trajectories = discover(cfg.data_path, split)
            run.generation.output_dir = out_dir
            run.generation.save_df = args.save_df
            run.generation.calculate_mse_metrics = True
            generate(run)
        results[split] = score(out_dir)

    for split, r in results.items():
        print(
            f"\n{split}: Qbar RMSE {r['qbar_rmse']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] over {r['n']} trajectories"
        )
        for row in r["rows"]:
            print(
                f"  {row['trajectory']:24s} gt {row['gt']:8.2f}  pred {row['pred_mean']:8.2f} ± {row['pred_std']:6.2f}"
            )
    with open(os.path.join(root, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
