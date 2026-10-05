"""Rate-distortion driver: score NF-PINC and traditional baselines across SAME compression ratio span for one rate-distortion plot; thin wrapper over eval package; traditional: calibrate each method (zfp/wavelet/pca/jpeg2000/sz3) to every target CR via eval.discovery.calibrate, skip unreachable; NF/NF-PINC: built from nf_ckps_scale/<cr>x/ with nf_scaling_reconstructors, run_scaling does metrics + per-family resumable save to scaling.pkl, skips erroring variants
Examples:
  python scripts/run_scaling.py --gpu 0
  python scripts/run_scaling.py --methods zfp,sz3 --crs 10,200,5000 --gpu 3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run_pinc_eval
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # repo root

from neugk.pinc.eval import run_scaling, nf_scaling_reconstructors
from neugk.pinc.eval.reconstructors import Traditional
from neugk.pinc.eval.discovery import TRAD, NF_PREFIX

# NF scaling-config CR span (configs/pinc_revival/scaling/*.yaml)
DEFAULT_CRS = [10, 50, 200, 500, 1168, 2000, 5000, 10000, 50000]
# traditional knob name per method (matches eval.discovery.TRAD); for naming only
KNOB = {m: TRAD[m][1] for m in TRAD}
# the 60-trajectory test partition of configs/pinc_revival/ablation_components.yaml
TEST_TRAJS = [
    "iteration_0.h5",
    "iteration_8.h5",
    "iteration_20.h5",
    "iteration_36.h5",
    "iteration_41.h5",
    "iteration_48.h5",
    "iteration_55.h5",
    "iteration_65.h5",
    "iteration_73.h5",
    "iteration_79.h5",
    "iteration_85.h5",
    "iteration_94.h5",
    "iteration_100.h5",
    "iteration_104.h5",
    "iteration_108.h5",
    "iteration_113.h5",
    "iteration_117.h5",
    "iteration_121.h5",
    "iteration_125.h5",
    "iteration_130.h5",
    "iteration_134.h5",
    "iteration_138.h5",
    "iteration_142.h5",
    "iteration_146.h5",
    "iteration_151.h5",
    "iteration_155.h5",
    "iteration_159.h5",
    "iteration_163.h5",
    "iteration_168.h5",
    "iteration_172.h5",
    "iteration_176.h5",
    "iteration_180.h5",
    "iteration_185.h5",
    "iteration_189.h5",
    "iteration_193.h5",
    "iteration_197.h5",
    "iteration_202.h5",
    "iteration_206.h5",
    "iteration_210.h5",
    "iteration_214.h5",
    "iteration_218.h5",
    "iteration_223.h5",
    "iteration_227.h5",
    "iteration_231.h5",
    "iteration_235.h5",
    "iteration_240.h5",
    "iteration_244.h5",
    "iteration_248.h5",
    "iteration_252.h5",
    "iteration_257.h5",
    "iteration_261.h5",
    "iteration_265.h5",
    "iteration_269.h5",
    "iteration_274.h5",
    "iteration_278.h5",
    "iteration_282.h5",
    "iteration_286.h5",
    "iteration_291.h5",
    "iteration_295.h5",
    "iteration_299.h5",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scale-dir",
        default="nf_ckps_scale",
        help="dir holding NF scaling checkpoints in <cr>x/ subdirs",
    )
    ap.add_argument("--crs", default=",".join(map(str, DEFAULT_CRS)))
    ap.add_argument(
        "--methods",
        default="all",
        help="comma list of nf,nf-pinc,zfp,wavelet,pca,jpeg2000,sz3 (or 'all')",
    )
    ap.add_argument(
        "--trajs",
        default=",".join(TEST_TRAJS),
        help="comma list of trajectory files to evaluate on",
    )
    ap.add_argument(
        "--timesteps",
        default="110,160,210,260",
        help="comma list of timesteps (default = physics-component-ablation set)",
    )
    ap.add_argument("--path", default="/local00/bioinf/galletti/preprocessed_kvikio")
    ap.add_argument("--backend", default="gds")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", default="pinc_eval_results/scaling")
    args = ap.parse_args()

    crs = [int(c) for c in args.crs.split(",") if c]
    trajectories = [t for t in args.trajs.split(",") if t]
    timesteps = [int(t) for t in args.timesteps.split(",") if t]
    import torch

    torch.cuda.set_device(args.gpu)  # set the default device too, else tensors created
    # without an explicit device land on cuda:0 -> cross-device illegal access
    device = f"cuda:{args.gpu}"
    sel = list(NF_PREFIX) + list(TRAD) if args.methods == "all" else args.methods.split(",")

    groups = []  # (family_name, [reconstructors])

    # traditional: calibrate each method to each target CR (failproof: skip unreachable)
    trad_sel = [m for m in sel if m in TRAD]
    if trad_sel:
        for m in trad_sel:
            recs = []
            for cr in crs:
                # per-snapshot knob search
                recs.append(Traditional(f"{m.upper()}_x{cr}", method=m, target_cr=float(cr)))
                print(f"[scaling] {m.upper()}@{cr}x: per-snapshot knob search")
            if recs:
                groups.append((m.upper(), recs))

    # NF / NF-PINC: per-CR variants from the scaling checkpoint subdirs (if trained)
    nf_families = {"nf-pinc": ("NF-PINC", True), "nf": ("NF", False)}
    for key, (fam, int_only) in nf_families.items():
        if key not in sel:
            continue
        recs = []
        for cr in crs:
            d = os.path.join(args.scale_dir, f"{cr}x")
            if os.path.isdir(d):
                recs += [
                    r
                    for _, r in nf_scaling_reconstructors(
                        d,
                        device=device,
                        include_int_only=int_only,
                        path=args.path,
                        backend=args.backend,
                    )
                ]
        if recs:
            groups.append((fam, recs))
        else:
            print(
                f"[scaling] {fam}: no checkpoints under {args.scale_dir}/<cr>x "
                f"(train configs/pinc_revival/scaling/* first)"
            )

    if not groups:
        print("nothing to evaluate")
        return
    print(
        f"families: {[(g[0], len(g[1])) for g in groups]}  "
        f"on {len(trajectories)} trajs x {len(timesteps)} t"
    )
    run_scaling(groups, trajectories, timesteps, args.path, args.backend, device, args.outdir)


if __name__ == "__main__":
    main()
