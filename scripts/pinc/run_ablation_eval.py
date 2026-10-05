"""Evaluate the 8 loss-component ablation neural fields through the SAME eval
pipeline as every other method (neugk.pinc.eval.runner.evaluate_method): per-snapshot
ml_eval (df/phi PSNR, eflux L1) + time-averaged spectral WD(k_y) + temporal EPE.

The 8 rows are the power set of {int, diag, mono} on the df anchor. Two endpoints
already exist from the main revival run; the six intermediate combos come from the
resume run (combo{0..5}). All use the canonical prefixes
(best_mlp = density, best_int_mlp = PINC), identical to run_pinc_eval.NF_PREFIX.

  python scripts/run_ablation_eval.py --gpus 1,2,3,4,5,6
"""

import os
import sys
import re
import glob
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.multiprocessing as mp

from neugk.pinc.eval.reconstructors import NeuralField
from neugk.pinc.eval.runner import evaluate_method

REVIVAL = "/system/user/galletti/git/neural-gyrokinetics-gitlab/nf_ckps_revival"
ABL = "/system/user/publicwork/galletti/pinc_revival/nf_ckps_ablation"
TIMESTEPS = [110, 160, 210, 260]

# label -> (checkpoint dir, prefix). order = table order (power set on df anchor).
METHODS = {
    "none": (REVIVAL, "best_mlp"),  # df only (density base)
    "int": (f"{ABL}/combo0", "best_int_mlp"),
    "diag": (f"{ABL}/combo1", "best_int_mlp"),
    "mono": (f"{ABL}/combo2", "best_int_mlp"),
    "int+diag": (f"{ABL}/combo3", "best_int_mlp"),
    "int+mono": (f"{ABL}/combo4", "best_int_mlp"),
    "diag+mono": (f"{ABL}/combo5", "best_int_mlp"),
    "int+diag+mono": (REVIVAL, "best_int_mlp"),  # full PINC
}


def discover(ckpt_dir, prefix):
    """{traj: {t: path}} restricted to TIMESTEPS, plus the CR."""
    rx = re.compile(re.escape(prefix) + r"_(iteration_\d+)_t(\d+)_x(\d+)\.pt$")
    weights, cr = defaultdict(dict), None
    for p in glob.glob(os.path.join(ckpt_dir, prefix + "_*.pt")):
        m = rx.search(os.path.basename(p))
        if m and int(m.group(2)) in TIMESTEPS:
            weights[m.group(1)][int(m.group(2))] = p
            cr = int(m.group(3))
    return {k: dict(v) for k, v in weights.items()}, cr


def worker(gpu, labels, path, backend, q):
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    # epe is not an ablation-table column
    import neugk.pinc.eval.runner as _runner

    _runner.temporal_epe = lambda *a, **k: float("nan")
    out = {}
    for label in labels:
        ckpt_dir, prefix = METHODS[label]
        weights, cr = discover(ckpt_dir, prefix)
        trajs = [f"{t}.h5" for t in sorted(weights, key=lambda s: int(s.split("_")[1]))]
        rec = NeuralField(name=label, weights=weights, path=path, backend=backend)
        with torch.no_grad():
            agg, diags = evaluate_method(rec, trajs, TIMESTEPS, path, backend, dev)
        out[label] = {"cr": cr, "n_traj": len(trajs), **{k: v[0] for k, v in agg.items()}}
        # post-peak monotonicity violation of the spectra (the l_mono term), pred and gt
        from neugk.physics.diagnostics import monotonicity_loss

        mv = defaultdict(list)
        for td in diags.values():
            for sk, src in (("kyspec", ""), ("qspec", ""), ("kyspec", "_gt"), ("qspec", "_gt")):
                for arr in td.get(sk + src, []):
                    ml = monotonicity_loss(
                        {sk: torch.as_tensor(arr, dtype=torch.float32)}, keys=(sk,)
                    )
                    mv[f"{sk}_mono_viol{src}"].append(float(ml[f"{sk} monotonicity loss"]))
        out[label].update({k: (sum(v) / len(v) if v else float("nan")) for k, v in mv.items()})
        print(
            f"[gpu{gpu}] {label}: psnr_f={out[label].get('psnr'):.2f} "
            f"phi_psnr={out[label].get('phi_psnr'):.2f} "
            f"eflux_l1={out[label].get('eflux_l1'):.4f} "
            f"ky_wd={out[label].get('kyspec_wd'):.5f}",
            flush=True,
        )
    q.put(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,3,4,5,6")
    ap.add_argument("--path", default="/local00/bioinf/galletti/preprocessed_kvikio")
    ap.add_argument("--backend", default="gds")
    ap.add_argument("--out", default=f"{ABL}/ablation_eval.json")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    labels = list(METHODS)
    shards = [labels[i :: len(gpus)] for i in range(len(gpus))]
    shards = [(g, s) for g, s in zip(gpus, shards) if s]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for g, s in shards:
        p = ctx.Process(target=worker, args=(g, s, args.path, args.backend, q))
        p.start()
        procs.append(p)
    results = {}
    for _ in shards:
        results.update(q.get())
    for p in procs:
        p.join()

    results = {k: results[k] for k in labels if k in results}  # table order
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    hdr = (
        f"{'combo':16} {'PSNR(f)':>8} {'L_Q':>8} {'PSNR(phi)':>10} {'WD(ky)':>9}"
        f" | {'kyMonoViol':>10} {'(GT)':>7} {'qMonoViol':>10} {'(GT)':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for k in labels:
        r = results.get(k, {})
        g = lambda key: r.get(key, float("nan"))
        print(
            f"{k:16} {g('psnr'):8.2f} {g('eflux_l1'):8.4f} {g('phi_psnr'):10.2f} {g('kyspec_wd'):9.5f}"
            f" | {g('kyspec_mono_viol'):10.4f} {g('kyspec_mono_viol_gt'):7.4f}"
            f" {g('qspec_mono_viol'):10.4f} {g('qspec_mono_viol_gt'):7.4f}"
        )


if __name__ == "__main__":
    main()
