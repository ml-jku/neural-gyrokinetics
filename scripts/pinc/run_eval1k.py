"""Unified 1k-set eval for ALL PINC baselines through the shared evaluate_method, so every row carries the
same schema (df/phi PSNR, flux L1, temporal EPE, k_y/Q spectral WD, velocity moments) and the same matched
CR convention (taken from the nf-pinc checkpoints).

Incremental checkpointing: each (method, trajectory) result is written to its own file under <outdir>/rows/
the moment it finishes, so an interruption never loses completed fits and a re-launch RESUMES where it
stopped (skips rows already on disk unless --overwrite). The per-method aggregate eval1k_<LABEL>.json is
(re)built from whatever rows exist -- run with --aggregate-only to refresh it without computing.

Methods: nf, nf-pinc, sz3, jpeg2000, zfp, ae, ae-pinc, pigs.

  # everything on the free B300s, resumable:
  python scripts/run_eval1k.py --methods nf,nf-pinc,sz3,jpeg2000,zfp,ae,ae-pinc,pigs --gpus 2,3
  # just resume PIGS (long pole):
  python scripts/run_eval1k.py --methods pigs --gpus 2,3
  # rebuild aggregates from on-disk rows without recomputing:
  python scripts/run_eval1k.py --methods pigs --aggregate-only
"""

import os
import sys
import json
import argparse
import glob

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neugk.pinc.eval.reconstructors import NeuralField, Traditional, Autoencoder, PIGS
from neugk.pinc.eval.runner import evaluate_method
from neugk.pinc.eval.discovery import discover, calibrate, TRAD, NF_PREFIX, CycloneNFDataset

# method id -> kind. trad ids match TRAD keys; ae ids map to run_ae_eval.AE_CKPTS labels.
NF_METHODS = ("nf", "nf-pinc")
TRAD_METHODS = ("sz3", "jpeg2000", "zfp", "pca", "wavelet")
AE_METHODS = {"ae": "AE-PRETRAIN", "ae-pinc": "PINC-AE"}
ALL_METHODS = NF_METHODS + TRAD_METHODS + tuple(AE_METHODS) + ("pigs",)


def label_of(method):
    """Disk/aggregate label for a method id (filename- and table-friendly)."""
    if method in TRAD_METHODS:
        return method.upper()
    if method in AE_METHODS:
        return AE_METHODS[method]
    if method == "pigs":
        return "PIGS"
    return method  # nf, nf-pinc


def rowfile(rowdir, label, traj):
    return os.path.join(rowdir, f"{label}__{traj}.json")


def get_recon(method, cache, ctx, dev):
    """Lazily build + cache the reconstructor for a method on this worker's GPU."""
    if method in cache:
        return cache[method]
    if method in NF_METHODS:
        rec = NeuralField(
            method, ctx["weights"][NF_PREFIX[method]], path=ctx["path"], backend=ctx["backend"]
        )
    elif method in TRAD_METHODS:
        if method in ctx.get("trad_knob", {}):
            from functools import partial

            fn = TRAD[method][0]
            knob = TRAD[method][1]
            v = ctx["trad_knob"][method]
            print(f"  [forced] {method.upper()}: {knob}={v}", flush=True)
            rec = Traditional(method.upper(), partial(fn, **{knob: v}))
            cache[method] = rec
            return rec
        if ctx.get("legacy_calib"):
            # legacy: one knob for the whole set, calibrated on a single slice (undershoots
            # on turbulent frames -> variable CR). Kept only for reproducing the old table.
            s_df = CycloneNFDataset(
                ctx["t0"],
                [ctx["tt"]],
                path=ctx["path"],
                backend=ctx["backend"],
                realpotens=True,
                normalize=None,
            ).full_df
            c = calibrate(method, s_df, ctx["nf_cr"])
            if c is None:
                raise RuntimeError(f"{method} uncalibratable at CR {ctx['nf_cr']}")
            fn, ach, v = c
            print(
                f"  [calib] {method.upper()}: {TRAD[method][1]}={v:.4g} -> CR {ach:.0f}x",
                flush=True,
            )
            rec = Traditional(method.upper(), fn)
        else:
            # default: per-snapshot knob search so every frame hits the learned CR (true iso-CR)
            print(
                f"  [match-cr] {method.upper()}: per-snapshot knob -> {ctx['nf_cr']:.0f}x",
                flush=True,
            )
            rec = Traditional(method.upper(), method=method, target_cr=ctx["nf_cr"])
    elif method in AE_METHODS:
        from neugk.pinc.autoencoders.ae_utils import load_autoencoder
        from neugk.pinc.eval.ae_data import build_make_val_dataset, AE_CKPTS

        ckp, load_peft, vqvae = AE_CKPTS[AE_METHODS[method]]
        model, _ck, config = load_autoencoder(ckp, dev, model=None, load_peft=load_peft)
        model = model.to(dev).eval()
        mk = build_make_val_dataset(config, ctx["path"], ctx["backend"])
        rec = Autoencoder(AE_METHODS[method], model, mk, vqvae=vqvae, vapor=False)
    elif method == "pigs":
        rec = PIGS("PIGS", ctx["path"], ctx["backend"], ctx["nf_cr"])
    else:
        raise ValueError(f"unknown method {method!r}")
    cache[method] = rec
    return rec


def worker(gpu, work, ctx, ts_by_traj, rowdir, q):
    torch.cuda.set_device(gpu)
    dev = torch.device(f"cuda:{gpu}")
    cache = {}
    done = 0
    for method, traj in work:
        label = label_of(method)
        rf = rowfile(rowdir, label, traj)
        if os.path.exists(rf) and not ctx.get(
            "overwrite"
        ):  # another worker / prior run already did it
            continue
        ts = ts_by_traj[traj]
        try:
            rec = get_recon(method, cache, ctx, dev)
            agg, diags = evaluate_method(rec, [traj + ".h5"], ts, ctx["path"], ctx["backend"], dev)
            row = {k: v[0] for k, v in agg.items()}
            row.update(method=label, traj=traj, n_timesteps=len(ts))
            if ctx.get(
                "dump_diags"
            ):  # per-(method,traj) per-timestep spectra for the cascade figure
                import pickle

                ddir = os.path.join(rowdir, "diags")
                os.makedirs(ddir, exist_ok=True)
                pickle.dump(diags, open(os.path.join(ddir, f"{label}__{traj}.pkl"), "wb"))
        except Exception as e:
            import traceback

            traceback.print_exc()
            row = dict(method=label, traj=traj, error=f"{type(e).__name__}: {e}")
        with open(rf, "w") as f:  # checkpoint: this (method, traj) survives any later crash
            json.dump(row, f, indent=2)
        done += 1
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        print(
            f"  [gpu{gpu}] {label} {traj}: "
            f"psnr={row.get('psnr', float('nan')):.2f} phi_psnr={row.get('phi_psnr', float('nan')):.2f} "
            f"eflux_l1={row.get('eflux_l1', float('nan')):.4f} (peak {peak:.1f} GB)"
            + (f"  ERROR {row['error']}" if "error" in row else ""),
            flush=True,
        )
    q.put(done)


def aggregate(rowdir, outdir, labels):
    """(Re)build eval1k_<LABEL>.json from the per-traj row files currently on disk."""
    for label in labels:
        rows = []
        for fp in sorted(glob.glob(rowfile(rowdir, label, "*"))):
            try:
                rows.append(json.load(open(fp)))
            except Exception:
                continue
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["traj"].split("_")[1]))
        good = [r for r in rows if "error" not in r]
        keys = [k for k in good[0] if k not in ("method", "traj", "n_timesteps")] if good else []
        agg = {
            k: [float(np.mean([r[k] for r in good])), float(np.std([r[k] for r in good]))]
            for k in keys
        }
        agg["n_traj"] = len(good)
        agg["n_err"] = len(rows) - len(good)
        json.dump(
            {"rows": rows, "agg": agg},
            open(os.path.join(outdir, f"eval1k_{label}.json"), "w"),
            indent=2,
        )
        print(
            f"{label}: n={len(good)} (+{len(rows) - len(good)} err)  "
            f"psnr={agg.get('psnr', [float('nan')])[0]:.2f} "
            f"phi_psnr={agg.get('phi_psnr', [float('nan')])[0]:.2f} "
            f"eflux_l1={agg.get('eflux_l1', [float('nan')])[0]:.4f} "
            f"endpoint={agg.get('endpoint', [float('nan')])[0]:.4f} "
            f"kyspec_wd={agg.get('kyspec_wd', [float('nan')])[0]:.5f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--ckpts", default="nf_ckps_revival")
    ap.add_argument("--path", default="/local00/bioinf/galletti/preprocessed_kvikio")
    ap.add_argument("--backend", default="kvikio")
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument(
        "--weights",
        default="",
        help="per-GPU work weights (comma list matching --gpus); "
        "e.g. '2,2,1,1' gives the first two GPUs 2x the trajs. Empty = even round-robin.",
    )
    ap.add_argument(
        "--outdir", default="/system/user/publicwork/galletti/pinc_revival/eval1k_results"
    )
    ap.add_argument("--max-trajs", type=int, default=0)
    ap.add_argument(
        "--trajs",
        default="",
        help="comma list of trajectory names to restrict to "
        "(e.g. iteration_0,iteration_8); empty = all discovered",
    )
    ap.add_argument("--overwrite", action="store_true", help="recompute even rows already on disk")
    ap.add_argument(
        "--dump-diags",
        action="store_true",
        help="dump per-(method,traj) per-timestep spectra pkls for the cascade figure",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild eval1k_<LABEL>.json from on-disk rows, no compute",
    )
    ap.add_argument(
        "--trad-knob", default="", help="force traditional knob, e.g. sz3=1.2,zfp=3.4,wavelet=11.7"
    )
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise SystemExit(f"unknown methods {bad}; choose from {list(ALL_METHODS)}")
    labels = [label_of(m) for m in methods]
    trad_knob = {}
    if args.trad_knob:
        for kv in args.trad_knob.split(","):
            k, v = kv.split("=")
            trad_knob[k.strip()] = float(v)
    rowdir = os.path.join(args.outdir, "rows")
    os.makedirs(rowdir, exist_ok=True)

    if args.aggregate_only:
        aggregate(rowdir, args.outdir, labels)
        return

    # canonical (traj, timesteps) set + matched CR from the nf-pinc checkpoints
    weights = {p: discover(args.ckpts, p)[0] for p in NF_PREFIX.values()}
    _, nf_cr = discover(args.ckpts, NF_PREFIX["nf-pinc"])
    trajs = sorted(weights[NF_PREFIX["nf-pinc"]], key=lambda s: int(s.split("_")[1]))
    if args.trajs:
        keep = set(args.trajs.split(","))
        trajs = [t for t in trajs if t in keep]
    if args.max_trajs:
        trajs = trajs[: args.max_trajs]
    ts_by_traj = {t: sorted(int(x) for x in weights[NF_PREFIX["nf-pinc"]][t]) for t in trajs}
    ctx = dict(
        weights=weights,
        nf_cr=nf_cr,
        path=args.path,
        backend=args.backend,
        t0=trajs[0],
        tt=ts_by_traj[trajs[0]][0],
        overwrite=args.overwrite,
        trad_knob=trad_knob,
        dump_diags=args.dump_diags,
    )
    print(
        f"{len(trajs)} trajs x ~{len(ts_by_traj[trajs[0]])} ts; methods={methods}; NF CR={nf_cr}x",
        flush=True,
    )

    # flat (method, traj) work list, resume-filtered, round-robin sharded so the expensive
    # method (PIGS) is balanced evenly across GPUs.
    work = [(m, t) for m in methods for t in trajs]
    if not args.overwrite:
        skip = sum(1 for m, t in work if os.path.exists(rowfile(rowdir, label_of(m), t)))
        work = [(m, t) for m, t in work if not os.path.exists(rowfile(rowdir, label_of(m), t))]
        if skip:
            print(
                f"resuming: {skip} (method,traj) rows already on disk, {len(work)} to do",
                flush=True,
            )
    if not work:
        print("nothing to compute; aggregating.")
        aggregate(rowdir, args.outdir, labels)
        return

    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    weights = [int(w) for w in args.weights.split(",")] if args.weights else [1] * len(gpus)
    if len(weights) != len(gpus):
        raise SystemExit(f"--weights ({len(weights)}) must match --gpus ({len(gpus)})")
    # weighted round-robin: a GPU with weight w gets ~w/sum(w) of the items (faster/free GPUs
    # carry more). slots expands each GPU by its weight, then work is dealt over the slots.
    slots = [g for g, w in zip(gpus, weights) for _ in range(max(1, w))]
    buckets = {g: [] for g in gpus}
    for i, item in enumerate(work):
        buckets[slots[i % len(slots)]].append(item)
    shards = [(g, buckets[g]) for g in gpus if buckets[g]]
    print(
        f"sharding {len(work)} items over {len(shards)} GPU(s) "
        f"(weights {dict(zip(gpus, weights))}): {[(g, len(s)) for g, s in shards]}",
        flush=True,
    )

    ctxmp = mp.get_context("spawn")
    q = ctxmp.Queue()
    procs = []
    for g, s in shards:
        p = ctxmp.Process(target=worker, args=(g, s, ctx, ts_by_traj, rowdir, q))
        p.start()
        procs.append(p)
    for _ in shards:
        q.get()
    for p in procs:
        p.join()

    aggregate(rowdir, args.outdir, labels)


if __name__ == "__main__":
    main()
