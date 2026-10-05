"""Per-snapshot rate-distortion records for the traditional codecs (and optionally the NF families).

Each (codec, target CR) is encoded per snapshot at iso-CR, scored with ``ml_eval`` and stored with
its compressed size, so the plot can use the pooled CR (total raw / total compressed bytes) and drop
unreachable targets. CPU-parallel over (codec, CR, trajectory) jobs.
Examples:
  python scripts/pinc/run_scaling_codecs.py --workers 30 --threads 2
  python scripts/pinc/run_scaling_codecs.py --nf nf-pinc --crs 5064,9811 --gpu 4
"""

import os
import sys
import argparse
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from run_scaling import TEST_TRAJS  # noqa: E402

NF_CRS = [10, 50, 200, 498, 1168, 1966, 5064, 9812, 37386]
PATH = "/local00/bioinf/galletti/preprocessed_kvikio"


def _metrics(pred, gt, geom):
    from neugk.pinc.eval.metrics import ml_eval, integrate

    p_phi, p_ef = integrate(pred, geom)
    g_phi, g_ef = integrate(gt, geom)
    m = ml_eval(pred, gt, p_phi, g_phi, p_ef, g_ef)
    return {k: m[k] for k in ("psnr", "l1", "phi_l1", "phi_psnr", "eflux_l1")}


def codec_job(args):
    codec, target, traj, timesteps = args
    import torch
    from neugk.pinc.eval.discovery import CycloneNFDataset
    from neugk.pinc.eval.reconstructors import _encode_at_cr

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    gt = CycloneNFDataset(
        traj, list(timesteps), path=PATH, backend="kvikio", realpotens=True, normalize=None
    )
    rows, warm = [], None
    for i, t in enumerate(timesteps):
        df = gt.full_df[:, i] if gt.full_df.ndim > 6 else gt.full_df
        t0 = time.time()
        recon, size, warm = _encode_at_cr(codec, df, float(target), warm=warm)
        r = dict(traj=traj, t=t, raw=int(df.nbytes), bytes=int(size), knob=warm)
        r.update(_metrics(recon.float(), df.float(), gt.geom))
        r["sec"] = time.time() - t0
        rows.append(r)
    return codec, target, rows


def nf_rows(fam, cr, trajs, timesteps, device, scale_dir):
    import torch
    from neugk.pinc.eval.discovery import CycloneNFDataset
    from neugk.pinc.eval.reconstructors import nf_scaling_reconstructors

    recs = nf_scaling_reconstructors(
        os.path.join(scale_dir, f"{cr}x"), device=device, include_int_only=(fam == "nf-pinc"),
        path=PATH, backend="kvikio",
    )
    rows = []
    for ncr, rec in recs:
        for traj in trajs:
            gt = CycloneNFDataset(
                traj, list(timesteps), path=PATH, backend="kvikio", realpotens=True, normalize=None
            )
            with torch.no_grad():
                dfs, size = rec.reconstruct(traj, timesteps, gt, device)
            geom = {k: v.to(device) for k, v in gt.geom.items()}
            for i, t in enumerate(timesteps):
                r = dict(traj=traj, t=t, raw=int(gt.full_df[:, i].nbytes), bytes=size // len(timesteps))
                r.update(_metrics(dfs[i].to(device).float(), gt.full_df[:, i].to(device).float(), geom))
                rows.append(r)
            print(f"[nf] {fam} x{ncr} {traj} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="wavelet,jpeg2000,pca,sz3,zfp")
    ap.add_argument("--crs", default=",".join(map(str, NF_CRS)))
    ap.add_argument("--trajs", default=",".join(TEST_TRAJS))
    ap.add_argument("--timesteps", default="110,160,210,260")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--nf", default="", help="nf or nf-pinc: score that family instead of codecs")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--scale-dir", default="/system/user/publicwork/galletti/pinc_revival/nf_ckps_scale")
    ap.add_argument("--out", default="/system/user/publicwork/galletti/pinc_revival/scaling_v2/codecs.pkl")
    args = ap.parse_args()
    crs = [int(c) for c in args.crs.split(",") if c]
    trajs = [t for t in args.trajs.split(",") if t]
    ts = [int(t) for t in args.timesteps.split(",") if t]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = pickle.load(open(args.out, "rb")) if os.path.exists(args.out) else {}

    if args.nf:
        import torch

        torch.cuda.set_device(args.gpu)
        for cr in crs:
            res[(args.nf, cr)] = nf_rows(args.nf, cr, trajs, ts, f"cuda:{args.gpu}", args.scale_dir)
            pickle.dump(res, open(args.out, "wb"))
        return

    import multiprocessing as mp

    jobs = [
        (c, cr, tr, ts)
        for c in args.codecs.split(",")
        for cr in crs
        if len(res.get((c, cr), [])) < len(trajs) * len(ts)
        for tr in trajs
    ]
    print(f"{len(jobs)} jobs on {args.workers} workers", flush=True)
    acc = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, (c, cr, rows) in enumerate(pool.imap_unordered(codec_job, jobs), 1):
            acc.setdefault((c, cr), []).extend(rows)
            if len(acc[(c, cr)]) == len(trajs) * len(ts):
                res[(c, cr)] = acc.pop((c, cr))
                raw = sum(r["raw"] for r in res[(c, cr)])
                cb = sum(r["bytes"] for r in res[(c, cr)])
                print(f"[{time.time() - t0:7.0f}s] {c}@{cr}: pooled cr {raw / cb:.1f}", flush=True)
                pickle.dump(res, open(args.out, "wb"))
            if n % 50 == 0:
                print(f"[{time.time() - t0:7.0f}s] {n}/{len(jobs)} jobs", flush=True)


if __name__ == "__main__":
    main()
