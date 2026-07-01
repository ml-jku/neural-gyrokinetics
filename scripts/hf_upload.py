"""Upload the gyrokinetics datasets + checkpoints to the HuggingFace Hub.

Two targets:
- pinc_gkw (existing public repo): the compression test set in FLOAT32 -- 60 held-out
  turbulent trajectories (10x-downsampled turbulent window) plus the trained neural-field
  checkpoints (PINC + PINN), so users can reproduce the paper compression results. No AE.
- full (new repo): the COMPLETE trajectory set in BFLOAT16 (the .bf16.bin shards), the
  compact archive of every trajectory with data.

Dry-run by default (these are multi-TB uploads): it lists what would go up and
the total size without touching the hub. Pass --execute to actually upload.

  python scripts/hf_upload.py --target pinc_gkw            # dry run (f32 test set + NFs)
  python scripts/hf_upload.py --target full --execute       # real upload (bf16, everything)
auth: uses --token or $HF_TOKEN.
"""
import os
import re
import sys
import argparse

from huggingface_hub import HfApi

DATA_ROOT = "/local00/bioinf/galletti/preprocessed_kvikio"
PINC_GKW_REPO = "gerkone/pinc_gkw"
# schematic: gyrokinetic model, adiabatic electrons, full trajectory count
FULL_REPO_DEFAULT = "gerkone/cbc-gyroswin-256traj"  # 251 in-distribution + 5 ood
TRAJ_SUFFIX = "_ifft_realpotens"
RAW_ROOT = "/restricteddata/ukaea/gyrokinetics/raw"  # holds per-trajectory input.dat
# eval set: the 18 timesteps per traj the neural fields were fit on, 10x-downsampled
# turbulent window [90,260] step 10 (matches nf_ckps_revival: 60 trajs x 18 = 1080)
TEST_TIMESTEPS = list(range(90, 261, 10))

# 60 held-out turbulent test trajectories (matches configs/dataset/pinc.yaml)
TEST_IDS = [0, 8, 20, 36, 41, 48, 55, 65, 73, 79, 85, 94, 100, 104, 108, 113, 117,
            121, 125, 130, 134, 138, 142, 146, 151, 155, 159, 163, 168, 172, 176,
            180, 185, 189, 193, 197, 202, 206, 210, 214, 218, 223, 227, 231, 235,
            240, 244, 248, 252, 257, 261, 265, 269, 274, 278, 282, 286, 291, 295, 299]

# neural-field checkpoints -> (path within repo, allow_patterns). We ship ONLY the best basic NF
# (best_mlp_*) and the best NF-PINC (best_int_mlp_*) from nf_ckps_revival -- not the last-epoch
# (mlp_*/int_mlp_*) snapshots, the autoencoder, or the (incomplete) PINN baseline. Can be extended later.
CHECKPOINTS = {
    "nf_ckps_revival": ("checkpoints/nf", ["best_mlp_*.pt", "best_int_mlp_*.pt"]),
}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


# the data shards come in two formats per snapshot: f32 (timestep_NNNNN.bin / poten_NNNNN.bin)
# and a bfloat16 sibling (..bf16.bin). pinc_gkw ships f32, the full repo ships bf16.
F32_SHARD = re.compile(r"_\d{5}\.bin$")  # f32 data shard; the .bf16.bin sibling does NOT match
# upload-pattern complements: drop the OTHER format's data shards (metadata/stats kept either way)
IGNORE_F32 = ["*.bf16.bin"]        # keep f32 shards, drop bf16
IGNORE_BF16 = ["*_?????.bin"]      # keep bf16 shards, drop f32 (5-digit f32 names only)


def _skip(fmt):
    """file predicate: True for data shards of the OTHER format (excluded from the size report)."""
    if fmt == "bf16":
        return lambda f: bool(F32_SHARD.search(f))      # skip f32 shards
    return lambda f: f.endswith(".bf16.bin")            # skip bf16 shards (f32 archive)


def dir_size(path, fmt="f32"):
    skip = _skip(fmt)
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            if skip(f):
                continue
            try:
                tot += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return tot


def has_data(d):
    # a real trajectory has df shards under data/; the zero-flux ones are metadata-only
    dd = os.path.join(d, "data")
    try:
        return os.path.isdir(dd) and any(
            f.startswith("timestep_") and f.endswith(".bin") for f in os.listdir(dd)
        )
    except OSError:
        return False


def has_bf16(d):
    # a trajectory ready for the bf16 archive has the .bf16.bin shards (the 60-traj
    # compression test set + ood were only kept as f32, so they lack them)
    dd = os.path.join(d, "data")
    try:
        return os.path.isdir(dd) and any(f.endswith(".bf16.bin") for f in os.listdir(dd))
    except OSError:
        return False


def raw_input_dat(base):
    # map iteration_N_ifft_realpotens -> its gkw input.dat; ood lives under raw/ood/
    name = base.split("_ifft")[0]
    cands = [f"{RAW_ROOT}/{name}/input.dat", f"{RAW_ROOT}/{name}_Lin/input.dat"]
    if name.startswith("ood_"):
        cands.append(f"{RAW_ROOT}/ood/{name[4:]}/input.dat")  # raw/ood/iteration_N
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    return None


def upload_aux(api, repo, bases, execute):
    # pair each trajectory with its gkw input.dat and attach the dataset-level
    # generation config, so the data_generation folder is captured in the dataset
    # pair each trajectory with its gkw input.dat (the reproducibility source).
    # gyaradax config yaml + initial-condition dump are pending the data decision.
    found = [b for b in bases if raw_input_dat(b)]
    print(f"  input.dat paired: {len(found)}/{len(bases)} trajectories")
    if not execute:
        return
    for b in found:
        api.upload_file(path_or_fileobj=raw_input_dat(b), path_in_repo=f"{b}/input.dat",
                        repo_id=repo, repo_type="dataset")


def ckpt_size(local, allow):
    """bytes of the files in a checkpoint dir matching any of the allow globs (by basename)."""
    import fnmatch
    tot = n = 0
    for r, _, fs in os.walk(local):
        for f in fs:
            if any(fnmatch.fnmatch(f, pat) for pat in allow):
                try:
                    tot += os.path.getsize(os.path.join(r, f)); n += 1
                except OSError:
                    pass
    return tot, n


def report(label, paths, fmt="f32"):
    total = 0
    print(f"  [{label}]")
    for p in paths:
        if os.path.exists(p):
            s = dir_size(p, fmt=fmt) if os.path.isdir(p) else os.path.getsize(p)
            total += s
            print(f"    {human(s):>9}  {p}")
        else:
            print(f"    {'MISSING':>9}  {p}")
    print(f"    {'-'*9}")
    print(f"    {human(total):>9}  total")
    return total


def card_pinc_gkw():
    return (
        "---\nlicense: mit\ntags: [plasma-physics, gyrokinetics, turbulence, "
        "neural-compression]\n---\n\n"
        "# Gyrokinetics compression validation dataset (PINC)\n\n"
        "Subsampled 5D gyrokinetics test set to reproduce the compression evaluation of "
        "**Physics-Informed Neural Compression of High-Dimensional Plasma Data** (PINC).\n\n"
        "- Paper: https://arxiv.org/abs/2602.04758v2\n"
        "- Code: https://github.com/ml-jku/neural-gyrokinetics\n"
        "- Blog: https://ml-jku.github.io/blog/2026/pinc/\n\n"
        "## Data\n"
        "60 held-out turbulent trajectories of adiabatic-electron gyrokinetic simulations (GKW), "
        "10x-downsampled over the turbulent window (timesteps 90-260, step 10) to 1080 snapshots, "
        "stored in **float32**. Layout:\n\n"
        "```\n"
        "iteration_<n>_ifft_realpotens/\n"
        "  data/timestep_<t>.bin   # 5D distribution function f at timestep t\n"
        "  data/poten_<t>.bin      # electrostatic potentials at timestep t\n"
        "  metadata_light.pkl, data_source.txt   # geometry, grid, spectra metadata\n"
        "checkpoints/nf/\n"
        "  best_mlp_*.pt      # best basic neural field, one per snapshot\n"
        "  best_int_mlp_*.pt  # best PINC (physics-informed) neural field, one per snapshot\n"
        "```\n"
        "The complete trajectory set (all trajectories, bfloat16) is at "
        f"[{FULL_REPO_DEFAULT}](https://huggingface.co/datasets/{FULL_REPO_DEFAULT}).\n\n"
        "## Usage with PINC\n"
        "```bash\n"
        "git clone https://github.com/ml-jku/neural-gyrokinetics && cd neural-gyrokinetics\n"
        "```\n"
        "```python\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download('{PINC_GKW_REPO}', repo_type='dataset', local_dir='data/pinc_gkw')\n"
        "```\n"
        "Reproduce the compression evaluation (neural fields + traditional baselines) through the "
        "shared metrics pipeline:\n"
        "```bash\n"
        "python scripts/run_eval1k.py --methods nf,nf-pinc,sz3,jpeg2000,zfp \\\n"
        "    --ckpts data/pinc_gkw/checkpoints/nf --path data/pinc_gkw\n"
        "```\n"
        "See the repository README for the full PINC training and evaluation pipeline.\n"
    )


def card_full(n_traj, repo=FULL_REPO_DEFAULT):
    return (
        "---\nlicense: cc-by-4.0\ntags: [plasma-physics, gyrokinetics, turbulence, "
        "scientific-data]\n---\n\n"
        f"# Gyrokinetic adiabatic-electron turbulence ({n_traj} trajectories)\n\n"
        "Adiabatic-electron gyrokinetic turbulence simulations (GKW): the full 5D distribution "
        "function and electrostatic potentials at every timestep of each trajectory, stored in "
        "**bfloat16**. This is the dataset used to train the GyroSwin neural surrogates.\n\n"
        "- Source paper: **GyroSwin** (https://arxiv.org/abs/2510.07314), "
        "blog https://ml-jku.github.io/blog/2025/gyroswin/\n"
        "- Code: https://github.com/ml-jku/neural-gyrokinetics\n"
        "- Physics solver: [gyaradax](https://github.com/gerkone/gyaradax)\n\n"
        "## Structure\n"
        "```\n"
        "iteration_<n>_ifft_realpotens/\n"
        "  data/timestep_<t>.bf16.bin  # 5D distribution function f at timestep t (bf16)\n"
        "  data/poten_<t>.bf16.bin     # electrostatic potentials at timestep t (bf16)\n"
        "  metadata_light.pkl  # geometry, grid, spectra metadata\n"
        "  input.dat                   # exact GKW input deck for the trajectory\n"
        "```\n"
        "Trajectories `ood_iteration_*` are the out-of-distribution cases; the rest are "
        "in-distribution. Zero-flux (non-turbulent) trajectories are not included.\n\n"
        "## Usage\n"
        "```python\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download('{repo}', repo_type='dataset', local_dir='gk_full')\n"
        "```\n"
        "Loading and training utilities (the `CycloneDataset` reader, GyroSwin models, configs) are in "
        "the code repository: https://github.com/ml-jku/neural-gyrokinetics.\n\n"
        "## Related\n"
        "GyroSwin surrogates: "
        "[small](https://huggingface.co/ml-jku/gyroswin_small) | "
        "[medium](https://huggingface.co/ml-jku/gyroswin_medium) | "
        "[large](https://huggingface.co/ml-jku/gyroswin_large). "
        "PINC compression test set (float32) + neural-field checkpoints: "
        f"[{PINC_GKW_REPO}](https://huggingface.co/datasets/{PINC_GKW_REPO}).\n"
    )


def write_card(api, repo, text, execute):
    if not execute:
        return
    api.upload_file(
        path_or_fileobj=text.encode(), path_in_repo="README.md",
        repo_id=repo, repo_type="dataset",
    )


def do_pinc_gkw(api, execute):
    print(f"=== pinc_gkw ({PINC_GKW_REPO}): test split + checkpoints ===")
    # only the eval-used (10x-downsampled) timesteps per test traj, not all ~266
    allow, sel_bytes, n_samp = [], 0, 0
    for i in TEST_IDS:
        base = f"iteration_{i}{TRAJ_SUFFIX}"
        d = os.path.join(DATA_ROOT, base)
        for ti in TEST_TIMESTEPS:
            n_samp += 1
            for pre in ("timestep", "poten"):
                allow.append(f"{base}/data/{pre}_{ti:05d}.bin")
                fp = os.path.join(d, "data", f"{pre}_{ti:05d}.bin")
                sel_bytes += os.path.getsize(fp) if os.path.exists(fp) else 0
        # ship only the lightweight metadata; the 446 MB metadata.pkl is df normalization-stat
        # arrays (df_mean/std/var/min/max) that the loader never reads (it recomputes from data).
        allow += [f"{base}/metadata_light.pkl", f"{base}/data_source.txt"]
    # normalization stats are NOT shipped (recomputed by the loader); do not upload *_stats.pkl
    print(f"  test data: {n_samp} snapshots over {len(TEST_IDS)} trajs "
          f"(18 timesteps {TEST_TIMESTEPS[0]}-{TEST_TIMESTEPS[-1]} step 10) = "
          f"{human(sel_bytes)} of df/phi (+ metadata), f32")
    print("  [checkpoints]")
    ck_total = 0
    for local, (dst, allow_ck) in CHECKPOINTS.items():
        if os.path.exists(local):
            s, n = ckpt_size(local, allow_ck)
            ck_total += s
            print(f"    {human(s):>9}  {dst}  ({n} files: {', '.join(allow_ck)})")
        else:
            print(f"    {'MISSING':>9}  {local}")
    print(f"    {'-'*9}\n    {human(ck_total):>9}  total")
    if execute:
        api.create_repo(PINC_GKW_REPO, repo_type="dataset", exist_ok=True)
        api.upload_large_folder(repo_id=PINC_GKW_REPO, repo_type="dataset",
                                folder_path=DATA_ROOT, allow_patterns=allow,
                                ignore_patterns=IGNORE_F32)
        for local, (dst, allow_ck) in CHECKPOINTS.items():
            if os.path.exists(local):
                api.upload_folder(repo_id=PINC_GKW_REPO, repo_type="dataset",
                                  folder_path=local, path_in_repo=dst,
                                  allow_patterns=allow_ck)
        write_card(api, PINC_GKW_REPO, card_pinc_gkw(), execute)
        print("  uploaded.")
    else:
        print("  (dry run; pass --execute to upload)")


def do_full(api, execute, repo, cleanup=False, threads=16):
    import glob
    # used (non-zero-flux) in-distribution trajs + ood; skip the zero-flux ones
    # (metadata only, no df shards) so we do not upload empty trajectories
    cand = sorted(glob.glob(os.path.join(DATA_ROOT, f"iteration_*{TRAJ_SUFFIX}")) +
                  glob.glob(os.path.join(DATA_ROOT, f"ood_iteration_*{TRAJ_SUFFIX}")))
    with_data = [d for d in cand if has_data(d)]
    # cleanup mode uploads EVERY f32 trajectory, converting its bf16 on the fly (the disk is too full
    # to hold all bf16 at once, and ~50/250 have no/partial bf16). non-cleanup uploads only trajs that
    # already have bf16 on disk.
    dirs = with_data if cleanup else [d for d in with_data if has_bf16(d)]
    bases = [os.path.basename(d) for d in dirs]
    n_ood = sum("ood_" in b for b in bases)
    mode = "convert-on-the-fly + per-traj cleanup" if cleanup else "preexisting bf16"
    print(f"=== full dataset ({repo}): {len(dirs)} trajs ({n_ood} ood); "
          f"skipped {len(cand)-len(with_data)} empty/zero-flux; mode={mode}, BFLOAT16 ===", flush=True)
    if not execute:
        if not cleanup:
            report(f"data (bf16 on disk, {len(dirs)} trajs)", dirs, fmt="bf16")
        upload_aux(api, repo, bases, execute=False)
        print("  (dry run; pass --execute to upload)")
        return
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    if cleanup:
        # space-bounded parallel path for a near-full disk: a pool of `threads` workers each, per
        # trajectory, (1) converts its bf16 shards if missing (idempotent), (2) uploads bf16 +
        # metadata + input.dat, (3) deletes the trajectory's .bf16.bin. Each worker holds at most one
        # trajectory's bf16 between its convert and delete, so peak extra disk is ~threads x 11 GB and
        # the ~2.8 TB of existing/new bf16 is reclaimed as the push proceeds (ends f32-only). Upload
        # raises before delete, so a failed upload never deletes; per-traj errors are isolated and an
        # interrupted run just re-converts + re-uploads (HF hash-dedups what is already up).
        from neugk.dataset.preprocess import convert_trajs_to_bf16
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        lock, st = threading.Lock(), {"freed": 0, "ok": 0, "err": 0}

        def process(b, d):
            convert_trajs_to_bf16([d], num_workers=1, force=False)  # fill any missing bf16 shards
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=d, path_in_repo=b,
                              ignore_patterns=IGNORE_BF16 + ["metadata.pkl"])  # bf16 + metadata_light; drop f32 + heavy stats
            if raw_input_dat(b):
                api.upload_file(path_or_fileobj=raw_input_dat(b), path_in_repo=f"{b}/input.dat",
                                repo_id=repo, repo_type="dataset")
            nrm = brm = 0
            for r, _, fs in os.walk(os.path.join(d, "data")):
                for f in fs:
                    if f.endswith(".bf16.bin"):
                        p = os.path.join(r, f)
                        brm += os.path.getsize(p); os.remove(p); nrm += 1
            return b, nrm, brm

        print(f"  parallel convert+upload+cleanup, {threads} workers (peak ~{threads * 11} GB bf16 on disk)", flush=True)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(process, b, d): b for b, d in zip(bases, dirs)}
            for fut in as_completed(futs):
                b = futs[fut]
                try:
                    b, nrm, brm = fut.result()
                    with lock:
                        st["freed"] += brm; st["ok"] += 1
                        print(f"  [{st['ok'] + st['err']}/{len(dirs)}] {b}: ok, -{nrm} bf16 "
                              f"({brm/1e9:.1f} GB, {st['freed']/1e12:.2f} TB reclaimed)", flush=True)
                except Exception as e:
                    with lock:
                        st["err"] += 1
                        print(f"  [{st['ok'] + st['err']}/{len(dirs)}] {b}: ERROR {type(e).__name__}: {e}", flush=True)
        write_card(api, repo, card_full(st["ok"], repo), execute)
        print(f"  done: {st['ok']} ok, {st['err']} err, {st['freed']/1e12:.2f} TB bf16 reclaimed from {DATA_ROOT}")
    else:
        allow = [f"{b}/**" for b in bases]
        # keep the bf16 shards (+ metadata/stats), drop the f32 siblings
        api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=DATA_ROOT,
                                allow_patterns=allow, ignore_patterns=IGNORE_BF16)
        upload_aux(api, repo, bases, execute=True)
        write_card(api, repo, card_full(len(dirs), repo), execute)
        print("  uploaded.")


def main():
    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["pinc_gkw", "full", "both"], default="both")
    ap.add_argument("--full-repo", default=FULL_REPO_DEFAULT)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--execute", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--cleanup-bf16", action="store_true",
                    help="full target only: convert each trajectory's bf16 on the fly, upload, then "
                         "delete its bf16 shards (bounded disk; uploads ALL f32 trajs, reclaims ~2.8 TB)")
    ap.add_argument("--threads", type=int, default=16,
                    help="parallel convert+upload workers for --cleanup-bf16 (peak ~threads x 11 GB on disk)")
    args = ap.parse_args()
    DATA_ROOT = args.data_root
    if args.execute and not args.token:
        sys.exit("no HF token: set $HF_TOKEN or pass --token")
    api = HfApi(token=args.token)
    if args.target in ("pinc_gkw", "both"):
        do_pinc_gkw(api, args.execute)
    if args.target in ("full", "both"):
        do_full(api, args.execute, args.full_repo, cleanup=args.cleanup_bf16, threads=args.threads)


if __name__ == "__main__":
    main()
