"""Command line entry point.

    python -m neugk.gyrosplat fit      --dataset synthetic|gkw [--config recipe.yaml] [key=value ...] --out row.json
    python -m neugk.gyrosplat baseline --dataset ... --method fourier[_gn]|wavelet[_gn]|pca|zfp|sz3|splats|neural_field
                       [--budget N] --out row.json
    python -m neugk.gyrosplat ablate   --dataset ... --control <name> [--config recipe.yaml] [key=value ...] --out row.json
    python -m neugk.gyrosplat table    results/*.json

Dataset-specific arguments (paths, run names, time steps) come from `--data` (a YAML under
`configs/gyrosplat/dataset/`) and from `data.key=value` overrides, which win.  The supervised and
evaluated flux band is the recipe's `band` unless `data.band_lo` / `data.band_hi` say
otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from omegaconf import OmegaConf

from neugk.gyrosplat.config import load_config


def _split_overrides(items: List[str]):
    data, recipe = [], []
    for it in items:
        (data if it.startswith("data.") else recipe).append(it)
    return [d[len("data.") :] for d in data], recipe


def build(
    dataset: str,
    data_overrides: List[str],
    device: str,
    data_yaml=None,
    band: Optional[Sequence[int]] = None,
):
    """Load a snapshot and its backend from a dataset name, an optional dataset YAML
    (`configs/gyrosplat/dataset/*.yaml`) and `data.*` overrides, which win.  `band` is the recipe's
    [lo, hi) flux band, the default when the dataset arguments give none."""
    dcfg = OmegaConf.from_dotlist(data_overrides)
    if data_yaml:
        dcfg = OmegaConf.merge(OmegaConf.load(data_yaml), dcfg)
    lo, hi = (band[0], band[1]) if band else (1, 16)
    if dataset == "synthetic":
        from neugk.gyrosplat.backends.synthetic import SyntheticBackend
        from neugk.gyrosplat.datasets.synthetic import make_synthetic_snapshot

        shape = tuple(dcfg.get("shape", (8, 4, 6, 16, 16)))
        snap = make_synthetic_snapshot(
            shape,
            n_species=int(dcfg.get("n_species", 1)),
            seed=int(dcfg.get("seed", 0)),
            device=device,
        )
        return snap, SyntheticBackend(
            snap, band=list(dcfg.get("band", [1, 2, 3, 4])), device=device
        )
    if dataset == "gkw":
        from neugk.gyrosplat.backends.gkw import GkwBackend
        from neugk.gyrosplat.datasets.gkw import load_gkw_snapshot

        snap = load_gkw_snapshot(
            dcfg.root,
            dcfg.trajectory,
            int(dcfg.timestep),
            device=device,
            raw_root=dcfg.get("raw_root", None),
        )
        return snap, GkwBackend(
            snap,
            band=range(int(dcfg.get("band_lo", lo)), int(dcfg.get("band_hi", hi))),
            device=device,
        )
    raise SystemExit(f"unknown dataset {dataset}")


def _common(sub, name: str, help: str):
    p = sub.add_parser(name, help=help)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None, help="dataset YAML; data.key=value wins")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    return p


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m neugk.gyrosplat",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = _common(sub, "fit", "fit the atom codec")
    f.add_argument("--save", default=None, help="write the fitted model (.pt)")
    b = _common(sub, "baseline", "run a baseline codec at a budget of stored reals")
    b.add_argument("--method", required=True)
    b.add_argument(
        "--budget",
        type=int,
        default=None,
        help="stored reals per species (default: the recipe's)",
    )
    a = _common(sub, "ablate", "one row of the paper's ablation ladder")
    a.add_argument("--control", required=True, help="see gyrosplat.ablation.CONTROLS")
    t = sub.add_parser("table", help="medians over result rows")
    t.add_argument("rows", nargs="+")
    args = p.parse_args(argv)

    if args.cmd == "table":
        from neugk.gyrosplat.eval.tables import print_table

        print_table(args.rows)
        return
    data_ov, recipe_ov = _split_overrides(args.overrides)
    cfg = load_config(args.config, recipe_ov)
    snap, backend = build(args.dataset, data_ov, args.device, args.data, cfg.band)
    if args.cmd == "fit":
        from neugk.gyrosplat.model.pack import save_model
        from neugk.gyrosplat.recipe import compress

        model = compress(backend, cfg, device=args.device)
        out = {
            "dataset": args.dataset,
            "snapshot": snap.name,
            "config": OmegaConf.to_container(OmegaConf.structured(cfg)),
            **model.info,
        }
        if args.save:
            save_model(args.save, model.banks, model.stats, cfg)
    elif args.cmd == "baseline":
        from neugk.gyrosplat.baselines import run_baseline

        out = run_baseline(
            args.method, backend, cfg, budget=args.budget, device=args.device
        )
        out.update({"dataset": args.dataset, "snapshot": snap.name})
    else:
        from neugk.gyrosplat.eval.ablation import run_control

        out = run_control(args.control, backend, cfg, device=args.device)
        out.update({"dataset": args.dataset, "snapshot": snap.name})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"wrote {args.out}")
    else:
        json.dump(
            {k: v for k, v in out.items() if k != "config"},
            sys.stdout,
            indent=1,
            default=float,
        )


if __name__ == "__main__":
    main()
