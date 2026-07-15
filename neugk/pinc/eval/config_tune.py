"""ConFIG tuning harness for the PINC second-stage training.

Standalone script: density warmup (precondition) then PINC stage, with either
weighted-sum or ConFIG multi-objective gradient combination. Computes the
paper metrics (df_l1, phi_l1, flux_absQ) vs ground truth on a trajectory subset.

Usage:
    python -m neugk.pinc.eval.config_tune --gpu 3 --variant baseline
    python -m neugk.pinc.eval.config_tune --gpu 4 --variant config_default
"""

import sys
import argparse
from copy import deepcopy

import torch
from omegaconf import OmegaConf

sys.path.extend([".", ".."])

from neugk.pinc.nf_main import build_data, get_model, density_phase, pinc_loss_weights
from neugk.pinc.neural_fields.nf_train import train_pinc
from neugk.pinc.neural_fields.nf_utils import sample_field
from neugk.pinc.eval.metrics import integrate


SUBSET = [0, 8, 20, 36, 41, 48, 55, 65, 73, 79]
MINISUBSET = [0, 36, 65]


@torch.no_grad()
def snapshot_metrics(model, data, device):
    pred_df = sample_field(model, data, device, timestep=None).to(device)
    gt_df = data.full_df.to(device)
    pred_phi, pred_eflux = integrate(pred_df, data.geom)
    gt_phi, gt_eflux = integrate(gt_df, data.geom)
    return {
        "df_l1": float((pred_df - gt_df).abs().mean()),
        "phi_l1": float((pred_phi - gt_phi).abs().mean()),
        "flux_absQ": float((pred_eflux.sum() - gt_eflux.sum()).abs()),
    }


def run_one(cfg, traj, device, weights):
    data, loader = build_data(cfg, traj, cfg.timestep)
    model = get_model(cfg, data)
    _, best_density, _, _ = density_phase(cfg, model, data, loader, device, verbose=False)

    pinc_model = deepcopy(best_density)
    opt = torch.optim.AdamW(pinc_model.parameters(), cfg.pinc_lr, weight_decay=1e-12)
    sched = None
    if cfg.get("pinc_lr_sched", True):
        from transformers.optimization import get_scheduler

        sched = get_scheduler(
            "cosine_with_min_lr",
            optimizer=opt,
            num_warmup_steps=cfg.pinc_epochs // 5,
            num_training_steps=cfg.pinc_epochs,
            scheduler_specific_kwargs={"min_lr": cfg.get("min_lr", 1e-8)},
        )
    _, best_pinc, _, _ = train_pinc(
        pinc_model,
        n_epochs=cfg.pinc_epochs,
        data=data,
        optim=opt,
        sched=sched,
        device=device,
        use_flux_fields=cfg.use_flux_fields,
        pinc_loss_weight=weights,
        use_print=False,
        eval_every=cfg.get("pinc_eval_every", 2),
        use_config=cfg.get("use_config", False),
        config_op=cfg.get("config_op", "config"),
        config_length=cfg.get("config_length", "projection"),
        config_lstsq=cfg.get("config_lstsq", True),
        config_weights=cfg.get("config_weights", None),
        config_clip=cfg.get("config_clip", None),
        select=cfg.get("pinc_select", "phi"),
    )
    return snapshot_metrics(best_pinc, data, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--use_config", action="store_true")
    ap.add_argument("--pinc_lr", type=float, default=None)
    ap.add_argument("--pinc_epochs", type=int, default=None)
    ap.add_argument("--config_op", type=str, default=None)
    ap.add_argument("--config_length", type=str, default=None)
    ap.add_argument("--config_lstsq", type=str, default=None)
    ap.add_argument("--flux_w", type=float, default=None, help="direction weight on flux")
    ap.add_argument("--select", type=str, default=None, help="phi | balanced")
    ap.add_argument(
        "--clip", type=float, default=None, help="rescale combined |g| to clip*max|g_i|"
    )
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--subset", type=str, default=None, help="comma ids; default 10-subset")
    ap.add_argument("--timestep", type=int, default=200)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    cfg = OmegaConf.load("configs/pinc_revival/nf_train.yaml")
    cfg.epochs = 20
    cfg.pinc_epochs = args.pinc_epochs if args.pinc_epochs else 100
    cfg.timestep = args.timestep
    cfg.use_config = args.use_config
    if args.pinc_lr is not None:
        cfg.pinc_lr = args.pinc_lr
    if args.config_op is not None:
        cfg.config_op = args.config_op
    if args.config_length is not None:
        cfg.config_length = args.config_length
    if args.config_lstsq is not None:
        cfg.config_lstsq = args.config_lstsq.lower() == "true"
    if args.flux_w is not None:
        cfg.config_weights = {"flux": args.flux_w}
    if args.select is not None:
        cfg.pinc_select = args.select
    if args.clip is not None:
        cfg.config_clip = args.clip
    if args.eval_every is not None:
        cfg.pinc_eval_every = args.eval_every

    ids = [int(x) for x in args.subset.split(",")] if args.subset else SUBSET
    weights = pinc_loss_weights(cfg)

    results = []
    for i in ids:
        traj = f"iteration_{i}.h5"
        try:
            m = run_one(cfg, traj, device, weights)
            m["traj"] = i
            results.append(m)
            print(
                f"[{args.variant}] iter_{i}: df={m['df_l1']:.4f} "
                f"phi={m['phi_l1']:.4f} flux={m['flux_absQ']:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[skip] iter_{i}: {type(e).__name__}: {e}", flush=True)

    import statistics as st

    keys = ["df_l1", "phi_l1", "flux_absQ"]
    print(
        f"\n=== {args.variant} (lr={cfg.pinc_lr} ep={cfg.pinc_epochs} "
        f"op={cfg.get('config_op')} len={cfg.get('config_length')} "
        f"lstsq={cfg.get('config_lstsq')}) n={len(results)} ==="
    )
    for k in keys:
        vals = [r[k] for r in results]
        print(f"{k}: mean={st.mean(vals):.4f} median={st.median(vals):.4f}")


if __name__ == "__main__":
    main()
