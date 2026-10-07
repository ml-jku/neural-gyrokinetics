"""Representation interpolation of the PINC models.

For two snapshots ``t`` and ``t + 2 * half`` of a trajectory, the midpoint ``t + half`` is predicted by
linearly interpolating the representations at the extremes: the weights of neural fields trained from a
shared per-trajectory initialization (configs/pinc_revival/nf_train_interp.yaml), and the latents of the
autoencoders (VQ-VAE latents before quantization). Neural fields are denormalized with the mean of the
two extremes' statistics. ``Extremes`` scores the first extreme and ``f (data)`` the data-space average.
PSNR and L1 of f are those of the main evaluation.

    python scripts/pinc/run_interp.py --nf-ckpts nf_ckps_interp \\
        --path /system/user/publicwork/galletti/pinc_revival_eval --gpu 0
"""

import argparse
import json
import os
import re
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from neugk.pinc.autoencoders.ae_utils import load_autoencoder
from neugk.pinc.eval.ae_data import AE_CKPTS, build_make_val_dataset
from neugk.pinc.neural_fields.data import CycloneNFDataset
from neugk.pinc.neural_fields.nf_utils import load_nf, sample_field
from neugk.utils import recombine_zf

NF_ROWS = {"NF": "best_mlp", "PINC-NF": "best_int_mlp"}


def metrics(pred, gt):
    pred = pred.reshape(gt.shape).to(gt.device, torch.float32)
    mse = ((pred - gt) ** 2).mean()
    return {"psnr": float(10 * torch.log10(gt.max() ** 2 / mse)), "l1": float((pred - gt).abs().mean())}


def nf_checkpoint(ckpt_dir, prefix, traj, t):
    rx = re.compile(rf"^{prefix}_{traj}_t{t}_x\d+\.pt$")
    hits = [f for f in os.listdir(ckpt_dir) if rx.match(f)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{prefix}_{traj}_t{t} in {ckpt_dir}: {hits}")
    return os.path.join(ckpt_dir, hits[0])


def nf_midpoint(ckpt_dir, prefix, traj, ta, tb, path, backend, device):
    data = [
        CycloneNFDataset(traj, timesteps=t, path=path, backend=backend, realpotens=True,
                         normalize="zscore", normalize_coords=False)
        for t in (ta, tb)
    ]
    nfs = [load_nf(nf_checkpoint(ckpt_dir, prefix, traj, t), device, grid_size=d.grid_size)
           for t, d in zip((ta, tb), data)]
    sa, sb = nfs[0].state_dict(), nfs[1].state_dict()
    nfs[0].load_state_dict({k: 0.5 * (sa[k] + sb[k]) for k in sa})
    stats = SimpleNamespace(
        grid=data[0].grid,
        ndim=data[0].ndim,
        scale={"df": 0.5 * (data[0].scale["df"] + data[1].scale["df"])},
        shift={"df": 0.5 * (data[0].shift["df"] + data[1].shift["df"])},
    )
    return sample_field(nfs[0].to(device).eval(), stats, device)


def vq_continuous(model, df, condition):
    """Pre-quantization latent of a Swin5DVQVAE (its ``encode`` up to the quantizer)."""
    if condition is not None and condition.shape[-1] != model.enc_cond_dim:
        condition = model.condition(
            {"condition": condition}, model.enc_cond_embed, model.encoder_condition_keys,
            indices=model.enc_indices,
        ).get("condition")
    kwcond = {"condition": condition} if condition is not None else {}
    z, pad_axes = model.patch_encode(df)
    for blk in model.down_blocks:
        z = blk(z, return_skip=False, **kwcond)
    if hasattr(model, "middle_pe"):
        z = model.middle_pe(z)
    z = model.middle_pre(z, **kwcond)
    return model.middle_vq_downproj(z), pad_axes


def ae_midpoint(model, val, ta, tb, vqvae, device):
    sa, sb = val[ta], val[tb]
    cond = sa.conditioning.unsqueeze(0).to(device)
    xa, xb = sa.df.unsqueeze(0).to(device), sb.df.unsqueeze(0).to(device)
    if vqvae:
        (za, pad), (zb, _) = vq_continuous(model, xa, cond), vq_continuous(model, xb, cond)
        z = 0.5 * (za + zb)
        z = model.vq(z.view(z.shape[0], -1, z.shape[-1]))[0].view(z.shape)
    else:
        (za, pad), (zb, _) = model.encode(xa, condition=cond), model.encode(xb, condition=cond)
        z = 0.5 * (za + zb)
    df = val.denormalize(0, df=model.decode(z, pad, condition=cond)["df"].squeeze(0).cpu())
    return recombine_zf(df, dim=0) if df.shape[0] == 4 else df


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pinc_revival/nf_train_interp.yaml")
    ap.add_argument("--nf-ckpts", default="nf_ckps_interp")
    ap.add_argument("--path", default="/system/user/publicwork/galletti/pinc_revival_eval")
    ap.add_argument("--backend", default="kvikio")
    ap.add_argument("--ae", default="AE-PRETRAIN,PINC-AE-JAX,VQ-VAE-77K,PINC-VQ-VAE-77K")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/system/user/publicwork/galletti/pinc_jax/interp_results.json")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    cfg = OmegaConf.load(args.config)
    trajs = [t.replace(".h5", "") for t in cfg.trajectory]
    ext = [int(t) for t in cfg.timesteps]
    pairs = [(a, b, (a + b) // 2) for a, b in zip(ext[:-1], ext[1:])]

    aes = {}
    for label in [a for a in args.ae.split(",") if a]:
        ckpt, load_peft, vqvae = AE_CKPTS[label]
        model, _, ae_cfg = load_autoencoder(ckpt, device, model=None, load_peft=load_peft)
        aes[label] = (model.to(device).eval(), build_make_val_dataset(ae_cfg, args.path, args.backend), vqvae)

    rows = []
    for traj in trajs:
        vals = {label: make(f"{traj}.h5") for label, (_, make, _) in aes.items()}
        for ta, tb, tc in pairs:
            gt = {
                t: CycloneNFDataset(traj, timesteps=t, path=args.path, backend=args.backend,
                                    realpotens=True, normalize=None).full_df.to(device)
                for t in (ta, tb, tc)
            }
            row = {"traj": traj, "t": tc,
                   "Extremes": metrics(gt[ta], gt[tc]),
                   "f (data)": metrics(0.5 * (gt[ta] + gt[tb]), gt[tc])}
            for name, prefix in NF_ROWS.items():
                pred = nf_midpoint(args.nf_ckpts, prefix, traj, ta, tb, args.path, args.backend, device)
                row[f"{name} (weights)"] = metrics(pred, gt[tc])
            for label, (model, _, vqvae) in aes.items():
                row[f"{label} (latents)"] = metrics(ae_midpoint(model, vals[label], ta, tb, vqvae, device), gt[tc])
            rows.append(row)
            print(traj, tc, {k: round(v["psnr"], 2) for k, v in row.items() if isinstance(v, dict)}, flush=True)
            torch.cuda.empty_cache()

    agg = defaultdict(dict)
    for key in [k for k in rows[0] if isinstance(rows[0][k], dict)]:
        for m in ("psnr", "l1"):
            v = np.array([r[key][m] for r in rows])
            agg[key][m] = [float(v.mean()), float(v.std())]
    json.dump({"pairs": [list(p) for p in pairs], "trajectories": trajs, "agg": agg, "rows": rows},
              open(args.out, "w"), indent=1)
    for key, v in agg.items():
        print(f"{key:28s} & {v['psnr'][0]:.1f}$_{{\\pm {v['psnr'][1]:.1f}}}$ \\\\")


if __name__ == "__main__":
    main()
