"""CLI: torch ``.pth`` (AE, DiT or GyroSwin) → Equinox ``.eqx``. See ``neugk_jax.translate``."""

from __future__ import annotations

import argparse

import jax.random as jr

from neugk_jax import translate
from neugk_jax.models import build
from neugk_jax.training.checkpoint import load_model_only, save_model_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("ae", "dit", "gyroswin"), default="ae")
    p.add_argument("--torch-ckpt", required=True, help="path to .pth (torch state_dict)")
    p.add_argument("--config", required=True, help="the checkpoint's Hydra config.yaml")
    p.add_argument("--out", required=True, help="path to write the equinox checkpoint")
    p.add_argument("--ae-ckpt", help="dit: translated AE checkpoint")
    p.add_argument("--ae-config", help="dit: AE config.yaml")
    p.add_argument(
        "--strict", action="store_true", help="abort on any leaf that has no torch counterpart"
    )
    p.add_argument("--legacy-swin-shortcut", action="store_true", help="doubled swin residual")
    args = p.parse_args()

    torch_state = translate.load_torch_state(args.torch_ckpt)
    print(f"loaded torch state: {len(torch_state)} keys")
    key = jr.PRNGKey(0)
    legacy = {"legacy_double_shortcut": args.legacy_swin_shortcut or None}
    if args.kind == "dit":
        if not (args.ae_ckpt and args.ae_config):
            p.error("--kind=dit needs --ae-ckpt and --ae-config")
        ae = load_model_only(
            args.ae_ckpt, build.build_ae_from_config(args.ae_config, key=key, **legacy)
        )
        template = build.build_dit_from_config(args.config, ae, key=key)
        print(f"built DiT: latent_shape={template.latent_shape}, cond_dim={template.cond_dim}")
    elif args.kind == "gyroswin":
        template = build.build_gyroswin_from_config(args.config, key=key, **legacy)
    else:
        template = build.build_ae_from_config(args.config, key=key, **legacy)
    fn = {"ae": translate.translate_ae, "dit": translate.translate_dit}.get(
        args.kind, translate.translate_gyroswin
    )
    model, missing, unused = fn(template, torch_state, strict=args.strict)
    translate.report(template, torch_state, missing, unused)
    save_model_only(args.out, model)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
