"""CLI: torch DiT ``.pth`` → Equinox ``.eqx``. See ``neugk_jax.translate``."""

from __future__ import annotations

import argparse

import jax.random as jr

from neugk_jax.training.checkpoint import load_model_only, save_model_only
from neugk_jax.translate import (
    build_ae_from_config,
    build_dit_from_config,
    iter_leaves,
    load_torch_state,
    translate_dit as translate,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-ckpt", required=True)
    p.add_argument("--config", required=True, help="DIFF_FLOW config.yaml")
    p.add_argument("--ae-ckpt", required=True, help="translated AE checkpoint")
    p.add_argument("--ae-config", required=True, help="AE config.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    ae_template = build_ae_from_config(args.ae_config, key=jr.PRNGKey(0))
    ae = load_model_only(args.ae_ckpt, ae_template)

    torch_state = load_torch_state(args.torch_ckpt)
    print(f"loaded torch DiT state: {len(torch_state)} keys")
    template = build_dit_from_config(args.config, ae, key=jr.PRNGKey(0))
    print(f"built DiT: latent_shape={template.latent_shape}, cond_dim={template.cond_dim}")
    model, missing, unused = translate(template, torch_state, strict=args.strict)
    total = len(list(iter_leaves(template)))
    print(f"translated leaves: {total - len(missing)} / {total}")
    if missing:
        print(f"missing JAX leaves ({len(missing)}):")
        for n, s in missing[:20]:
            print(f"  {n}  shape={s}")
    if unused:
        print(f"unused torch keys ({len(unused)}):")
        for n in unused[:20]:
            print(f"  {n}  shape={torch_state[n].shape}")
    save_model_only(args.out, model)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
