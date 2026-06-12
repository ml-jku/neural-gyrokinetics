"""CLI: torch AE ``.pth`` → Equinox ``.eqx``. See ``neugk_jax.translate``."""

from __future__ import annotations

import argparse

import jax.random as jr

from neugk_jax.training.checkpoint import save_model_only
from neugk_jax.translate import (
    build_ae_from_config,
    iter_leaves,
    load_torch_state,
    translate_ae as translate,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-ckpt", required=True, help="path to .pth (torch state_dict)")
    p.add_argument("--config", required=True, help="upstream Hydra config.yaml")
    p.add_argument("--out", required=True, help="path to write the equinox checkpoint")
    p.add_argument("--strict", action="store_true",
                   help="abort on any leaf that has no torch counterpart")
    args = p.parse_args()

    torch_state = load_torch_state(args.torch_ckpt)
    print(f"loaded torch state: {len(torch_state)} keys")
    template = build_ae_from_config(args.config, key=jr.PRNGKey(0))
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
