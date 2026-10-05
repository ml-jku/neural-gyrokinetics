"""CLI: JAX PINC-AE PEFT checkpoint -> torch ``{model_state_dict, epoch, loss}`` + ``config.yaml``.

The adapters are merged into the base weights (``--peft-format`` keeps them as peft keys, for a
strategy-selected adapter set only); keys and shapes follow the run's base AE (``ae_checkpoint``,
a torch ``.pth`` or a JAX ``.eqx``), whose config is written out with the dataset resolution of
the checkpoint.

    python scripts/export_pinc_torch.py <jax_run_dir> --out <dir> [--ckpt best.eqx] [--peft-format]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.random as jr
import yaml
from omegaconf import OmegaConf

from neugk_jax.models.build import build_ae_from_config, run_config
from neugk_jax.training.checkpoint import load_checkpoint, read_checkpoint_meta, resolve_checkpoint
from neugk_jax.translate import (
    LORA_DEFAULT_STRATEGY,
    ae_state_template,
    attach_ae_lora,
    export_ae_state,
    load_torch_state,
    save_torch_checkpoint,
)
from neugk_jax.utils import to_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="jax pinc peft run directory")
    p.add_argument("--ckpt", default="best.eqx", help="checkpoint file inside the run directory")
    p.add_argument("--out", required=True, help="output directory (best.pth + config.yaml)")
    p.add_argument("--peft-format", action="store_true", help="keep the adapters as peft keys")
    args = p.parse_args()

    run, out = Path(args.run), Path(args.out)
    cfg = OmegaConf.load(run / "config.yaml")
    lora_cfg = to_dict(cfg.model.peft.lora)
    if args.peft_format and (lora_cfg.get("target_modules") or lora_cfg.get("exclude_patterns")):
        p.error(
            "--peft-format: the reloaded adapters follow model.peft.lora.strategy alone; "
            "export merged or train without target_modules / exclude_patterns"
        )
    base_file = resolve_checkpoint(cfg.ae_checkpoint)
    base_cfg = to_dict(base_file.parent / "config.yaml")
    resolution = read_checkpoint_meta(run / args.ckpt).get("resolution")
    resolution = resolution or base_cfg.get("dataset", {}).get("resolution")
    if not resolution:
        p.error("no dataset resolution in the checkpoint meta or the base run config")
    base = build_ae_from_config(run_config(cfg), key=jr.PRNGKey(0), resolution=resolution)
    template = attach_ae_lora(base, lora_cfg, key=jr.PRNGKey(0))
    state = load_checkpoint(run / args.ckpt, template)
    if base_file.suffix == ".eqx":
        base_state = ae_state_template(base)
    else:
        base_state = load_torch_state(str(base_file))
    sd = export_ae_state(state.model, base_state, peft_format=args.peft_format)
    out.mkdir(parents=True, exist_ok=True)
    stage = "peft" if args.peft_format else "autoencoder"
    save_torch_checkpoint(
        str(out / "best.pth"), sd, epoch=state.epoch, loss=float(state.loss), stage=stage
    )
    base_cfg.update(output_path=str(out), exported_from=str(run / args.ckpt))
    base_cfg.setdefault("dataset", {})["resolution"] = list(resolution)
    if args.peft_format:
        strategy = lora_cfg.get("strategy", LORA_DEFAULT_STRATEGY)
        lora = {**lora_cfg, "strategy": strategy, "lora_dropout": 0.0, "bias": "none"}
        base_cfg["stage"] = "peft"
        base_cfg["model"]["peft"] = {"method": "lora", "lora": lora}
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)
    print(f"exported {len(sd)} tensors (epoch {state.epoch}, val {state.loss:.4e}) to {out}")


if __name__ == "__main__":
    main()
