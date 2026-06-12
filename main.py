"""Hydra entry point for the JAX/Equinox port.

Mirrors the upstream torch ``main.py`` dispatch pattern: read the
Hydra config, build an output directory, log the resolved config, and
hand off to the workflow-appropriate runner. JAX's distributed setup
piggybacks on the same SLURM / torchrun env vars (see
``neugk_jax.training.ddp.init_distributed``), so there's no separate
launcher tier to thread through.

Usage::

    python main.py workflow=ae training.n_epochs=1
    python main.py workflow=diffusion ae_checkpoint=/path/to/ae.eqx
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def dispatch_runner(cfg: DictConfig) -> None:
    """Workflow → runner dispatch. Matches ``neugk.main.dispatch_runner``."""
    workflow = cfg.get("workflow", "ae")
    base = workflow.split("_")[0] if "_" in workflow else workflow
    if base in ("ae", "pinc"):  # accept the upstream label too
        from neugk_jax.autoencoders.runner import AERunner
        AERunner(cfg, output_path=cfg.output_path)()
    elif base == "diffusion":
        from neugk_jax.diffusion.runner import FlowMatchingRunner
        FlowMatchingRunner(cfg, output_path=cfg.output_path)()
    elif base == "gyroswin":
        from neugk_jax.gyroswin import GyroSwinRunner
        GyroSwinRunner(cfg, output_path=cfg.output_path)()
    else:
        raise NotImplementedError(f"unknown workflow: {workflow}")


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rand_suffix = random.randint(0, 999)
    date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S") + f"_{rand_suffix:03d}"

    if cfg.get("output_path") is None:
        cfg.output_path = str(Path("outputs") / date_and_time)
    else:
        cfg.output_path = str(Path(cfg.output_path) / date_and_time)
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, Path(cfg.output_path) / "config.yaml")
    print("#" * 88)
    print("Starting neugk-jax with configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88)
    dispatch_runner(cfg)


if __name__ == "__main__":
    main()
