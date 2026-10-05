"""Hydra entry point for the JAX/Equinox port.

Reads the Hydra config, builds (or, with ``load_ckpt``, reuses) the output
directory and hands off to the workflow runner, which saves the resolved config.
Distributed setup reads SLURM / torchrun env vars
(``neugk_jax.training.ddp.init_distributed``); every process shares the run id of
process 0.

Usage (one experiment preset per workflow, see ``configs/experiment``)::

    python main.py                                   # ae (default preset)
    python main.py experiment=diffusion ae_checkpoint=/path/to/ae_run_dir
    python main.py experiment=gyroswin
    python main.py experiment=pinc_revival ae_checkpoint=/path/to/pretrained_ae_run
    python main.py experiment=vqvae model.vq.quantizer=fsq
    python main.py experiment=ae training.n_epochs=1 logging=wandb
    python main.py load_ckpt=true output_path=/path/to/run_dir   # resume in place
    torchrun --nproc_per_node=2 main.py experiment=ae   # batch_size is per device
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def dispatch_runner(cfg: DictConfig) -> None:
    """Workflow → runner dispatch."""
    workflow = cfg.get("workflow", "ae")
    base = workflow.split("_")[0]
    vq = (cfg.get("model") or {}).get("model_type") == "vqvae"
    if base == "pinc" and cfg.get("stage") == "peft":
        from neugk_jax.pinc.peft import PINCPEFTRunner as Runner
    elif base == "vqvae" or (base in ("ae", "pinc") and vq):
        from neugk_jax.pinc.runner import VQVAERunner as Runner
    elif base in ("ae", "pinc"):
        from neugk_jax.pinc.runner import AERunner as Runner
    elif base == "diffusion":
        from neugk_jax.diffusion.runner import FlowMatchingRunner as Runner
    elif base == "gyroswin":
        from neugk_jax.gyroswin import GyroSwinRunner as Runner
    else:
        raise NotImplementedError(f"unknown workflow: {workflow}")
    Runner(cfg, output_path=cfg.output_path)()


def _drop_cli_overridden(cli: list[str], source: DictConfig, prefix: str = "") -> None:
    keys = {c.split("=")[0].lstrip("+~") for c in cli}
    for k in list(source.keys()):
        path = f"{prefix}.{k}" if prefix else str(k)
        if path in keys:
            del source[k]
        elif OmegaConf.is_dict(source[k]):
            _drop_cli_overridden(cli, source[k], path)


def resume_config(cfg: DictConfig) -> DictConfig:
    """Config for resuming ``cfg.output_path`` in place: its saved config, CLI overrides on top."""
    run = Path(cfg.output_path or "")
    if not (run / "ckp.eqx").exists():
        raise FileNotFoundError(f"load_ckpt=true but {run}/ckp.eqx does not exist")
    saved = OmegaConf.load(run / "config.yaml")
    cli = list(HydraConfig.get().overrides.task) if HydraConfig.initialized() else []
    _drop_cli_overridden(cli, saved)
    return OmegaConf.merge(cfg, saved)


def run_id() -> str:
    """``YYYYmmdd_HHMMSS_<rand>`` of process 0, identical on every process."""
    from neugk_jax.training.ddp import init_distributed

    dist = init_distributed()
    now = datetime.today()
    stamp = np.asarray(
        [int(now.strftime("%Y%m%d")), int(now.strftime("%H%M%S")), random.randint(0, 999)], np.int32
    )
    if dist.num_processes > 1:
        from jax.experimental import multihost_utils

        stamp = np.asarray(multihost_utils.broadcast_one_to_all(stamp))
    day, time, rand = (int(v) for v in stamp)
    return f"{day:08d}_{time:06d}_{rand:03d}"


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if cfg.get("load_ckpt"):
        cfg = resume_config(cfg)
    else:
        cfg.output_path = str(Path(cfg.get("output_path") or "outputs") / run_id())
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)
    print("#" * 88)
    print("Starting neugk-jax with configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88)
    dispatch_runner(cfg)
    import jax

    if jax.distributed.is_initialized():
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
