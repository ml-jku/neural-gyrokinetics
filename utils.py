from typing import Dict, Tuple
import glob
import zipfile
import os
import random
import importlib

import torch
from torch import nn
import numpy as np
from omegaconf import DictConfig, OmegaConf


def wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper():
        print(
            "Not using wandb for logging, if this is not intended, unset WANDB_DISABLED env var"
        )
        return False
    return importlib.util.find_spec("wandb") is not None


assert (
    wandb_available()
), "wandb is not installed but is selected as default for logging, please install via pip install wandb"
import wandb


class WandbManager:
    def __init__(self) -> None:
        self._initialized = False

    def setup(self, args, **kwargs):
        if not isinstance(args, dict):
            args = args.__dict__
        project_name = args["logging"].get("project", "debug")

        combined_dict = {**args, **kwargs}
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            entity=args["logging"].get("entity", None),
            # track hyperparameters and run metadata
            config=combined_dict,
            id=args["logging"].get("run_id", None),
            resume="allow" if args["logging"].get("run_id", None) else False,
            reinit=False,
        )
        self._initialized = True

    def log(self, logs, commit=True):
        wandb.log(logs, commit=commit)

    def close(self):
        pass

    def summarize(self, outputs):
        # add values to the wandb summary => only works for scalars
        for k, v in outputs.items():
            self._wandb.run.summary[k] = v.item()


def setup_logging(config):
    if config.logging.writer == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=f"./logs/tiger_exp{config['exp_id']}")
    elif config.logging.writer == "wandb":
        if config.logging.mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        writer = WandbManager()
        merged_config = OmegaConf.to_container(config)
        writer.setup(merged_config)
    elif config.logging.writer is None:
        return None
    else:
        raise NotImplementedError("Specified writer not recognized!")
    return writer


def save_model_and_config(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    epoch: int,
    val_loss: float,
    loss_val_min: float,
) -> None:
    # create directory if it s not there
    os.makedirs(cfg.ckpt_path, exist_ok=True)

    with open(os.path.join(cfg.ckpt_path, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        },
        f"{cfg.ckpt_path}/ckp.pth",
    )

    if val_loss < loss_val_min:
        loss_val_min = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            },
            f"{cfg.ckpt_path}/best.pth",
        )

    return loss_val_min


def load_model_and_config(
    ckp_path: str, model: nn.Module, device: torch.DeviceObjType
) -> Tuple[nn.Module, Dict, int]:
    # TODO latest or best?

    loaded_ckp = torch.load(ckp_path, map_location=device, weights_only=True)
    optimizer_state_dict = loaded_ckp["optimizer_state_dict"]
    model.load_state_dict(loaded_ckp["model_state_dict"])
    resume_epoch = loaded_ckp["epoch"]
    resume_loss = loaded_ckp["loss"]
    print(
        f"Loading model {ckp_path} (stopped at epoch {resume_epoch}) "
        f"with loss {resume_loss:5f}"
    )

    return model, optimizer_state_dict, resume_epoch


def compress_src(path):
    files = glob.glob("**", recursive=True)
    # Read all directory, subdirectories and list files
    zf = zipfile.ZipFile(
        os.path.join(path, "src.zip"),
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )
    for name in files:
        if name.endswith(".py") or name.endswith(".yaml"):
            zf.write(name, arcname=name)
    zf.close()


def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
