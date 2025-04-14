from typing import Dict, Tuple, Optional
import glob
import zipfile
import os
import random
import importlib
import socket
import re

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
import wandb  # noqa


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

    def log(self, logs, commit: bool = True, step: Optional[int] = None):
        wandb.log(logs, step=step, commit=commit)

    def close(self):
        pass

    def summarize(self, outputs):
        # add values to the wandb summary => only works for scalars
        for k, v in outputs.items():
            self._wandb.run.summary[k] = v.item()

    def finish(self):
        # End the W&B run
        wandb.finish()


def edit_tag(dict, prefix, postfix):
    return {f"{prefix}/{k}_{postfix}": v for k, v in dict.items()}


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
) -> float:
    # create directory if it s not there
    os.makedirs(cfg.ckpt_path, exist_ok=True)

    with open(os.path.join(cfg.ckpt_path, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

    state_dict = model.state_dict()

    if hasattr(model, "module"):
        # using DDP wrapper
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
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
                "model_state_dict": state_dict,
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
    temp_key = list(loaded_ckp["model_state_dict"].keys())[0]
    if temp_key.startswith("module."):
        loaded_ckp["model_state_dict"] = {
            k.replace("module.", ""): v
            for k, v in loaded_ckp["model_state_dict"].items()
        }
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
        if (
            name.endswith(".py")
            or name.endswith(".yaml")
            or name.endswith(".ipynb")
            and "wandb" not in name
            and "outputs" not in name
        ):
            zf.write(name, arcname=name)
    zf.close()


def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def expand_as(src: np.ndarray, tgt: np.ndarray):
    while src.ndim < tgt.ndim:
        src = np.expand_dims(src, axis=-1)
    return src


def split_in_two(dictionary, idx):
    first = {k: v[:idx] for k, v in dictionary.items()}
    second = {k: v[idx:] for k, v in dictionary.items()}
    dictionary = [first, second]
    return dictionary


def split_batch_into_phases(phase_change, inputs, gts, conds, idx_data):
    split_idx = torch.searchsorted(conds["timestep"], phase_change, right=False)
    if split_idx == conds["timestep"].shape[0]:
        # whole batch in linear
        inputs = [inputs]
        gts = [gts]
        conds = [conds]
        idx_data = [idx_data]
        phase_list = ["linear"]
    elif split_idx == 0:
        # whole batch in saturated phase
        inputs = [inputs]
        gts = [gts]
        conds = [conds]
        idx_data = [idx_data]
        phase_list = ["saturated"]
    else:
        inputs = split_in_two(inputs, split_idx)
        gts = split_in_two(gts, split_idx)
        conds = split_in_two(conds, split_idx)
        idx_data = split_in_two(idx_data, split_idx)
        phase_list = ["linear", "saturated"]
    return (
        inputs,
        gts,
        conds,
        idx_data,
        phase_list,
    )


def is_number(string):
    pattern = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
    return bool(re.fullmatch(pattern, string.strip()))


def load_geom_dat(file_path):
    data = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    key = None
    values = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 1 and not is_number(parts[0]):
            try:
                if len(values) == 0:
                    values.extend(map(float, parts))
                    data[key] = values[0]
                    key = None
                    values = []
                    continue
                else:
                    raise ValueError
            except:
                if key is not None:
                    data[key] = np.array(values, dtype=np.float64)
                key = parts[0]
                values = []
        else:
            values.extend(map(float, parts))

    if key is not None:
        data[key] = np.array(values, dtype=np.float64)

    return data


def get_geometry(path: str = "/restricteddata/ukaea/gyrokinetics/raw/cyclone4_2_2"):
    geometry = {}

    geometry["parseval"] = torch.tensor([1.0] + [32.0] * (32 - 1), dtype=torch.float32)
    geometry["signz"] = 1.0
    geometry["vthrat"] = 1.0
    geometry["tmp"] = 1.0
    geometry["mas"] = 1.0
    geometry["d2X"] = 1.0
    geometry["signB"] = 1.0

    geom = load_geom_dat(os.path.join(path, "geom.dat"))

    geometry["kxrh"] = torch.tensor(
        np.loadtxt(os.path.join(path, "kxrh"))[0], dtype=torch.float32
    )
    geometry["krho"] = torch.tensor(
        np.loadtxt(os.path.join(path, "krho")).T[0] / geom["kthnorm"],
        dtype=torch.float32,
    )

    # mugr and intmu
    mugr = np.zeros(8 + 1)
    intmu = np.zeros(8 + 1)
    mumax = 4.5
    dvperp = np.sqrt(2.0 * mumax) / 8
    for j in range(8 + 1):
        vperp = (j - 0.5) * dvperp
        mugr[j] = vperp**2 / 2.0
        intmu[j] = abs(
            np.pi * ((vperp + 0.5 * dvperp) ** 2 - (vperp - 0.5 * dvperp) ** 2)
        )

    geometry["intmu"] = torch.tensor(intmu[1:], dtype=torch.float32)
    geometry["mugr"] = torch.tensor(mugr[1:], dtype=torch.float32)

    geometry["intvp"] = torch.tensor(
        np.loadtxt(os.path.join(path, "intvp.dat"))[0], dtype=torch.float32
    )
    geometry["vpgr"] = torch.tensor(
        np.loadtxt(os.path.join(path, "vpgr.dat"))[0], dtype=torch.float32
    )

    ints = np.concatenate(
        [np.array([0.0]), np.diff(np.loadtxt(os.path.join(path, "sgrid")))]
    )
    ints[0] = ints[1]
    geometry["ints"] = torch.tensor(ints, dtype=torch.float32)

    geometry["efun"] = torch.tensor(-geom["E_eps_zeta"], dtype=torch.float32)

    geometry["little_g"] = torch.tensor(
        np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1),
        dtype=torch.float32,
    )

    geometry["bn"] = torch.tensor(geom["bn"])
    geometry["bt_frac"] = torch.tensor(geom["Bt_frac"])
    geometry["rfun"] = torch.tensor(geom["R"])

    return geometry
