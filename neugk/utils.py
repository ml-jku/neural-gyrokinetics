"""General utils."""

from typing import Dict, Tuple, Optional, Sequence
from datetime import timedelta
import glob
import zipfile
import os
import gc
import random
import importlib
import socket
import re

import torch
import torch.distributed as dist
from torch import nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
from einops import rearrange


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

        name_parts = []
        name_suffix = args["logging"].get("name_suffix", None)
        if (
            name_suffix and name_suffix.strip()
        ):  # Skip if None, empty, or whitespace-only
            name_parts.append(name_suffix)
        exp_id = args["logging"].get("run_id", None)
        if exp_id:
            name_parts.append(exp_id)
        run_name = "_".join(name_parts) if name_parts else None

        tags = args["logging"].get("tags", [])

        combined_dict = {**args, **kwargs}
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=run_name,
            entity=args["logging"].get("entity", None),
            # track hyperparameters and run metadata
            config=combined_dict,
            id=args["logging"].get("run_id", None),
            resume="allow" if args["logging"].get("run_id", None) else False,
            reinit=False,
            tags=tags,
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


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


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
    scheduler: torch.optim.lr_scheduler,
    cfg: DictConfig,
    epoch: int,
    val_loss: float,
    loss_val_min: float,
) -> float:
    # create directory if it s not there
    os.makedirs(cfg.output_path, exist_ok=True)

    with open(os.path.join(cfg.output_path, "config.yaml"), "w") as f:
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
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": val_loss,
        },
        f"{cfg.output_path}/ckp.pth",
    )

    if val_loss < loss_val_min:
        loss_val_min = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_loss,
            },
            f"{cfg.output_path}/best.pth",
        )

    return loss_val_min


def load_model_and_config(
    ckp_path: str,
    model: nn.Module,
    device: torch.DeviceObjType,
    for_ddp=False,
) -> Tuple[nn.Module, Dict, int]:
    loaded_ckpt = torch.load(ckp_path, map_location=device, weights_only=True)
    if for_ddp:
        loaded_ckpt["model_state_dict"] = {
            "module." + k: v for k, v in loaded_ckpt["model_state_dict"].items()
        }
    new_state_dict = loaded_ckpt["model_state_dict"]
    model.load_state_dict(new_state_dict, strict=False)
    resume_epoch = loaded_ckpt["epoch"]
    resume_loss = loaded_ckpt["loss"]
    print(
        f"Loading model {ckp_path} (stopped at epoch {resume_epoch}) "
        f"with loss {resume_loss:5f}"
    )
    return model, loaded_ckpt


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
    if src.ndim == tgt.ndim and all(
        ss == 1 or ss == st for ss, st in zip(src.shape, tgt.shape)
    ):
        return src
    # squeeze is causing issues with arbitrary aggregating of dimensions for stats computation
    # src = src.squeeze()
    while src.ndim < tgt.ndim:
        if isinstance(src, np.ndarray):
            src = np.expand_dims(src, axis=-1)
        elif isinstance(src, torch.Tensor):
            src = src.unsqueeze(-1)
        else:
            raise NotImplementedError("Unsupported datatype")
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


def filter_config_subset(superset: DictConfig, subset: DictConfig):
    for k in list(subset.keys()):
        if k not in superset:
            del subset[k]
        elif OmegaConf.is_dict(superset[k]) and OmegaConf.is_dict(subset[k]):
            filter_config_subset(superset[k], subset[k])


def filter_cli_priority(cli: Sequence, source: DictConfig, key: str = ""):
    for k in list(source.keys()):
        subkey = k
        if key is not None and len(key) > 0:
            subkey = f"{key}.{k}"
        if any([subkey in c.split("=")[0] for c in cli]):
            del source[k]
        elif OmegaConf.is_dict(source[k]):
            filter_cli_priority(cli, source[k], key=subkey)


class RunningMeanStd:
    def __init__(self, shape: Optional[Sequence[int]] = None, epsilon: float = 1e-4):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        if shape is not None:
            self.mean = np.zeros(shape, np.float32)
            self.min = np.full_like(self.mean, fill_value=np.inf, dtype=np.float32)
            self.max = np.zeros(shape, np.float32)
            self.var = np.ones(shape, np.float32)
        else:
            self.mean = self.var = self.min = self.max = None
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        assert self.mean is not None, "Cannot copy an uninitialized RunningMeanStd object"
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        assert self.mean is not None and other.mean is not None, "Cannot combine uninitialized RunningMeanStd objects"
        self.update_from_moments(
            other.mean, other.var, other.min, other.max, other.count
        )

    def update(self, mean, var, min, max, count=1.) -> None:
        if self.mean is None:
            # initialize with shape that we receive
            self.mean = np.zeros_like(mean, np.float32)
            self.min = np.full_like(self.mean, fill_value=np.inf, dtype=np.float32)
            self.max = np.zeros_like(mean, np.float32)
            self.var = np.ones_like(mean, np.float32)
        self.update_from_moments(mean, var, min, max, count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_min: np.ndarray,
        batch_max: np.ndarray,
        batch_count: float = 1.0,
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @staticmethod
    def aggregate_stats(means, vars, mins, maxs, agg_axes=(1,2,3,4,5)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduces coordinate-wise stats to channel-wise stats.
        
        Args:
            means: Array of shape (C, D, H, W)
            vars:  Array of shape (C, D, H, W)
        """
        # The mean of means is the global mean.
        channel_means = np.mean(means, axis=agg_axes, keepdims=True)
        
        # average of the local variances
        avg_of_vars = np.mean(vars, axis=agg_axes, keepdims=True)
        
        # variance of the local means 
        diff_sq = (means - channel_means) ** 2
        var_of_means = np.mean(diff_sq, axis=agg_axes, keepdims=True)
        
        # Total Variance = Average of Variances + Variance of Means
        channel_vars = avg_of_vars + var_of_means
        channel_mins = np.min(mins, axis=agg_axes, keepdims=True)
        channel_maxs = np.max(maxs, axis=agg_axes, keepdims=True)
        
        return channel_means, channel_vars, channel_mins, channel_maxs


def load_geom(file_path):
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


def load_geometry(directory, dtype=torch.float64):
    geometry = {}

    geometry["parseval"] = torch.tensor([1.0] + [32.0] * (32 - 1), dtype=dtype)
    geometry["signz"] = torch.tensor(1.0, dtype=dtype)
    geometry["vthrat"] = torch.tensor(1.0, dtype=dtype)
    geometry["tmp"] = torch.tensor(1.0, dtype=dtype)
    geometry["mas"] = torch.tensor(1.0, dtype=dtype)
    geometry["d2X"] = torch.tensor(1.0, dtype=dtype)
    geometry["signB"] = torch.tensor(1.0, dtype=dtype)

    geom = load_geom(os.path.join(directory, "geom.dat"))  # bn CHECK

    geometry["kxrh"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "kxrh"))[0], dtype=dtype
    )  # CHECK
    geometry["krho"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "krho")).T[0] / geom["kthnorm"],
        dtype=dtype,
    )  # CHECK

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

    geometry["intmu"] = torch.tensor(intmu[1:], dtype=dtype)  # CHECK?
    geometry["mugr"] = torch.tensor(mugr[1:], dtype=dtype)  # CHECK?

    geometry["intvp"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "intvp.dat"))[0], dtype=dtype
    )  # CHECK
    geometry["vpgr"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "vpgr.dat"))[0], dtype=dtype
    )

    ints = np.concatenate(
        [np.array([0.0]), np.diff(np.loadtxt(os.path.join(directory, "sgrid")))]
    )
    ints[0] = ints[1]  # CHECK
    geometry["ints"] = torch.tensor(ints, dtype=dtype)

    geometry["efun"] = torch.tensor(-geom["E_eps_zeta"], dtype=dtype) 
    geometry["little_g"] = torch.tensor(
        np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1),
        dtype=dtype,
    )

    geometry["bn"] = torch.tensor(geom["bn"], dtype=dtype)
    geometry["bt_frac"] = torch.tensor(geom["Bt_frac"], dtype=dtype)
    geometry["rfun"] = torch.tensor(geom["R"], dtype=dtype)
    return geometry


def get_linear_burn_in_fn(
    start: float, end: float, end_fraction: float, start_fraction: float
):

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        elif (1 - progress_remaining) < start_fraction:
            return start
        else:
            return start + (1 - progress_remaining - start_fraction) * (end - start) / (
                end_fraction - start_fraction
            )

    return func


def remainig_progress(cur_step, total_steps):
    return 1.0 - (cur_step / total_steps)


def parse_input_dat(file_path):
    parsed_data = {}
    with open(file_path, "r") as file:
        content = file.read()
    # split the content by section headers (e.g., &SPECIES, &SPCGENERAL, etc.)
    sections = re.split(r"&\w+", content)
    # get all the headers by finding the section names
    section_headers = re.findall(r"&(\w+)", content)
    # remove comments
    sections = [
        section.strip()
        for section in sections
        if len(section) and section[0] != "!" and section.strip()
    ]
    for header, section in zip(section_headers, sections):
        section_dict = {}
        params = re.findall(r"(\w+)\s*=\s*([-\d\.e\w]+)", section)
        for param, value in params:
            try:
                section_dict[param] = (
                    float(value) if "e" in value or "." in value else int(value)
                )
            except ValueError:
                section_dict[param] = value.strip()
        while header in parsed_data:
            header = f"{header}0"
        parsed_data[header] = section_dict

    return parsed_data


def K_files(directory):
    files = os.listdir(directory)
    digit_files = sorted(
        [file for file in files if file.isdigit()], key=lambda x: int(x)
    )
    k_files = sorted(
        [file for file in files if file.startswith("K") and not file.endswith(".dat")]
    )
    return k_files + digit_files


def poten_files(directory):
    files = os.listdir(directory)
    poten_files = sorted([file for file in files if file.startswith("Poten")])
    timestep_slices = [int(f.replace("Poten", "")) for f in poten_files]
    return poten_files, np.array(timestep_slices) - 1


def exclude_from_weight_decay(model, param_names, weight_decay):
    decay, no_decay = [], []
    no_decay_names, decay_names = [], []
    if param_names == "all":
        return [
            {"params": model.parameters(), "weight_decay": 0.0},
        ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        add_to_no_decay = False
        for param_name in param_names:
            if param_name in name.lower():
                add_to_no_decay = True

        if add_to_no_decay:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)
            decay_names.append(name)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def memory_cleanup(device=None, aggressive=False):
    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()

    # Standard cleanup
    if torch.cuda.is_available() and device is not None:
        torch.cuda.empty_cache()

    if aggressive:
        # More aggressive cleanup
        gc.collect()
        if torch.cuda.is_available() and device is not None:
            # Force synchronization to ensure all operations are complete
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
