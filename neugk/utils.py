from typing import Dict, Tuple, Optional, Sequence, Union
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


def recombine_zf(x, dim: int = 1):
    """
    Recombine Zonal Flow (ZF) and non-ZF components by summing.
    Layout: [zf, x - zf] -> [x]
    """
    if x.shape[dim] <= 2 or x.shape[dim] % 2 != 0:
        return x

    half_c = x.shape[dim] // 2
    if torch.is_tensor(x):
        zf, non_zf = x.narrow(dim, 0, half_c), x.narrow(dim, half_c, half_c)
    else:
        zf, non_zf = np.split(x, 2, axis=dim)

    return zf + non_zf


def separate_zf(x, dim: int = 1):
    """
    Separate Zonal Flow (ZF) and non-ZF components.
    Returns [zf, x - zf] layout.
    """
    if torch.is_tensor(x):
        nky = x.shape[-1]
        zf = torch.repeat_interleave(x.mean(dim=-1, keepdim=True), repeats=nky, dim=-1)
        return torch.cat([zf, x - zf], dim=dim)
    else:
        nky = x.shape[-1]
        zf = np.repeat(x.mean(axis=-1, keepdims=True), repeats=nky, axis=-1)
        return np.concatenate([zf, x - zf], axis=dim)


def wandb_available():
    """Check if wandb is available and not explicitly disabled."""
    # check environment
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
    """Wrapper for weights and biases logging initialization and updates."""

    def __init__(self) -> None:
        self._initialized = False

    def setup(self, args, **kwargs):
        """Initialize the wandb run."""
        if not isinstance(args, dict):
            args = args.__dict__
        project_name = args["logging"].get("project", "debug")

        # build run name
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

        # initialize
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
        """Log metrics to the current wandb run."""
        wandb.log(logs, step=step, commit=commit)

    def close(self):
        """No-op for compatibility."""
        pass

    def summarize(self, outputs):
        """Add summary values to the wandb run."""
        # add values to the wandb summary => only works for scalars
        for k, v in outputs.items():
            self._wandb.run.summary[k] = v.item()

    def finish(self):
        """End the current wandb run."""
        # End the W&B run
        wandb.finish()


def ddp_setup(rank, world_size):
    """Initialize distributed data parallel environment."""
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )


def edit_tag(d, prefix=None, postfix=None):
    """Update dictionary keys with prefix and postfix tags, avoiding duplicates."""
    res = {}
    for k, v in d.items():
        nk = k
        if prefix and not k.startswith(f"{prefix}/"):
            nk = f"{prefix}/{nk}"
        if postfix and not k.endswith(f"_{postfix}"):
            nk = f"{nk}_{postfix}"
        res[nk] = v
    return res


def setup_logging(config):
    """Configure and return the selected logging writer."""
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
    """Save model checkpoint and its configuration to disk."""
    # create directory if it s not there
    os.makedirs(cfg.output_path, exist_ok=True)

    with open(os.path.join(cfg.output_path, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f.name)

    state_dict = model.state_dict()

    if hasattr(model, "module"):
        # using DDP wrapper
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # persist checkpoint
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

    # persist best
    best_path = f"{cfg.output_path}/best.pth"
    if val_loss < loss_val_min or not os.path.exists(best_path):
        loss_val_min = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_loss,
            },
            best_path,
        )

    return loss_val_min


def load_model_and_config(
    ckp_path: str,
    model: nn.Module,
    device: torch.DeviceObjType,
    for_ddp=False,
) -> Tuple[nn.Module, Dict, int]:
    """Load model state and checkpoint info from disk."""
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
    """Zip source code files into a specified directory."""
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
            (name.endswith(".py") or name.endswith(".yaml") or name.endswith(".ipynb"))
            and "wandb" not in name
            and "outputs" not in name
            and os.path.isfile(name)
        ):
            try:
                zf.write(name, arcname=name)
            except FileNotFoundError:
                pass
    zf.close()


def set_seed(seed):
    """Set global random seeds for reproducibility."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def find_free_port():
    """Find and return a free network port."""
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def expand_as(
    src: Union[np.ndarray, torch.Tensor], tgt: Union[np.ndarray, torch.Tensor]
):
    """Expand dimensions of src to match tgt, handling batch, time, and spatial dims."""
    if src.ndim == tgt.ndim:
        return src

    # determine where src fits into tgt
    src_shape = list(src.shape)
    tgt_shape = list(tgt.shape)

    # covers casex where src is (C, ...) and tgt is (B, T, C, ...)
    start_axis = -1
    for i in range(len(tgt_shape) - len(src_shape) + 1):
        if tgt_shape[i] == src_shape[0]:
            match = True
            for j in range(1, len(src_shape)):
                if src_shape[j] != 1 and src_shape[j] != tgt_shape[i + j]:
                    match = False
                    break
            if match:
                start_axis = i
                break

    if start_axis == -1:
        res = src
        while res.ndim < tgt.ndim:
            if isinstance(res, np.ndarray):
                res = np.expand_dims(res, axis=-1)
            else:
                res = res.unsqueeze(-1)
        return res

    # reshape src to have for broadcast
    new_shape = [1] * len(tgt_shape)
    for i, s in enumerate(src_shape):
        new_shape[start_axis + i] = s

    if isinstance(src, np.ndarray):
        return src.reshape(new_shape)
    else:
        return src.view(new_shape)


def is_number(string):
    """Regex check if string represents a number."""
    pattern = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
    return bool(re.fullmatch(pattern, string.strip()))


def filter_config_subset(superset: DictConfig, subset: DictConfig):
    """Recursively remove keys from subset if they are not in superset."""
    for k in list(subset.keys()):
        if k not in superset:
            del subset[k]
        elif OmegaConf.is_dict(superset[k]) and OmegaConf.is_dict(subset[k]):
            filter_config_subset(superset[k], subset[k])


def filter_cli_priority(cli: Sequence, source: DictConfig, key: str = ""):
    """Remove keys from source config if they were overridden via CLI."""
    for k in list(source.keys()):
        subkey = k
        if key is not None and len(key) > 0:
            subkey = f"{key}.{k}"
        if any([subkey in c.split("=")[0] for c in cli]):
            del source[k]
        elif OmegaConf.is_dict(source[k]):
            filter_cli_priority(cli, source[k], key=subkey)


class RunningMeanStd:
    """Calculates online statistics for a data stream."""

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
        assert (
            self.mean is not None
        ), "Cannot copy an uninitialized RunningMeanStd object"
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
        if other.count <= 1e-3:
            return
        if self.count <= 1e-3:
            self.mean = other.mean.copy()
            self.var = other.var.copy()
            self.min = other.min.copy()
            self.max = other.max.copy()
            self.count = other.count
            return

        self.update_from_moments(
            other.mean, other.var, other.min, other.max, other.count
        )

    def update(self, mean, var, min, max, count=1.0) -> None:
        """Update current moments with batch statistics."""
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
        """Parallel variance algorithm implementation."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # compute new mean
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

        # update min and max only after first real update (excluding epsilon)
        if self.count <= 1e-3:
            self.min = batch_min
            self.max = batch_max
        else:
            self.min = np.minimum(self.min, batch_min)
            self.max = np.maximum(self.max, batch_max)

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @staticmethod
    def aggregate_stats(
        means, stds, agg_axes=(1, 2, 3, 4, 5)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduces coordinate-wise stats to channel-wise stats.

        Args:
            coord_means: Array of shape (C, D, H, W)
            coord_stds:  Array of shape (C, D, H, W)
        """
        # The mean of means is the global mean.
        channel_means = np.mean(means, axis=agg_axes, keepdims=True)

        # Convert stds to variance first
        coord_vars = stds**2

        # average of the local variances
        avg_of_vars = np.mean(coord_vars, axis=agg_axes, keepdims=True)

        # variance of the local means
        diff_sq = (means - channel_means) ** 2
        var_of_means = np.mean(diff_sq, axis=agg_axes, keepdims=True)

        # Total Variance = Average of Variances + Variance of Means
        channel_vars = avg_of_vars + var_of_means
        channel_stds = np.sqrt(channel_vars)

        return channel_means, channel_stds


def load_geom_dat_file(file_path):
    """Load geometric parameters from a .dat file."""
    data = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    key = None
    values = []

    # parse lines
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
            except Exception:
                if key is not None:
                    data[key] = np.array(values, dtype=np.float64)
                key = parts[0]
                values = []
        else:
            values.extend(map(float, parts))

    # final commit
    if key is not None:
        data[key] = np.array(values, dtype=np.float64)

    return data


def load_geometry(directory, dtype=torch.float64):
    # load geom dat file and input data
    geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))

    geometry = {}
    geometry["signz"] = torch.tensor(1.0, dtype=dtype)
    geometry["vthrat"] = torch.tensor(1.0, dtype=dtype)
    geometry["tmp"] = torch.tensor(1.0, dtype=dtype)
    geometry["mas"] = torch.tensor(1.0, dtype=dtype)
    geometry["d2X"] = torch.tensor(1.0, dtype=dtype)
    geometry["signB"] = torch.tensor(1.0, dtype=dtype)

    # load physics switches and beta
    control = input_data.get("control", {})

    def parse_gkw_bool(val):
        if isinstance(val, str):
            val = val.lower().strip()
            if val == ".true.":
                return 1.0
            if val == ".false.":
                return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    geometry["nlapar"] = torch.tensor(
        parse_gkw_bool(control.get("nlapar", 0.0)), dtype=dtype
    )
    geometry["nlbpar"] = torch.tensor(
        parse_gkw_bool(control.get("nlbpar", 0.0)), dtype=dtype
    )

    # beta is often in 'parameters' or 'control'
    parameters = input_data.get("parameters", {})
    geometry["beta"] = torch.tensor(float(parameters.get("beta", 0.0)), dtype=dtype)

    # gather active species
    num_sp = 1
    for sec in input_data.values():
        if "number_of_species" in sec:
            num_sp = int(sec["number_of_species"])
            break
    species_keys = [k for k in input_data.keys() if k.startswith("species")][:num_sp]
    if species_keys:
        mas, tmp, de, signz = [], [], [], []
        for k in species_keys:
            sp = input_data[k]
            mas.append(sp.get("mass", 1.0))
            tmp.append(sp.get("temp", 1.0))
            de.append(sp.get("dens", 1.0))
            signz.append(sp.get("z", 1.0))

        geometry["mas"] = torch.tensor(mas, dtype=dtype)
        geometry["tmp"] = torch.tensor(tmp, dtype=dtype)
        geometry["de"] = torch.tensor(de, dtype=dtype)
        geometry["signz"] = torch.tensor(signz, dtype=dtype)

        # compute vthrat = sqrt(T_s / m_s)
        vthrat = [np.sqrt(t / m) for t, m in zip(tmp, mas)]
        geometry["vthrat"] = torch.tensor(vthrat, dtype=dtype)
        # if multiple species are found, electrons are kinetic, so not adiabatic
        geometry["adiabatic"] = torch.tensor(
            0.0 if len(species_keys) > 1 else 1.0, dtype=dtype
        )
    else:
        geometry["mas"] = torch.tensor([1.0], dtype=dtype)
        geometry["tmp"] = torch.tensor([1.0], dtype=dtype)
        geometry["de"] = torch.tensor([1.0], dtype=dtype)
        geometry["signz"] = torch.tensor([1.0], dtype=dtype)
        geometry["vthrat"] = torch.tensor([1.0], dtype=dtype)
        geometry["adiabatic"] = torch.tensor(1.0, dtype=dtype)

    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))[0]
    krho = np.loadtxt(os.path.join(directory, "krho")).T[0] / geom["kthnorm"]
    geometry["kxrh"] = torch.tensor(kxrh, dtype=dtype)
    geometry["krho"] = torch.tensor(krho, dtype=dtype)
    geometry["parseval"] = torch.tensor(
        [1.0] + [float(len(krho))] * (len(krho) - 1), dtype=dtype
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

    geometry["intmu"] = torch.tensor(intmu[1:], dtype=dtype)  # CHECK?
    geometry["mugr"] = torch.tensor(mugr[1:], dtype=dtype)  # CHECK?

    intvp = np.loadtxt(os.path.join(directory, "intvp.dat"))[0]
    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))[0]
    geometry["intvp"] = torch.tensor(intvp, dtype=dtype)
    geometry["vpgr"] = torch.tensor(vpgr, dtype=dtype)

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    ints = np.concatenate([np.array([0.0]), np.diff(sgrid)])
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

    # if multiple species are present, adiabatic should be 0.0
    if len(geometry.get("de", [1.0])) > 1:
        geometry["adiabatic"] = torch.tensor(0.0, dtype=dtype)
    else:
        geometry["adiabatic"] = torch.tensor(
            np.squeeze(geom.get("adiabatic", 1.0)), dtype=dtype
        )

    # ensure species-specific fields are updated in the returned geometry
    for k in ["mas", "tmp", "de", "signz", "vthrat"]:
        if k in geom:
            geometry[k] = torch.tensor(geom[k], dtype=dtype)
        elif k not in geometry:
            geometry[k] = torch.tensor(1.0, dtype=dtype)

    return geometry


def get_linear_burn_in_fn(
    start: float, end: float, end_fraction: float, start_fraction: float
):
    """Return a linear scheduler function for progress-based weighting."""

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
    """Compute remaining progress fraction."""
    return 1.0 - (cur_step / total_steps)


def parse_input_dat(file_path):
    """Parse GKW input.dat configuration file."""
    parsed_data = {}
    with open(file_path, "r") as file:
        content = file.read()
    # split sections
    sections = re.split(r"&\w+", content)
    section_headers = re.findall(r"&(\w+)", content)
    # clean comments
    sections = [
        section.strip()
        for section in sections
        if len(section) and section[0] != "!" and section.strip()
    ]
    # iterate over sections
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
    """List distribution function files in a directory."""
    files = os.listdir(directory)
    digit_files = sorted(
        [file for file in files if file.isdigit()], key=lambda x: int(x)
    )
    k_files = sorted(
        [file for file in files if file.startswith("K") and not file.endswith(".dat")]
    )
    return k_files + digit_files


def poten_files(directory):
    """List potential field files in a directory."""
    files = os.listdir(directory)
    poten_files = sorted([file for file in files if file.startswith("Poten")])
    timestep_slices = [int(f.replace("Poten", "")) for f in poten_files]
    return poten_files, np.array(timestep_slices) - 1


def exclude_from_weight_decay(model, param_names, weight_decay):
    """Split model parameters into groups with and without weight decay."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        add_to_no_decay = False
        for param_name in param_names:
            if param_name in name.lower():
                add_to_no_decay = True

        if add_to_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def memory_cleanup(device=None, aggressive=False):
    """Perform garbage collection and clear GPU cache."""
    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()

    # clear cache
    if torch.cuda.is_available() and device is not None:
        torch.cuda.empty_cache()

    # aggressive cleanup
    if aggressive:
        gc.collect()
        if torch.cuda.is_available() and device is not None:
            # force sync
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
