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
    src = src.squeeze()
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
    def __init__(self, shape: Sequence[int], epsilon: float = 1e-4):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float32)
        self.min = np.zeros(shape, np.float32)
        self.max = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
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
        self.update_from_moments(
            other.mean, other.var, other.min, other.max, other.count
        )

    def update(self, mean, var, min, max, count=1.0) -> None:
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


def pev_flux_df_phi(
    df: torch.Tensor,
    phi: torch.Tensor,
    geometry,
    aggregate: bool = True,
    magnitude: bool = False,
):
    """
    Computes particle, heat and momentum fluxes based on the distribution function (df)
    and electrostatic potential (phi).

    Args:
        df (torch.Tensor): 5D density function. Shape: (b, c, vpar, vmu, s, x, y).
        phi (torch.Tensor): 3D electrostatic potential. Shape: (b, 1, x, s, y).
        geometry (Dict): Dictionary containing geometry parameters and settings.
        aggregate (bool, optional): Whether to return the summed fluxes. Default: True.
        magnitude (bool, optional): Whether to use df and phi absolutes. Default: False.
    """
    # expand geometry constants for broadcasting
    # grids
    krho = rearrange(geometry["krho"], "y -> 1 1 1 1 y")
    kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 x 1")
    ints = rearrange(geometry["ints"], "s -> 1 1 s 1 1")
    intmu = rearrange(geometry["intmu"], "mu -> 1 mu 1 1 1")
    intvp = rearrange(geometry["intvp"], "par -> par 1 1 1 1")
    vpgr = rearrange(geometry["vpgr"], "par -> par 1 1 1 1")
    mugr = rearrange(geometry["mugr"], "mu -> 1 mu 1 1 1")
    # settings
    little_g = rearrange(geometry["little_g"], "s three -> three 1 1 s 1 1")
    bn = rearrange(geometry["bn"], "s -> 1 1 s 1 1")
    efun = rearrange(geometry["efun"], "s -> 1 1 s 1 1")
    rfun = rearrange(geometry["rfun"], "s -> 1 1 s 1 1")
    bt_frac = rearrange(geometry["bt_frac"], "s -> 1 1 s 1 1")
    parseval = rearrange(geometry["parseval"], "y -> 1 1 1 1 y")
    mas, vthrat, signz = geometry["mas"], geometry["vthrat"], geometry["signz"]
    # gyroaveraged phi
    krloc = torch.sqrt(
        krho**2 * little_g[0] + 2 * krho * kxrh * little_g[1] + kxrh**2 * little_g[2]
    )
    bessel = torch.special.bessel_j0(
        mas * vthrat * krloc * torch.sqrt(2.0 * mugr / bn) / signz
    )

    phi_gyro = bessel * rearrange(phi, "x s y -> 1 1 s x y")
    # absolute values of df and phi
    if magnitude:
        df = -1j * torch.abs(df)
        phi_gyro = torch.abs(phi_gyro)
    # grid derivatives
    dum = parseval * ints * (efun * krho) * df
    dum1 = dum * torch.conj(phi_gyro)
    dum2 = dum1 * bn
    d3X = ints * geometry["d2X"]
    d3v = intmu * bn * intvp
    signB = geometry["signB"]
    # flux fields
    pflux_det = d3X * d3v * torch.imag(dum1)
    eflux_det = d3X * d3v * (vpgr**2 * torch.imag(dum1) + 2 * mugr * torch.imag(dum2))
    vflux_det = d3X * d3v * (torch.imag(dum1) * vpgr * rfun * bt_frac * signB)
    # sum total fluxes
    if aggregate:
        pflux_det = pflux_det.sum()
        eflux_det = eflux_det.sum()
        vflux_det = vflux_det.sum()
    return pflux_det, eflux_det, vflux_det


def phi_integral(df: torch.Tensor, geometry: Dict):
    ns, nx, ny = df.shape[3:]
    # df to fourier
    df = df.movedim(0, -1).contiguous()
    df = torch.view_as_complex(df)
    # phi tensor
    bufphi = torch.zeros((ns, nx, ny), dtype=df.dtype, device=df.device)
    # expand geometry constants for broadcasting
    # grids
    krho = rearrange(geometry["krho"], "y -> 1 1 1 1 y")
    kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 x 1")
    ints = rearrange(geometry["ints"], "s -> 1 1 s 1 1")
    intmu = rearrange(geometry["intmu"], "mu -> 1 mu 1 1 1")
    intvp = rearrange(geometry["intvp"], "par -> par 1 1 1 1")
    mugr = rearrange(geometry["mugr"], "mu -> 1 mu 1 1 1")
    # settings
    little_g = rearrange(geometry["little_g"], "s three -> three 1 1 s 1 1")
    bn = rearrange(geometry["bn"], "s -> 1 1 s 1 1")
    mas, vthrat, signz = geometry["mas"], geometry["vthrat"], geometry["signz"]
    tmp = geometry["tmp"]
    # gyroaveraged phi
    krloc = torch.sqrt(
        krho**2 * little_g[0] + 2 * krho * kxrh * little_g[1] + kxrh**2 * little_g[2]
    )
    bessel = torch.special.bessel_j0(
        mas * vthrat * krloc * torch.sqrt(2.0 * mugr / bn) / signz
    )
    # exponentially scaled bessel i0 function
    gamma = 0.5 * ((mas * vthrat * krloc) / (signz * bn)) ** 2
    gamma = torch.special.i0(gamma) * torch.exp(-gamma)

    # poisson terms
    # density of the species
    de = 1.0
    cfen = torch.zeros_like(ints)
    poisson_int = signz * de * intmu * intvp * bessel * bn
    poisson_int = torch.where(torch.abs(intvp) < 1e-9, 0.0, poisson_int)

    diagz = (
        signz
        * de
        * (signz * (gamma - 1.0) * torch.exp(-cfen) / tmp - torch.exp(-cfen) / tmp)
    )
    matz = -ints / diagz
    matz[..., 1:] = 0.0  # only keep y=0 (turb)

    maty = (-matz * torch.exp(-cfen)).sum((2,), keepdim=True)
    maty = tmp / (de * torch.exp(-cfen)) + maty / torch.exp(-cfen)
    maty[..., 0, :] = 1 + 0j
    maty = torch.where(maty == 0, 1.0, maty)  # avoid infs
    maty = 1 / maty
    maty[..., 1:] = 0.0  # only keep y=0 (turb)

    poisson_diag = torch.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
    poisson_diag[..., 0, 0] = 0.0
    poisson_diag = poisson_diag - signz * torch.exp(-cfen) * de / tmp
    poisson_diag = -1 / poisson_diag

    # first usmv
    phi = (1 + 0j) * poisson_int * df

    #  finish the species sum and the velspace integral
    phi = phi.sum((0, 1), keepdim=True)

    # second usmv
    bufphi = bufphi + (1 + 0j) * matz * phi

    # surface average
    bufphi = bufphi.sum(
        (
            2,
            4,
        ),
        keepdim=True,
    )

    # third usmv
    phi = phi + (1 + 0j) * maty * bufphi

    # normalize
    phi = phi * poisson_diag
    phi = rearrange(phi.squeeze(), "s x y -> x s y")
    return phi


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


def load_geometry(directory):
    geometry = {}

    geometry["parseval"] = torch.tensor([1.0] + [32.0] * (32 - 1), dtype=torch.float32)
    geometry["signz"] = 1.0
    geometry["vthrat"] = 1.0
    geometry["tmp"] = 1.0
    geometry["mas"] = 1.0
    geometry["d2X"] = 1.0
    geometry["signB"] = 1.0

    geom = load_geom(os.path.join(directory, "geom.dat"))  # bn CHECK

    geometry["kxrh"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "kxrh"))[0], dtype=torch.float32
    )  # CHECK
    geometry["krho"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "krho")).T[0] / geom["kthnorm"],
        dtype=torch.float32,
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

    geometry["intmu"] = torch.tensor(intmu[1:], dtype=torch.float32)  # CHECK?
    geometry["mugr"] = torch.tensor(mugr[1:], dtype=torch.float32)  # CHECK?

    geometry["intvp"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "intvp.dat"))[0], dtype=torch.float32
    )  # CHECK
    geometry["vpgr"] = torch.tensor(
        np.loadtxt(os.path.join(directory, "vpgr.dat"))[0], dtype=torch.float32
    )

    ints = np.concatenate(
        [np.array([0.0]), np.diff(np.loadtxt(os.path.join(directory, "sgrid")))]
    )
    ints[0] = ints[1]  # CHECK
    geometry["ints"] = torch.tensor(ints, dtype=torch.float32)

    geometry["efun"] = torch.tensor(-geom["E_eps_zeta"], dtype=torch.float32)  # CHECK

    geometry["little_g"] = torch.tensor(
        np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1),
        dtype=torch.float32,
    )

    geometry["bn"] = torch.tensor(geom["bn"])
    geometry["bt_frac"] = torch.tensor(geom["Bt_frac"])
    geometry["rfun"] = torch.tensor(geom["R"])
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
