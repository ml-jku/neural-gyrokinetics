from typing import Dict, Optional, Tuple, List

import os
import pickle
import os.path as osp
import sys
import hydra
import warnings
from omegaconf import OmegaConf

import h5py
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from omegaconf import DictConfig

from neugk.utils import RunningMeanStd, filter_config_subset, filter_cli_priority
from neugk.pinc.peft_utils import create_lora_model_wrapper
from neugk.pinc.autoencoders import get_autoencoder


def train_step_autoencoder(
    cfg: DictConfig,
    model: nn.Module,
    xs: Dict[str, torch.Tensor],
    condition: Dict[str, torch.Tensor],
    idx_data: Dict[str, torch.Tensor],
    geometry: Dict[str, torch.Tensor],
    loss_wrap: nn.Module,
    progress_remaining: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model_key = "autoencoder" if hasattr(cfg, "autoencoder") else "model"
    separate_zf = (
        cfg.dataset.separate_zf
        if hasattr(getattr(cfg, model_key), "extra_zf_loss")
        and getattr(cfg, model_key).extra_zf_loss
        else False
    )
    model.train()
    # model prediction
    # for ae we only use df
    x_preds = model(xs["df"], condition=condition)

    # compute losses
    # TODO(diff) get rid of loss_wrap?
    # loss = F.mse_loss(x_preds["df"], xs["df"])
    # losses = {"df": loss}

    return loss_wrap(
        x_preds,
        xs,  # autoencoder
        idx_data,
        geometry=geometry,
        progress_remaining=progress_remaining,
        separate_zf=separate_zf,
    )


def train_step_peft(
    cfg: DictConfig,
    model: nn.Module,
    xs: Dict[str, torch.Tensor],
    condition: Dict[str, torch.Tensor],
    idx_data: Dict[str, torch.Tensor],
    geometry: Dict[str, torch.Tensor],
    loss_wrap: nn.Module,
    progress_remaining: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model_key = "autoencoder" if hasattr(cfg, "autoencoder") else "model"
    separate_zf = (
        cfg.dataset.separate_zf
        if hasattr(getattr(cfg, model_key), "extra_zf_loss")
        and getattr(cfg, model_key).extra_zf_loss
        else False
    )
    model.train()
    x_preds = model(xs["df"], condition=condition)

    return loss_wrap(
        x_preds,
        xs,
        idx_data,
        geometry=geometry,
        progress_remaining=progress_remaining,
        separate_zf=separate_zf,
    )


def train_step_simsiam(
    cfg: DictConfig,
    model: nn.Module,
    xs: Dict[str, torch.Tensor],
    condition: Dict[str, torch.Tensor],
    idx_data: Dict[str, torch.Tensor],
    geometry: Dict[str, torch.Tensor],
    loss_wrap: nn.Module,
    progress_remaining: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.train()
    # stack along batch (same trajectory)
    df_, cond_ = torch.cat([xs["df"], xs["df_aug"]]), torch.cat([condition, condition])
    preds = model(df_, condition=cond_, decoder=True)
    xs["df"] = df_  # update target with stacked version (2x batch)
    return loss_wrap(
        preds,
        xs,
        idx_data,
        geometry=geometry,
        progress_remaining=progress_remaining,
        separate_zf=getattr(cfg.dataset, "separate_zf", False),
    )


def load_autoencoder(
    ckp_path: str,
    device: torch.DeviceObjType,
    model: Optional[nn.Module] = None,
    load_peft: bool = False,
) -> Tuple[nn.Module, Dict, int]:
    # TODO latest or best?
    if os.path.isdir(ckp_path):
        ckp_path = os.path.join(ckp_path, "best.pth")
    loaded_ckpt = torch.load(ckp_path, map_location=device, weights_only=True)
    state_dict = loaded_ckpt["model_state_dict"]

    config = None
    if model is None:
        # TODO move to its own function that loads everything
        import yaml
        from types import SimpleNamespace

        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(
                    **{k: dict_to_namespace(v) for k, v in d.items()}
                )
            elif isinstance(d, list):
                return [dict_to_namespace(v) for v in d]
            else:
                return d

        cfg_path = "/".join(ckp_path.split("/")[:-1]) + "/config.yaml"
        with open(cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        config = dict_to_namespace(cfg_dict)

        problem_dim = 2

        class DummyDataset:
            def __init__(self, problem_dim):
                self.active_keys = [0] * problem_dim
                self.resolution = (32, 8, 16, 85, 32)
        
        model = get_autoencoder(config, DummyDataset(problem_dim), rank=None)

    # Check if the checkpoint has 'module.' prefix and if the model expects it
    checkpoint_has_module = any(k.startswith("module.") for k in state_dict.keys())
    model_is_ddp = hasattr(model, "module")

    if checkpoint_has_module and not model_is_ddp:
        # Checkpoint has module prefix but model doesn't - remove prefix
        state_dict = {
            k[7:]: v for k, v in state_dict.items() if k.startswith("module.")
        }
    elif not checkpoint_has_module and model_is_ddp:
        # Checkpoint doesn't have module prefix but model does - add prefix
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    # Check if it is a PEFT checkpoint
    is_peft_checkpoint = loaded_ckpt.get("stage") == "peft"
    has_peft_params = any("lora_A" in k or "lora_B" in k for k in state_dict.keys())

    if is_peft_checkpoint and has_peft_params:
        if load_peft:
            # load the config to get PEFT configuration
            checkpoint_dir = "/".join(ckp_path.split("/")[:-1])
            config_path = f"{checkpoint_dir}/config.yaml"

            if os.path.exists(config_path):
                from omegaconf import OmegaConf

                full_config = OmegaConf.load(config_path)

                model_key = "autoencoder" if hasattr(config, "autoencoder") else "model"

                if hasattr(getattr(full_config, model_key), "peft"):
                    peft_config = getattr(full_config, model_key).peft

                    # reconstruct PEFT model
                    if peft_config.method.lower() == "eva":
                        eva_config = dict(peft_config.eva)
                        peft_model = create_lora_model_wrapper(
                            model, eva_config, method="eva"
                        )
                    else:
                        lora_config = dict(peft_config.lora)
                        peft_model = create_lora_model_wrapper(
                            model, lora_config, method="lora"
                        )

                    model = peft_model
                    print(f"Reconstructed PEFT model with method: {peft_config.method}")
                else:
                    print("Warning: Could not find PEFT config in checkpoint config")
            else:
                print(f"Warning: Could not find config file at {config_path}")
        else:
            # Loading PEFT checkpoint into base model - filter out PEFT parameters
            print(
                "Found PEFT checkpoint. Filtering out PEFT parameters to load the base model."
            )
            base_state_dict = {}
            for k, v in state_dict.items():
                # Skip PEFT-specific parameters
                if not any(
                    peft_key in k
                    for peft_key in ["lora_A", "lora_B", "lora_embedding", "eva_"]
                ):
                    base_state_dict[k] = v
            state_dict = base_state_dict
            print(
                f"Filtered state dict: {len(state_dict)} parameters (removed PEFT parameters)"
            )

    model.load_state_dict(
        state_dict,
        strict=not (is_peft_checkpoint and has_peft_params and not load_peft),
    )
    resume_epoch = loaded_ckpt["epoch"]
    print(f"Loading model {ckp_path} (stopped at epoch {resume_epoch}) ")
    if config is None:
        return model, loaded_ckpt
    else:
        return model, loaded_ckpt, config


def restart_config_autoencoder():
    config = hydra.compose("main", overrides=sys.argv[1:])

    if config.get("ae_checkpoint") is not None:
        checkpoint_path = osp.abspath(config.ae_checkpoint)
        config_path = osp.join(checkpoint_path, "config.yaml")

        # Determine correct weights file
        if getattr(config.training, "use_latest_checkpoint", False):
            checkpoint_path = osp.join(checkpoint_path, "ckp.pth")
        else:
            checkpoint_path = osp.join(checkpoint_path, "best.pth")

        if os.path.isfile(checkpoint_path) and os.path.isfile(config_path):
            if config.stage == "peft":
                print(f"PEFT stage: Will load model weights from {checkpoint_path}")
            else:
                checkpoint_config = OmegaConf.load(config_path)

                # Remove CLI args related to autoencoder to avoid conflicts
                try:
                    aecli_idx = ["autoencoder" in c for c in sys.argv].index(True)
                    aecli = sys.argv.pop(aecli_idx)
                    warnings.warn(
                        f"CLI arg '{aecli}' ignored in favor of checkpoint config."
                    )
                except ValueError:
                    pass

                # Copy dataset settings
                config.dataset.spatial_ifft = checkpoint_config.dataset.spatial_ifft
                config.dataset.separate_zf = checkpoint_config.dataset.separate_zf
                config.dataset.real_potens = checkpoint_config.dataset.real_potens

                # Clean config to allow merge
                if "autoencoder" in checkpoint_config:
                    for key in ["loss_scheduler", "loss_weights", "extra_loss_weights"]:
                        if key in checkpoint_config.model:
                            del checkpoint_config.model[key]

                filter_cli_priority(sys.argv[1:], checkpoint_config)
                filter_config_subset(config, checkpoint_config)
                config = OmegaConf.merge(config, checkpoint_config)
                config.ae_checkpoint = checkpoint_path
                print(f"Loaded config from checkpoint '{config_path}'")
        else:
            raise ValueError(f"{checkpoint_path} does not exist!")
    return config


def aggregate_dataset_stats(file_paths: List[str]) -> Dict[str, float]:
    """
    Aggregate statistics across multiple dataset files to get true dataset-wide statistics.
    This is the correct way to handle statistics for multi-file datasets.
    """

    # Initialize running statistics
    phi_stats = RunningMeanStd((1,))
    flux_stats = RunningMeanStd((1,))

    total_samples = 0

    for file_path in file_paths:
        phi_mean, phi_std = None, None
        flux_mean, flux_std = None, None
        n_samples = 0

        # standard h5
        try:
            with h5py.File(file_path, "r") as f:
                if "metadata" not in f:
                    continue

                metadata = f["metadata"]

                if "data" in f:
                    n_samples = len(
                        [k for k in f["data"].keys() if k.startswith("timestep_")]
                    )
                else:
                    n_samples = len(metadata["timesteps"][()])

                if "phi_mean" in metadata and "phi_std" in metadata:
                    phi_mean = metadata["phi_mean"][()]
                    phi_std = metadata["phi_std"][()]

                if "flux_mean" in metadata and "flux_std" in metadata:
                    flux_mean = metadata["flux_mean"][()]
                    flux_std = metadata["flux_std"][()]

        except Exception as h5_err:
            # kvikio pkl fallback
            try:
                meta_path = os.path.join(file_path, "metadata.pkl")
                with open(meta_path, "rb") as mf:
                    metadata = pickle.load(mf)

                data_dir = os.path.join(file_path, "data")
                if os.path.exists(data_dir):
                    n_samples = len(
                        [
                            k
                            for k in os.listdir(data_dir)
                            if k.startswith("timestep_") and k.endswith(".bin")
                        ]
                    )
                else:
                    n_samples = len(metadata["timesteps"])

                if "phi_mean" in metadata and "phi_std" in metadata:
                    phi_mean = metadata["phi_mean"]
                    phi_std = metadata["phi_std"]

                if "flux_mean" in metadata and "flux_std" in metadata:
                    flux_mean = metadata["flux_mean"]
                    flux_std = metadata["flux_std"]

            except Exception as pkl_err:
                print(
                    f"Warning: Could not process {file_path}.\n"
                    f"H5 Error: {h5_err}\n  -> Pickle Error: {pkl_err}"
                )
                continue

        if n_samples == 0:
            continue

        total_samples += n_samples

        if phi_mean is not None and phi_std is not None:
            phi_var = phi_std**2

            # phi has shape (2, 1, 1, 1) for [real, imaginary] channels
            # For integral loss normalization, use the magnitude (combined statistics)
            if (
                hasattr(phi_mean, "shape")
                and len(phi_mean.shape) > 0
                and phi_mean.shape[0] == 2
            ):
                # Compute magnitude statistics: sqrt(real^2 + imag^2)
                # For mean: use RMS of both channels
                phi_mean_combined = np.sqrt(np.mean(phi_mean**2))
                # For variance: combine variances assuming independence
                phi_var_combined = np.mean(phi_var)  # average variance across channels
            else:
                phi_mean_combined = (
                    float(phi_mean) if np.isscalar(phi_mean) else float(phi_mean.item())
                )
                phi_var_combined = (
                    float(phi_var) if np.isscalar(phi_var) else float(phi_var.item())
                )

            # Update running statistics (weighted by number of samples)
            phi_stats.update_from_moments(
                batch_mean=np.array([phi_mean_combined]),
                batch_var=np.array([phi_var_combined]),
                batch_min=np.array([phi_mean_combined]),  # Using mean as min/max
                batch_max=np.array([phi_mean_combined]),
                batch_count=float(n_samples),
            )

        if flux_mean is not None and flux_std is not None:
            flux_var = flux_std**2

            flux_mean = (
                float(flux_mean) if np.isscalar(flux_mean) else float(flux_mean.item())
            )
            flux_var = (
                float(flux_var) if np.isscalar(flux_var) else float(flux_var.item())
            )

            flux_stats.update_from_moments(
                batch_mean=np.array([flux_mean]),
                batch_var=np.array([flux_var]),
                batch_min=np.array([flux_mean]),  # Using mean as min/max
                batch_max=np.array([flux_mean]),
                batch_count=float(n_samples),
            )

    # Extract final aggregated statistics
    aggregated_stats = {}
    if phi_stats.count > 0:
        aggregated_stats["phi_mean"] = float(phi_stats.mean.item())
        aggregated_stats["phi_std"] = float(np.sqrt(phi_stats.var).item())

    if flux_stats.count > 0:
        aggregated_stats["flux_mean"] = float(flux_stats.mean.item())
        aggregated_stats["flux_std"] = float(np.sqrt(flux_stats.var).item())

    # print(f"Aggregated statistics from {len(file_paths)} files, {total_samples} total samples")

    return aggregated_stats


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad, state["momentum_buffer"], beta=group["momentum"]
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "momentum", "weight_decay", "use_muon"]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
