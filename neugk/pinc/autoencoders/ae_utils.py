from typing import Dict, Optional, Tuple, Callable

import os
import pickle

import torch
from torch import nn
import torch.distributed as dist
from omegaconf import DictConfig

from neugk.dataset.augment import reverse_ifft, de_normalize
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
    denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model_key = "autoencoder" if hasattr(cfg, "autoencoder") else "model"
    extra_zf_loss = (
        cfg.dataset.separate_zf
        if hasattr(getattr(cfg, model_key), "extra_zf_loss")
        and getattr(cfg, model_key).extra_zf_loss
        else False
    )
    separated_zf = cfg.dataset.separate_zf
    model.train()

    if cfg.dataset.augment.mask_modes.active:
        df_tgt = xs.pop("df_tgt")
        mask = xs.pop("mask")

    # model prediction
    # for ae we only use df
    return_latents = False
    for key in loss_wrap.active_losses:
        if key in ["vicreg_variance", "vicreg_covariance", "logdet"]:
            return_latents = True
    x_preds = model(xs["df"], condition=condition, return_latent=return_latents)

    if cfg.dataset.augment.mask_modes.active:
        assert (
            denormalize_fn is not None
        ), "denormalize_fn must be provided for masked spectral loss"
        pred_df_delta, gt_df_delta = masked_spectral_loss(
            x_preds["df"],
            df_tgt,
            mask,
            separated_zf,
            de_normalize_fn=denormalize_fn,
            file_idx=idx_data["file_index"],
        )
        x_preds["df_delta"] = pred_df_delta
        xs["df_delta"] = gt_df_delta
        # re-assign target to input for loss
        xs["df"] = df_tgt.detach()

    loss, losses = loss_wrap(
        x_preds,
        xs,  # autoencoder
        idx_data,
        geometry=geometry,
        progress_remaining=progress_remaining,
        separate_zf=extra_zf_loss,
    )
    return loss, losses


def masked_spectral_loss(y_hat, y, mask, zf_separated, de_normalize_fn, file_idx):
    """
    y_hat : predicted field (real space)
    y     : ground truth field (real space)
    mask  : binary mask in Fourier space (1=visible, 0=masked)
    zf_separated : applied channel-wise zonal flow separation
    """

    # de-normalize fields first
    y_hat = de_normalize(y_hat, file_idx, de_normalize_fn)
    y = de_normalize(y, file_idx, de_normalize_fn)
    # FFT to spectral space
    y_hat_k = reverse_ifft(y_hat.float(), zf_separated=zf_separated)
    y_k = reverse_ifft(y.float(), zf_separated=zf_separated)

    # Isolate masked modes
    masked_pred = (1.0 - mask) * y_hat_k
    masked_gt = (1.0 - mask) * y_k

    return masked_pred, masked_gt


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
    try:
        loaded_ckpt = torch.load(ckp_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        loaded_ckpt = torch.load(ckp_path, map_location=device, weights_only=False)
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

        problem_dim = len(config.dataset.active_keys)
        res = getattr(config.dataset, "resolution", (32, 8, 16, 85, 32))

        class DummyDataset:
            def __init__(self, problem_dim, resolution):
                self.active_keys = list(range(problem_dim))
                self.resolution = resolution

        model = get_autoencoder(config, DummyDataset(problem_dim, res), rank=None)

    # Check if the checkpoint has 'module.' prefix and if the model expects it
    checkpoint_has_module = any(k.startswith("module.") for k in state_dict.keys())
    model_is_ddp = hasattr(model, "module")

    if checkpoint_has_module and not model_is_ddp:
        # Checkpoint has module prefix but model doesn't - remove prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
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
            print("Found PEFT checkpoint. Filtering out parameters to load base model.")
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
                "Filtered state dict: "
                f"{len(state_dict)} parameters (removed PEFT parameters)"
            )

    # Remap old conditioning keys to new ones if necessary
    remapped_state_dict = {}
    did_remap = False

    # Pre-detect if we are loading into an architecture that might have changed (e.g. missing modulation)
    # If the state dict has dit keys but the model doesn't, we'll need strict=False
    has_dit_in_ckpt = any(".dit." in k for k in state_dict.keys())
    has_modulation_in_ckpt = any(".modulation." in k for k in state_dict.keys())

    def get_base_model(m):
        return m.module if hasattr(m, "module") else m

    base_model = get_base_model(model)
    has_dit_in_model = any(".dit." in n for n, _ in model.named_parameters())

    if (has_dit_in_ckpt or has_modulation_in_ckpt) and not has_dit_in_model:
        print("Architecture mismatch (modulation). Enabling relaxed loading.")
        did_remap = True

    for k, v in state_dict.items():
        new_k = k.replace("encoder_cond_embed", "enc_cond_embed").replace(
            "decoder_cond_embed", "dec_cond_embed"
        )
        if new_k != k:
            did_remap = True

        # Handle the very old 'cond_embed' name (pre-split)
        if new_k.startswith("cond_embed."):
            suffix = new_k[len("cond_embed.") :]

            enc_attr = getattr(base_model, "enc_cond_embed", None)
            dec_attr = getattr(base_model, "dec_cond_embed", None)

            mapped = False
            # Check if shapes match before mapping to avoid RuntimeError
            if enc_attr is not None:
                # Get the parameter shape from the actual module to be sure
                try:
                    target_param = (
                        enc_attr.get_parameter(suffix)
                        if hasattr(enc_attr, "get_parameter")
                        else None
                    )
                    if target_param is None:
                        # Fallback to dict lookup
                        target_param = dict(enc_attr.named_parameters()).get(suffix)

                    if target_param is not None and target_param.shape == v.shape:
                        remapped_state_dict[f"enc_cond_embed.{suffix}"] = v
                        did_remap = True
                        mapped = True
                except Exception:
                    pass

            if dec_attr is not None:
                try:
                    target_param = (
                        dec_attr.get_parameter(suffix)
                        if hasattr(dec_attr, "get_parameter")
                        else None
                    )
                    if target_param is None:
                        target_param = dict(dec_attr.named_parameters()).get(suffix)

                    if target_param is not None and target_param.shape == v.shape:
                        remapped_state_dict[f"dec_cond_embed.{suffix}"] = v
                        did_remap = True
                        mapped = True
                except Exception:
                    pass

            # If we didn't map to either, keep the original key if strict=False might save us
            if not mapped:
                remapped_state_dict[new_k] = v
        else:
            remapped_state_dict[new_k] = v

    state_dict = remapped_state_dict

    # Check if we have an eflux_head in the model but not in the state_dict
    has_eflux_head_in_model = getattr(base_model, "eflux_head", None)
    has_eflux_head_in_ckpt = any("eflux_head" in k for k in state_dict.keys())

    # Set strict=False if we did remapping or if we are missing the eflux head
    strict = not did_remap
    if has_eflux_head_in_model and not has_eflux_head_in_ckpt:
        print(
            "Model has eflux_head but checkpoint does not. Loading with strict=False."
        )
        strict = False

    # Force strict=False if loading a base model from a PEFT checkpoint (already handled by filtering usually)
    if is_peft_checkpoint and not load_peft:
        strict = False

    model.load_state_dict(state_dict, strict=strict)

    resume_epoch = loaded_ckpt["epoch"]
    print(f"Loading model {ckp_path} (stopped at epoch {resume_epoch}) ")
    if config is None:
        return model, loaded_ckpt
    else:
        return model, loaded_ckpt, config


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
