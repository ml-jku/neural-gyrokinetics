"""
PEFT (LoRA and EVA)
"""

from typing import Dict, List, Optional, Tuple, Union

import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from peft import LoraConfig, EvaConfig, get_peft_model, PeftModel, TaskType


def find_linear_layers(
    model: nn.Module, prefix: str = ""
) -> List[Tuple[str, nn.Linear]]:
    """Recursively find all Linear layers in the model."""
    linear_layers = []

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(module, nn.Linear):
            linear_layers.append((full_name, module))
        else:
            # Recursively search in child modules
            child_layers = find_linear_layers(module, full_name)
            linear_layers.extend(child_layers)

    return linear_layers


def get_target_modules_for_lora(
    model: nn.Module,
    strategy: str = "comprehensive",
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Get target module names for LoRA adaptation based on strategy.

    Args:
        model: The model to analyze
        strategy: "comprehensive", "attention_mlp",...
        exclude_patterns: Patterns to exclude from targeting

    Returns:
        List of module names suitable for LoRA targeting
    """
    if exclude_patterns is None:
        exclude_patterns = []

    linear_layers = find_linear_layers(model)

    # Group layers by type for strategic targeting
    layer_groups = {
        "attention_qkv": [],  # Query, Key, Value projections
        "attention_proj": [],  # Attention output projections
        "mlp_layers": [],  # MLP/Feed-forward layers
        "bottleneck": [],  # Encoder/decoder bottleneck
        "patch_embed": [],  # Patch embedding layers
        "modulation": [],  # DiT modulation layers
        "downsample": [],  # Downsampling/upsampling layers
        "other": [],  # Other linear layers
    }

    for name, _ in linear_layers:
        # Skip excluded patterns
        if any(pattern in name for pattern in exclude_patterns):
            continue

        if any(pattern in name for pattern in ["patch_embed.patch.mlp"]):
            layer_groups["patch_embed"].append(name)
        elif "qkv" in name:
            layer_groups["attention_qkv"].append(name)
        elif any(pattern in name for pattern in ["attn.proj", "attention.proj"]):
            layer_groups["attention_proj"].append(name)
        elif any(
            pattern in name
            for pattern in [
                "mlp.mlp.0",
                "mlp.mlp.2",
                "mlp.mlp.3",
                "mlp.0",
                "mlp.2",
                "mlp.3",
                "cond_embed.mlp",
                "cpb_mlp.mlp",
                "feed_forward",
            ]
        ):
            layer_groups["mlp_layers"].append(name)
        elif any(
            pattern in name
            for pattern in ["middle_downproj", "middle_upproj", "bottleneck"]
        ):
            layer_groups["bottleneck"].append(name)
        elif any(pattern in name for pattern in ["dit.modulation", "modulation"]):
            layer_groups["modulation"].append(name)
        elif any(
            pattern in name
            for pattern in [
                "downsample.reduction",
                "upsample.expansion",
                "downsample",
                "upsample",
            ]
        ):
            layer_groups["downsample"].append(name)
        else:
            layer_groups["other"].append(name)

    # Select modules based on strategy
    target_modules = []

    if strategy == "comprehensive":
        # Target attention, MLP, and bottleneck layers
        target_modules.extend(layer_groups["attention_qkv"])
        target_modules.extend(layer_groups["attention_proj"])
        target_modules.extend(layer_groups["mlp_layers"])
        target_modules.extend(layer_groups["bottleneck"])

    elif strategy == "attention_mlp":
        target_modules.extend(layer_groups["attention_qkv"])
        target_modules.extend(layer_groups["attention_proj"])
        target_modules.extend(layer_groups["mlp_layers"])

    elif strategy == "mlp_only":
        target_modules.extend(layer_groups["mlp_layers"])

    elif strategy == "attention_only":
        target_modules.extend(layer_groups["attention_qkv"])
        target_modules.extend(layer_groups["attention_proj"])

    elif strategy == "bottleneck_only":
        target_modules.extend(layer_groups["bottleneck"])

    elif strategy == "modulation_only":
        target_modules.extend(layer_groups["modulation"])

    elif strategy == "all_except_attention":
        target_modules.extend(layer_groups["mlp_layers"])
        target_modules.extend(layer_groups["bottleneck"])
        target_modules.extend(layer_groups["patch_embed"])
        target_modules.extend(layer_groups["modulation"])
        target_modules.extend(layer_groups["downsample"])

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            "Available strategies: comprehensive, attention_mlp, mlp_only, "
            "attention_only, bottleneck_only, modulation_only, all_except_attention"
        )

    return list(set(target_modules))


def create_lora_model_wrapper(
    model: nn.Module, peft_config: Dict, method: str = "lora"
) -> nn.Module:
    """
    Wrapper function to match the calling signature expected by setup_peft_stage.

    Args:
        model: Base model to adapt
        peft_config: PEFT configuration dictionary
        method: Method type ("lora" or "eva")

    Returns:
        Model with PEFT adapters attached and extracted
    """
    # Get target modules
    target_modules = get_target_modules_for_lora(
        model, strategy=peft_config.get("strategy", "attention_mlp")
    )

    if method.lower() == "eva":
        # EVA configuration
        eva_config = {
            "rho": peft_config.get("rho", 2.0),
            "tau": peft_config.get("tau", 0.99),
            "whiten": peft_config.get("whiten", False),
            "adjust_scaling_factors": peft_config.get("adjust_scaling_factors", True),
            "use_label_mask": peft_config.get("use_label_mask", False),
        }

        return create_lora_model(
            model=model,
            task_type="FEATURE_EXTRACTION",
            r=peft_config.get("r", 64),
            lora_alpha=peft_config.get("lora_alpha", 1),
            lora_dropout=peft_config.get("lora_dropout", 0.0),
            target_modules=target_modules,
            bias=peft_config.get("bias", "none"),
            eva_config=eva_config,
        )
    else:
        # default to LoRA
        return create_lora_model(
            model=model,
            task_type="FEATURE_EXTRACTION",
            r=peft_config.get("r", 64),
            lora_alpha=peft_config.get("lora_alpha", 1),
            lora_dropout=peft_config.get("lora_dropout", 0.0),
            target_modules=target_modules,
            bias=peft_config.get("bias", "none"),
            eva_config=None,
        )


def setup_peft_stage(
    model: nn.Module,
    peft_config: Dict,
    freeze_base_model: bool = True,
    dataloader: Optional[DataLoader] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    Setup PEFT stage with LoRA or EVA adaptation.

    Args:
        model: Pre-trained base model (typically autoencoder)
        peft_config: PEFT configuration dict
        freeze_base_model: Whether to freeze base model parameters
        dataloader: DataLoader for EVA statistics collection (required for EVA)

    Returns:
        Tuple of (peft_model, base_model) where:
        - peft_model: Model with PEFT adapters attached
        - base_model: Original model (for potential restoration)
    """
    method = peft_config.get("method", "lora").lower()

    # setup model
    if method in ["lora", "eva"]:
        adapted_model = create_lora_model_wrapper(model, peft_config, method=method)
    else:
        raise ValueError(f"Unsupported PEFT method: {method}")

    # freeze base model parameters
    if freeze_base_model:
        for name, param in adapted_model.named_parameters():
            if "lora_" not in name:  # Don't freeze LoRA/EVA parameters
                param.requires_grad = False

    # for printing stats
    total_params = sum(p.numel() for p in adapted_model.parameters())
    trainable_params = sum(
        p.numel() for p in adapted_model.parameters() if p.requires_grad
    )

    print(f"PEFT model setup complete:")
    print(f"  Method: {method.upper()}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    return adapted_model, model


def create_lora_model(
    model: nn.Module,
    task_type: str = "FEATURE_EXTRACTION",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    eva_config: Optional[Dict] = None,
) -> PeftModel:
    """
    Create a LoRA model with optional EVA initialization.

    Args:
        model: Base model to apply LoRA to
        task_type: Task type for LoRA config
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA
        bias: Bias configuration
        eva_config: EVA configuration parameters

    Returns:
        PeftModel with LoRA (and optionally EVA) applied
    """

    if eva_config is not None:
        # Use EVA initialization of PEFT
        from peft import EvaConfig

        eva_cfg = EvaConfig(
            rho=eva_config.get("rho", 2.0),
            tau=eva_config.get("tau", 0.99),
            whiten=eva_config.get("whiten", False),
            adjust_scaling_factors=eva_config.get("adjust_scaling_factors", True),
            use_label_mask=eva_config.get("use_label_mask", False),
        )

        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            init_lora_weights="eva",
            eva_config=eva_cfg,
        )
    else:
        # default LoRA
        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
        )

    peft_model = get_peft_model(model, peft_config)

    # IMPORTANT: Extract the base model to avoid PEFT parameter wrapping
    if hasattr(peft_model, "base_model") and hasattr(peft_model.base_model, "model"):
        extracted_model = peft_model.base_model.model
    elif hasattr(peft_model, "model"):
        extracted_model = peft_model.model
    else:
        extracted_model = peft_model.base_model

    # Store PEFT metadata for saving/loading
    extracted_model._peft_config = peft_config
    extracted_model._target_modules = target_modules
    extracted_model._lora_config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
        "task_type": task_type,
    }
    extracted_model._peft_method = "eva" if eva_config is not None else "lora"

    return extracted_model


def get_peft_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get LoRA/EVA parameters from model."""
    peft_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and any(
            key in name for key in ["lora_A", "lora_B", "lora_embedding", "eva_"]
        ):
            peft_params[name] = param.detach().clone()
    return peft_params


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Backward compatibility - alias for get_peft_parameters."""
    return get_peft_parameters(model)


def save_peft_weights(
    model: nn.Module,
    output_dir: Union[str, Path],
    config: Optional[DictConfig] = None,
    training_info: Optional[Dict] = None,
    save_peft_only: bool = True,
    save_full_model: bool = False,
) -> None:
    """
    Save PEFT (LoRA/EVA) weights + metadata

    Args:
        model: Model with PEFT adapters
        output_dir: Directory to save weights
        config: Training configuration
        training_info: Additional training metadata
        save_peft_only: Save only PEFT parameters (default: True)
        save_full_model: Also save full merged model (default: False)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_method = getattr(model, "_peft_method", "lora")

    if save_peft_only:
        peft_params = get_peft_parameters(model)
        if peft_params:
            # Create PEFT-specific checkpoint
            peft_checkpoint = {
                f"{peft_method}_state_dict": peft_params,
                "training_info": training_info or {},
                f"{peft_method}_config": getattr(model, "_lora_config", {}),
                "target_modules": getattr(model, "_target_modules", []),
                "peft_method": peft_method,
            }

            torch.save(peft_checkpoint, output_dir / f"{peft_method}_weights.pth")
        else:
            print(f"Warning: No {peft_method.upper()} parameters found to save")

    if save_full_model:
        full_checkpoint = {
            "model_state_dict": model.state_dict(),
            "training_info": training_info or {},
            "model_config": config,
            "peft_method": peft_method,
        }
        torch.save(full_checkpoint, output_dir / "full_model.pth")
        print(f"Saved full model state to {output_dir}/full_model.pth")

    peft_info = {
        "target_modules": getattr(model, "_target_modules", []),
        f"{peft_method}_config": getattr(model, "_lora_config", {}),
        f"num_{peft_method}_parameters": len(get_peft_parameters(model)),
        f"{peft_method}_parameter_names": list(get_peft_parameters(model).keys()),
        f"save_{peft_method}_only": save_peft_only,
        "save_full_model": save_full_model,
        "peft_method": peft_method,
    }

    if training_info:
        peft_info.update(training_info)

    if config:
        peft_info["base_config"] = {
            "model_type": getattr(config.autoencoder, "model_type", "ae"),
            "latent_dim": config.autoencoder.latent_dim,
            "bottleneck_dim": getattr(config.autoencoder.bottleneck, "dim", None),
        }

    with open(output_dir / f"{peft_method}_info.json", "w") as f:
        json.dump(peft_info, f, indent=2, default=str)

    # print(f"Saved PEFT metadata to {output_dir / f'{peft_method}_info.json'}")


def load_peft_weights_into_model(
    base_model: nn.Module,
    peft_weights_dir: Union[str, Path],
    device: torch.device,
) -> nn.Module:
    """
    Load PEFT weights into a base model.

    Args:
        base_model: The base model to load PEFT weights into
        peft_weights_dir: Directory containing PEFT weights
        device: Device to load weights on

    Returns:
        Model with PEFT weights loaded
    """
    peft_weights_dir = Path(peft_weights_dir)

    # Look for PEFT weights files
    lora_weights_file = peft_weights_dir / "lora_weights.pth"
    eva_weights_file = peft_weights_dir / "eva_weights.pth"

    if lora_weights_file.exists():
        weights_file = lora_weights_file
        method = "lora"
    elif eva_weights_file.exists():
        weights_file = eva_weights_file
        method = "eva"
    else:
        raise FileNotFoundError(f"No PEFT weights found in {peft_weights_dir}")

    # Load the PEFT checkpoint
    checkpoint = torch.load(weights_file, map_location=device, weights_only=True)
    peft_state_dict = checkpoint.get(f"{method}_state_dict", checkpoint)
    peft_method = checkpoint.get("peft_method", method)
    target_modules = checkpoint.get("target_modules", [])

    # Load PEFT parameters into the model
    missing_keys, unexpected_keys = base_model.load_state_dict(
        peft_state_dict, strict=False
    )

    # Store PEFT metadata
    base_model._peft_method = peft_method
    base_model._target_modules = target_modules
    base_model._lora_config = checkpoint.get(f"{method}_config", {})

    print(f"Loaded {len(peft_state_dict)} {method.upper()} parameters")
    if missing_keys:
        print(f"Missing keys (expected for base model): {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return base_model


def save_lora_weights(
    model: nn.Module,
    output_dir: Union[str, Path],
    config: Optional[DictConfig] = None,
    training_info: Optional[Dict] = None,
    save_lora_only: bool = True,
    save_full_model: bool = False,
) -> None:
    """Backward compatibility wrapper for save_peft_weights."""
    save_peft_weights(
        model,
        output_dir,
        config,
        training_info,
        save_peft_only=save_lora_only,
        save_full_model=save_full_model,
    )


def load_lora_weights(
    base_model: nn.Module,
    lora_weights_path: Union[str, Path],
    method: str = "lora_only",
) -> nn.Module:
    """
    load LoRA weights

    Args:
        base_model: The base model to load LoRA weights into
        lora_weights_path: Path to LoRA weights file or directory
        method: "lora_only" or "full_state"

    Returns:
        Model with LoRA weights loaded
    """
    lora_weights_path = Path(lora_weights_path)

    if method == "lora_only":
        if lora_weights_path.is_dir():
            lora_file = lora_weights_path / "lora_weights.pth"
        else:
            lora_file = lora_weights_path

        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA weights file not found: {lora_file}")

        checkpoint = torch.load(lora_file, map_location="cpu")
        lora_state_dict = checkpoint.get("lora_state_dict", checkpoint)

        # Load only LoRA parameters
        model_state_dict = base_model.state_dict()
        loaded_params = 0
        for name, param in lora_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
                loaded_params += 1

        print(f"Loaded {loaded_params} LoRA parameters from {lora_file}")

    elif method == "full_state":
        # Load full model state
        if lora_weights_path.is_dir():
            full_file = lora_weights_path / "full_model.pth"
            if full_file.exists():
                checkpoint = torch.load(full_file, map_location="cpu")
                base_model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded full model from {full_file}")
            else:
                raise FileNotFoundError(f"Full model file not found: {full_file}")
        else:
            checkpoint = torch.load(lora_weights_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                base_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                base_model.load_state_dict(checkpoint)
            print(f"Loaded model state from {lora_weights_path}")

    return base_model


def load_lora_weights(
    base_model: nn.Module,
    lora_weights_path: Union[str, Path],
    method: str = "full_state",
) -> nn.Module:
    """
    load LoRA weights

    Args:
        base_model: The base model to load LoRA weights into
        lora_weights_path: Path to LoRA weights or directory
        method: "full_state" or "lora_only"

    Returns:
        Model with LoRA weights loaded
    """
    lora_weights_path = Path(lora_weights_path)

    if lora_weights_path.is_dir():
        # Directory - look for standard files
        if method == "full_state" and (lora_weights_path / "full_model.pth").exists():
            state_dict_path = lora_weights_path / "full_model.pth"
        elif (lora_weights_path / "lora_weights.pth").exists():
            state_dict_path = lora_weights_path / "lora_weights.pth"
            method = "lora_only"
        else:
            raise FileNotFoundError(f"No LoRA weights found in {lora_weights_path}")
    else:
        # Direct file path
        state_dict_path = lora_weights_path

    # Load weights
    state_dict = torch.load(state_dict_path, map_location="cpu")

    if method == "full_state":
        # Load complete state dict (LoRA weights merged into base model)
        base_model.load_state_dict(state_dict)
        print(f"Loaded full model state from {state_dict_path}")
    elif method == "lora_only":
        # Load only LoRA parameters
        missing_keys, unexpected_keys = base_model.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            print(f"Missing keys (expected for LoRA-only loading): {len(missing_keys)}")
        print(f"Loaded LoRA parameters from {state_dict_path}")

    return base_model


def freeze_base_parameters(
    model: nn.Module, keep_trainable: Optional[List[str]] = None
) -> int:
    """
    Freeze base model parameters, keeping only PEFT parameters trainable.

    Args:
        model: Model with PEFT adapters (LoRA/EVA model)
        keep_trainable: Additional parameter patterns to keep trainable

    Returns:
        Number of trainable parameters
    """
    if keep_trainable is None:
        keep_trainable = []

    trainable_count = 0

    for name, param in model.named_parameters():
        # Keep PEFT parameters trainable (LoRA and EVA)
        is_peft = any(
            key in name for key in ["lora_A", "lora_B", "lora_embedding", "eva_"]
        )
        # Keep additional specified parameters trainable
        is_keep = any(pattern in name for pattern in keep_trainable)

        if is_peft or is_keep:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            # should be frozen but double-check
            if not any(
                key in name for key in ["lora_A", "lora_B", "lora_embedding", "eva_"]
            ):
                param.requires_grad = False

    return trainable_count


def setup_peft_stage(
    base_model: nn.Module,
    config: DictConfig,
    peft_config: Optional[Dict] = None,
    dataloader=None,
) -> Tuple[nn.Module, Dict]:
    """
    Set up model for PEFT fine-tuning stage (supports both LoRA and EVA).

    Args:
        base_model: Pretrained base model
        config: Full configuration
        peft_config: PEFT-specific configuration (LoRA or EVA)
        dataloader: Training dataloader (required for EVA initialization)

    Returns:
        (peft_model, peft_info)
    """

    # get PEFT method from config
    peft_method = getattr(config.autoencoder, "peft", {}).get("method", "lora")

    if peft_config is None:
        if peft_method.lower() == "eva":
            peft_config = getattr(config.autoencoder, "peft", {}).get("eva", {})
        else:
            peft_config = getattr(config.autoencoder, "peft", {}).get("lora", {})

    # Provide defaults if config is empty
    if not peft_config:
        peft_config = {
            "r": 64,
            "lora_alpha": 1,
            "lora_dropout": 0.0,
            "strategy": "comprehensive",
            "bias": "none",
        }
        if peft_method.lower() == "eva":
            peft_config.update(
                {
                    "rho": 2.0,
                    "tau": 0.99,
                    "use_label_mask": True,
                    "whiten": False,
                    "adjust_scaling_factors": True,
                }
            )

    # Get target modules
    target_modules = get_target_modules_for_lora(
        base_model, peft_config.get("strategy", "comprehensive")
    )

    # setup PEFT model
    if peft_method.lower() == "eva":
        # Create EVA model using LoRA config with EVA initialization
        eva_config = {
            "rho": peft_config.get("rho", 2.0),
            "tau": peft_config.get("tau", 0.99),
            "whiten": peft_config.get("whiten", False),
            "adjust_scaling_factors": peft_config.get("adjust_scaling_factors", True),
            "use_label_mask": peft_config.get("use_label_mask", False),
        }

        peft_model = create_lora_model(
            model=base_model,
            task_type="FEATURE_EXTRACTION",
            r=peft_config.get("r", 16),
            lora_alpha=peft_config.get("lora_alpha", 32),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias=peft_config.get("bias", "none"),
            eva_config=eva_config,
        )
    else:
        peft_model = create_lora_model(
            model=base_model,
            task_type="FEATURE_EXTRACTION",
            r=peft_config.get("r", 16),
            lora_alpha=peft_config.get("lora_alpha", 32),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias=peft_config.get("bias", "none"),
        )

    # freeze parameters
    trainable_params = freeze_base_parameters(peft_model)
    total_params = sum(p.numel() for p in peft_model.parameters())

    peft_info = {
        "target_modules": target_modules,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": trainable_params / total_params * 100,
        "peft_config": peft_config,
        "peft_method": peft_method,
    }

    print(f"PEFT ({peft_method.upper()}) Setup Complete:")
    print(f"  Total parameters: {total_params/1e6:.2f}M")
    print(
        f"  Trainable parameters: {trainable_params/1e6:.2f}M ({peft_info['trainable_percentage']:.2f}%)"
    )
    print(f"  Target modules: {len(target_modules)} layers")
    print(f"  Method: {peft_method.upper()}")

    return peft_model, peft_info
