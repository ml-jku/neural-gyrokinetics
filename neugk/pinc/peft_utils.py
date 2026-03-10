import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, PeftModel, EvaConfig


def find_linear_layers(
    model: nn.Module, prefix: str = ""
) -> List[Tuple[str, nn.Linear]]:
    """Recursively find all linear layers in the model"""
    linear_layers = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            linear_layers.append((full_name, module))
        else:
            linear_layers.extend(find_linear_layers(module, full_name))
    return linear_layers


def get_target_modules_for_lora(
    model: nn.Module,
    strategy: str = "comprehensive",
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Get target module names for lora adaptation based on strategy"""
    exclude_patterns = exclude_patterns or []
    linear_layers = find_linear_layers(model)

    layer_groups = {
        "attention_qkv": [],
        "attention_proj": [],
        "mlp_layers": [],
        "bottleneck": [],
        "patch_embed": [],
        "modulation": [],
        "downsample": [],
        "other": [],
    }

    for name, _ in linear_layers:
        if any(p in name for p in exclude_patterns):
            continue

        if "patch_embed.patch.mlp" in name:
            layer_groups["patch_embed"].append(name)
        elif "qkv" in name:
            layer_groups["attention_qkv"].append(name)
        elif "attn.proj" in name or "attention.proj" in name:
            layer_groups["attention_proj"].append(name)
        elif any(
            p in name
            for p in [
                "mlp.mlp.",
                "mlp.",
                "cond_embed.mlp",
                "cpb_mlp.mlp",
                "feed_forward",
            ]
        ):
            layer_groups["mlp_layers"].append(name)
        elif any(p in name for p in ["middle_downproj", "middle_upproj", "bottleneck"]):
            layer_groups["bottleneck"].append(name)
        elif "modulation" in name:
            layer_groups["modulation"].append(name)
        elif "downsample" in name or "upsample" in name:
            layer_groups["downsample"].append(name)
        else:
            layer_groups["other"].append(name)

    target_modules = []
    if strategy == "comprehensive":
        target_modules.extend(
            layer_groups["attention_qkv"]
            + layer_groups["attention_proj"]
            + layer_groups["mlp_layers"]
            + layer_groups["bottleneck"]
        )
    elif strategy == "attention_mlp":
        target_modules.extend(
            layer_groups["attention_qkv"]
            + layer_groups["attention_proj"]
            + layer_groups["mlp_layers"]
        )
    elif strategy == "mlp_only":
        target_modules.extend(layer_groups["mlp_layers"])
    elif strategy == "attention_only":
        target_modules.extend(
            layer_groups["attention_qkv"] + layer_groups["attention_proj"]
        )
    elif strategy == "bottleneck_only":
        target_modules.extend(layer_groups["bottleneck"])
    elif strategy == "modulation_only":
        target_modules.extend(layer_groups["modulation"])
    elif strategy == "all_except_attention":
        target_modules.extend(
            layer_groups["mlp_layers"]
            + layer_groups["bottleneck"]
            + layer_groups["patch_embed"]
            + layer_groups["modulation"]
            + layer_groups["downsample"]
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return list(set(target_modules))


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
    """Create a lora model with optional eva initialization"""
    if eva_config is not None:
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
        peft_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
        )

    peft_model = get_peft_model(model, peft_config)

    # extract base model
    if hasattr(peft_model, "base_model") and hasattr(peft_model.base_model, "model"):
        extracted_model = peft_model.base_model.model
    elif hasattr(peft_model, "model"):
        extracted_model = peft_model.model
    else:
        extracted_model = peft_model.base_model

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
    extracted_model._peft_method = "eva" if eva_config else "lora"

    return extracted_model


def create_lora_model_wrapper(
    model: nn.Module, peft_config: Dict, method: str = "lora"
) -> nn.Module:
    """Wrapper to create lora model"""
    target_modules = get_target_modules_for_lora(
        model, strategy=peft_config.get("strategy", "attention_mlp")
    )
    eva_config = None
    if method.lower() == "eva":
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


def get_peft_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get lora/eva parameters"""
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
        and any(k in n for k in ["lora_A", "lora_B", "lora_embedding", "eva_"])
    }


def save_peft_weights(
    model: nn.Module,
    output_dir: Union[str, Path],
    config: Optional[DictConfig] = None,
    training_info: Optional[Dict] = None,
    save_peft_only: bool = True,
    save_full_model: bool = False,
) -> None:
    """Save peft weights and metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_method = getattr(model, "_peft_method", "lora")

    if save_peft_only:
        peft_params = get_peft_parameters(model)
        if peft_params:
            torch.save(
                {
                    f"{peft_method}_state_dict": peft_params,
                    "training_info": training_info or {},
                    f"{peft_method}_config": getattr(model, "_lora_config", {}),
                    "target_modules": getattr(model, "_target_modules", []),
                    "peft_method": peft_method,
                },
                output_dir / f"{peft_method}_weights.pth",
            )
        else:
            print(f"warning: no {peft_method} parameters found")

    if save_full_model:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_info": training_info or {},
                "model_config": config,
                "peft_method": peft_method,
            },
            output_dir / "full_model.pth",
        )

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


def freeze_base_parameters(
    model: nn.Module, keep_trainable: Optional[List[str]] = None
) -> int:
    """Freeze base model parameters, keeping only peft parameters trainable"""
    keep_trainable = keep_trainable or []
    trainable_count = 0
    for name, param in model.named_parameters():
        is_peft = any(k in name for k in ["lora_A", "lora_B", "lora_embedding", "eva_"])
        is_keep = any(p in name for p in keep_trainable)
        if is_peft or is_keep:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    return trainable_count


def setup_peft_stage(
    base_model: nn.Module,
    config: DictConfig,
    peft_config: Optional[Dict] = None,
    dataloader=None,
) -> Tuple[nn.Module, Dict]:
    """Set up peft model"""
    peft_method = getattr(config.autoencoder, "peft", {}).get("method", "lora")
    peft_config = peft_config or getattr(config.autoencoder, "peft", {}).get(
        peft_method.lower(), {}
    )
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

    target_modules = get_target_modules_for_lora(
        base_model, peft_config.get("strategy", "comprehensive")
    )

    eva_config = None
    if peft_method.lower() == "eva":
        eva_config = {
            k: peft_config.get(k)
            for k in [
                "rho",
                "tau",
                "whiten",
                "adjust_scaling_factors",
                "use_label_mask",
            ]
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

    trainable_params = freeze_base_parameters(peft_model)
    total_params = sum(p.numel() for p in peft_model.parameters())
    return peft_model, {
        "target_modules": target_modules,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": trainable_params / total_params * 100,
        "peft_config": peft_config,
        "peft_method": peft_method,
    }
