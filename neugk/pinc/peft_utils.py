"""PEFT (LoRA / EVA) utilities for the PINC autoencoder fine-tuning stage.

The physics-loss fine-tuning stage takes a pretrained autoencoder, attaches
low-rank adapters (plain LoRA or EVA-initialised LoRA) to a chosen subset of its
linear layers, freezes the base weights, and trains only the adapters against the
``PINCLossWrapper`` physics losses.

Public surface:
    - ``find_linear_layers``        : discover ``nn.Linear`` modules by qualified name.
    - ``get_target_modules_for_lora``: pick adapter targets via a named strategy.
    - ``attach_peft_adapters``      : the single, reusable adapter-attach entry point.
    - ``setup_peft_stage``          : build adapters from a Hydra config (training).
    - ``create_lora_model_wrapper`` : build adapters from a plain dict (checkpoint reload).
"""

from typing import Dict, List, Optional, Tuple

from torch import nn
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, EvaConfig


# adapter parameter name fragments; used to select muon groups and filter base-vs-adapter weights in checkpoints
PEFT_PARAM_KEYS = ("lora_A", "lora_B", "lora_embedding", "eva_")

# named strategies map to ordered layer groups to adapt
_STRATEGY_GROUPS = {
    "comprehensive": ("attention_qkv", "attention_proj", "mlp_layers", "bottleneck"),
    "attention_mlp": ("attention_qkv", "attention_proj", "mlp_layers"),
    "mlp_only": ("mlp_layers",),
    "attention_only": ("attention_qkv", "attention_proj"),
    "bottleneck_only": ("bottleneck",),
    "modulation_only": ("modulation",),
    "all_except_attention": (
        "mlp_layers",
        "bottleneck",
        "patch_embed",
        "modulation",
        "downsample",
    ),
}

# default adapter config when none supplied
_DEFAULT_PEFT = {
    "r": 64,
    "lora_alpha": 1,
    "lora_dropout": 0.0,
    "strategy": "comprehensive",
    "bias": "none",
}
_DEFAULT_EVA = {
    "rho": 2.0,
    "tau": 0.99,
    "use_label_mask": True,
    "whiten": False,
    "adjust_scaling_factors": True,
}
_EVA_KEYS = ("rho", "tau", "whiten", "adjust_scaling_factors", "use_label_mask")


def find_linear_layers(
    model: nn.Module, prefix: str = ""
) -> List[Tuple[str, nn.Linear]]:
    """Recursively collect all ``nn.Linear`` layers as ``(qualified_name, module)``."""
    linear_layers = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            linear_layers.append((full_name, module))
        else:
            linear_layers.extend(find_linear_layers(module, full_name))
    return linear_layers


def _classify_linear(name: str) -> str:
    """Map a linear layer's qualified name to one of the layer groups."""
    if "patch_embed.patch.mlp" in name:
        return "patch_embed"
    if "qkv" in name:
        return "attention_qkv"
    if "attn.proj" in name or "attention.proj" in name:
        return "attention_proj"
    if any(
        p in name
        for p in ["mlp.mlp.", "mlp.", "cond_embed.mlp", "cpb_mlp.mlp", "feed_forward"]
    ):
        return "mlp_layers"
    if any(p in name for p in ["middle_downproj", "middle_upproj", "bottleneck"]):
        return "bottleneck"
    if "modulation" in name:
        return "modulation"
    if "downsample" in name or "upsample" in name:
        return "downsample"
    return "other"


def get_target_modules_for_lora(
    model: nn.Module,
    strategy: str = "comprehensive",
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Return the linear-layer names to adapt for a given strategy."""
    if strategy not in _STRATEGY_GROUPS:
        raise ValueError(f"unknown strategy: {strategy}")
    exclude_patterns = exclude_patterns or []

    groups: Dict[str, List[str]] = {}
    for name, _ in find_linear_layers(model):
        if any(p in name for p in exclude_patterns):
            continue
        groups.setdefault(_classify_linear(name), []).append(name)

    target_modules: List[str] = []
    for group in _STRATEGY_GROUPS[strategy]:
        target_modules.extend(groups.get(group, []))
    return list(set(target_modules))


def attach_peft_adapters(
    model: nn.Module,
    *,
    method: str = "lora",
    r: int = 64,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    bias: str = "none",
    target_modules: Optional[List[str]] = None,
    strategy: str = "comprehensive",
    eva_config: Optional[Dict] = None,
    task_type: str = "FEATURE_EXTRACTION",
    freeze_base: bool = True,
) -> nn.Module:
    """Attach LoRA/EVA adapters to ``model`` and return the (unwrapped) base model.

    This is the single reusable entry point for the physics-loss fine-tuning stage:
    given a pretrained autoencoder, it wraps the chosen linear layers with low-rank
    adapters, optionally freezes the base weights, and returns the model ready to be
    trained against a ``PINCLossWrapper``.

    Args:
        method: ``"lora"`` for random init, ``"eva"`` for explained-variance init.
        target_modules: explicit module names; if ``None`` they are derived from
            ``strategy`` via :func:`get_target_modules_for_lora`.
        eva_config: EVA hyperparameters; only used when ``method == "eva"``.
        freeze_base: freeze every non-adapter parameter (the standard PEFT setup).

    The returned model carries ``_peft_method`` / ``_lora_config`` / ``_target_modules``
    attributes for downstream logging and checkpointing.
    """
    if target_modules is None:
        target_modules = get_target_modules_for_lora(model, strategy)

    use_eva = method.lower() == "eva"
    lora_kwargs = dict(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )
    if use_eva:
        eva_config = eva_config or _DEFAULT_EVA
        lora_kwargs["init_lora_weights"] = "eva"
        lora_kwargs["eva_config"] = EvaConfig(
            rho=eva_config.get("rho", 2.0),
            tau=eva_config.get("tau", 0.99),
            whiten=eva_config.get("whiten", False),
            adjust_scaling_factors=eva_config.get("adjust_scaling_factors", True),
            use_label_mask=eva_config.get("use_label_mask", False),
        )
    peft_config = LoraConfig(**lora_kwargs)

    peft_model = get_peft_model(model, peft_config)

    # unwrap to underlying base model (PEFT injects adapters in place)
    if hasattr(peft_model, "base_model") and hasattr(peft_model.base_model, "model"):
        base = peft_model.base_model.model
    elif hasattr(peft_model, "model"):
        base = peft_model.model
    else:
        base = peft_model.base_model

    base._peft_method = "eva" if use_eva else "lora"
    base._target_modules = target_modules
    base._lora_config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
        "task_type": task_type,
    }

    if freeze_base:
        for name, param in base.named_parameters():
            param.requires_grad = any(k in name for k in PEFT_PARAM_KEYS)

    return base


def create_lora_model_wrapper(
    model: nn.Module, peft_config: Dict, method: str = "lora"
) -> nn.Module:
    """Attach adapters from a plain config dict (used when reloading checkpoints)."""
    eva_config = (
        {k: peft_config.get(k) for k in _EVA_KEYS} if method.lower() == "eva" else None
    )
    return attach_peft_adapters(
        model,
        method=method,
        r=peft_config.get("r", 64),
        lora_alpha=peft_config.get("lora_alpha", 1),
        lora_dropout=peft_config.get("lora_dropout", 0.0),
        bias=peft_config.get("bias", "none"),
        strategy=peft_config.get("strategy", "attention_mlp"),
        eva_config=eva_config,
        freeze_base=False,  # checkpoint reload restores requires_grad via state dict
    )


def setup_peft_stage(
    base_model: nn.Module,
    config: DictConfig,
    peft_config: Optional[Dict] = None,
) -> Tuple[nn.Module, Dict]:
    """Attach adapters for fine-tuning stage from Hydra ``autoencoder.peft`` config; return model + info dict."""
    # model block lives under "autoencoder" or "model"
    model_cfg = getattr(config, "autoencoder", None) or getattr(config, "model", None)
    peft_cfg = getattr(model_cfg, "peft", {}) if model_cfg is not None else {}
    method = peft_cfg.get("method", "lora") if peft_cfg else "lora"
    peft_config = peft_config or (peft_cfg.get(method.lower(), {}) if peft_cfg else {})

    if not peft_config:
        peft_config = dict(_DEFAULT_PEFT)
        if method.lower() == "eva":
            peft_config.update(_DEFAULT_EVA)

    eva_config = (
        {k: peft_config.get(k) for k in _EVA_KEYS} if method.lower() == "eva" else None
    )

    model = attach_peft_adapters(
        base_model,
        method=method,
        r=peft_config.get("r", 16),
        lora_alpha=peft_config.get("lora_alpha", 32),
        lora_dropout=peft_config.get("lora_dropout", 0.1),
        bias=peft_config.get("bias", "none"),
        strategy=peft_config.get("strategy", "comprehensive"),
        eva_config=eva_config,
        freeze_base=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {
        "target_modules": getattr(model, "_target_modules", []),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": trainable / total * 100 if total else 0.0,
        "peft_config": peft_config,
        "peft_method": method,
    }
