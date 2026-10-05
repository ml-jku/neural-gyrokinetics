"""Helpers for the torch reference: swin residual probe and tiny AE configs."""

from types import SimpleNamespace

from helpers import RES, tiny_ae_model_cfg

STUB_DS = SimpleNamespace(active_keys=["re", "im"], resolution=RES)


def torch_doubles_swin_shortcut() -> bool:
    """True when the torch ``SwinTransformerBlock`` adds the mlp residual twice."""
    import torch
    from neugk.models.nd_vit.swin_layers import SwinTransformerBlock

    blk = SimpleNamespace(
        skip=lambda x: x,
        use_checkpoint=False,
        drop_path=lambda x: x,
        forward_part1=torch.zeros_like,
        forward_part2=torch.zeros_like,
    )
    x = torch.ones(2, 3)
    return bool(torch.allclose(SwinTransformerBlock.forward(blk, x), 2 * x))


def torch_ae_cfg(**extra) -> dict:
    """The tiny test AE config with the keys the torch builder reads."""
    init = {"init_weights": "kaiming_uniform", "patching_init_weights": "kaiming_uniform"}
    m = tiny_ae_model_cfg(act_fn="GELU", norm_fn="RMSNorm", decouple_mu=True, **init, **extra)
    m.setdefault("cond_init_weights", "normal_smallvar")
    m["vit"].update(gradient_checkpoint=False, use_abs_pe=False, use_rope=False)
    m["bottleneck"].update(norm_learnable=False, normalized_latent=False)
    return {"model": m, "dataset": {"separate_zf": True, "resolution": list(RES)}}
