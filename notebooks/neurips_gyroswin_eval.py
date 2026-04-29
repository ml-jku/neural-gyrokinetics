"""Loader for the old-codebase GyroSwin checkpoint used as a FID backbone.

The trained xxl-fluxavg checkpoint was produced on a codebase snapshot that
was never merged into neugk. The public `neugk.gyroswin.models.get_model`
cannot rebuild its exact architecture (conditional FiLM flux head, GELU
activations, no LayerNorms). We load the bundled `src/` that ships inside
the checkpoint directory, call its `get_model`, then graft a reconstructed
`ConditionalFluxDecoder` on top so `load_state_dict(strict=True)` succeeds.

After loading, the model exposes `df_unet` and `flux_head` with the same
interface that `notebooks/neurips_diff_eval.py::extract_gyroswin_latents`
expects (both `source="bottleneck"` and `source="flux_head"` work).
"""
from __future__ import annotations

import os
import sys

import omegaconf
import torch
from torch import nn


def _prepare_old_src(checkpoint_dir: str) -> str:
    old_src = os.path.join(checkpoint_dir, "src")
    if not os.path.isdir(old_src):
        raise FileNotFoundError(
            f"expected bundled old codebase at {old_src}; this loader only "
            "works for checkpoints that shipped their training-time src/"
        )
    if old_src not in sys.path:
        sys.path.insert(0, old_src)
    for k in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
        del sys.modules[k]
    return old_src


def _build_conditional_flux_decoder(
    base: nn.Module,
    n_cond: int,
    num_heads: int,
    depth: int,
    cond_embed_dim: int = 128,
    drop: float = 0.0,
    attn_drop: float = 0.1,
) -> nn.Module:
    from models.utils import MLP, ContinuousConditionEmbed, Film
    from models.nd_vit.x_layers import MixingBlock

    class _CondLatentMixingTransformer(nn.Module):
        def __init__(self, left_dim, right_dim, depth, num_heads, n_cond,
                     cond_embed_dim, mlp_ratio, attn_drop, drop):
            super().__init__()
            self.left_dim = left_dim
            self.right_dim = right_dim
            self.cond_embed = ContinuousConditionEmbed(dim=cond_embed_dim, n_cond=n_cond)
            self.blocks = nn.ModuleList([
                MixingBlock(
                    left_dim=left_dim, right_dim=right_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=True,
                    drop=drop, attn_drop=attn_drop, drop_path=0.0,
                    act_fn=nn.GELU, init_weights=None,
                )
                for _ in range(depth)
            ])
            self.conditioning = nn.ModuleList(
                [Film(self.cond_embed.cond_dim, left_dim) for _ in range(depth)]
            )

        def forward(self, left, right, cond):
            condition = self.cond_embed(cond)
            for blk, film in zip(self.blocks, self.conditioning):
                left = film(left, condition)
                x = blk(left, right)
            return x

    class ConditionalFluxDecoder(nn.Module):
        def __init__(self, base, n_cond, num_heads, depth, cond_embed_dim, drop, attn_drop):
            super().__init__()
            self.detach_latents = base.detach_latents
            self.reduction = base.reduction
            self.reductions = base.reductions

            self.blocks = nn.ModuleList([
                _CondLatentMixingTransformer(
                    left_dim=blk.left_dim, right_dim=blk.right_dim,
                    depth=depth, num_heads=num_heads, n_cond=n_cond,
                    cond_embed_dim=cond_embed_dim, mlp_ratio=2.0,
                    attn_drop=attn_drop, drop=drop,
                )
                for blk in base.blocks
            ])
            self.cond_embed = ContinuousConditionEmbed(dim=cond_embed_dim, n_cond=n_cond)

            flux_latent_size = sum(b.left_dim for b in self.blocks)
            self.flux_mlp = MLP(
                [flux_latent_size, flux_latent_size // 2, 1],
                dropout_prob=drop, act_fn=nn.GELU,
            )
            self._cond_cache = None

        def mix(self, i, left, right=None):
            if self.detach_latents:
                left = left.detach()
                right = right.detach()
            x = self.blocks[i](left, right, self._cond_cache)
            if self.reduction == "max":
                x = x.amax(axis=list(range(1, x.ndim - 1)))
            elif self.reduction == "mean":
                x = x.mean(axis=list(range(1, x.ndim - 1)))
            else:
                x = self.reductions[i].forward(x)
            return x

        def forward(self, flux_latents):
            flux = self.flux_mlp(torch.cat(flux_latents, dim=-1))
            return flux.squeeze(1)

    return ConditionalFluxDecoder(
        base, n_cond=n_cond, num_heads=num_heads, depth=depth,
        cond_embed_dim=cond_embed_dim, drop=drop, attn_drop=attn_drop,
    )


def _get_gyroswin_model(cfg, dataset):
    """Drop-in for the old codebase's `get_model`. Must be called after
    `_prepare_old_src` so the `models` import resolves to the checkpoint's
    bundled src/."""
    from models import get_model as _old_get_model
    return _old_get_model(cfg, dataset=dataset)


def load_gyroswin_model(
    checkpoint_dir: str,
    dataset,
    device,
    ckpt_name: str = "best.pth",
):
    """Load the old-codebase GyroSwin checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing `best.pth`, `config.yaml`, and `src/`.
    dataset : object
        Dataset the old `get_model` reads shape info from (the diffusion
        runner's `trainset` is what the notebooks pass).
    device : torch.device or str
    ckpt_name : str
        Checkpoint filename inside `checkpoint_dir` (default `best.pth`).

    Returns
    -------
    model : nn.Module
        Eval-mode GyroSwin with `df_unet` + grafted `ConditionalFluxDecoder`
        flux_head. Compatible with `extract_gyroswin_latents`
        (both `source="bottleneck"` and `source="flux_head"`).
    cfg : DictConfig
        The config loaded from `checkpoint_dir/config.yaml`.
    ckpt : dict
        Raw torch.load payload (caller can inspect `epoch` / `loss`).
    """
    _prepare_old_src(checkpoint_dir)

    cfg = omegaconf.OmegaConf.load(os.path.join(checkpoint_dir, "config.yaml"))
    cfg.model.loss_weights.flux = 1.0  # force flux_head instantiation

    model = _get_gyroswin_model(cfg, dataset=dataset).to(device)

    model.flux_head = _build_conditional_flux_decoder(
        model.flux_head,
        n_cond=len(cfg.model.conditioning),
        num_heads=cfg.model.swin.flux_num_heads,
        depth=cfg.model.swin.flux_depth,
    ).to(device)

    cond_keys = sorted(list(cfg.model.conditioning))

    def _stash_flux_cond(module, args, kwargs):
        cond = torch.cat([kwargs[k].reshape(-1, 1) for k in cond_keys], dim=-1)
        module.flux_head._cond_cache = cond

    model.register_forward_pre_hook(_stash_flux_cond, with_kwargs=True)

    ckpt = torch.load(
        os.path.join(checkpoint_dir, ckpt_name),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    return model, cfg, ckpt
