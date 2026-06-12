"""Equinox model components (shared building blocks).

``DiT`` lives in ``neugk_jax.diffusion.dit`` and should be imported from
there directly — re-exporting it here would create a circular import via
``embeddings``/``vit``.
"""

from neugk_jax.models.utils import MLP, Film, DiTModulation, Linear, LayerNorm
from neugk_jax.models.embeddings import APE, ContinuousConditionEmbed
from neugk_jax.models.patching import (
    PatchEmbed,
    PatchMerge,
    PatchExpand,
    pad_to_blocks,
    unpad,
)
from neugk_jax.models.swin import SwinLayer, DiTSwinLayer
from neugk_jax.models.vit import ViTLayer, DiTLayer, LayerModes
from neugk_jax.models.gk_unet import SwinNDUnet, Swin5DUnet

__all__ = [
    "MLP",
    "Film",
    "DiTModulation",
    "Linear",
    "LayerNorm",
    "APE",
    "ContinuousConditionEmbed",
    "PatchEmbed",
    "PatchMerge",
    "PatchExpand",
    "pad_to_blocks",
    "unpad",
    "SwinLayer",
    "DiTSwinLayer",
    "ViTLayer",
    "DiTLayer",
    "LayerModes",
    "SwinNDUnet",
    "Swin5DUnet",
]
