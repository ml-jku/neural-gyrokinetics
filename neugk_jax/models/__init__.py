"""Equinox model components (shared building blocks).

``DiT`` lives in ``neugk_jax.diffusion.dit`` and should be imported from
there directly — re-exporting it here would create a circular import via
``embeddings``/``vit``.
"""

from neugk_jax.models.embeddings import APE, ContinuousConditionEmbed
from neugk_jax.models.gk_unet import Swin5DUnet, SwinNDUnet
from neugk_jax.models.patching import PatchEmbed, PatchExpand, PatchMerge, pad_to_blocks, unpad
from neugk_jax.models.swin import BlockStack, Film, swin_layer
from neugk_jax.models.utils import MLP, DiTModulation, LayerNorm, Linear
from neugk_jax.models.vit import vit_layer

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
    "BlockStack",
    "swin_layer",
    "vit_layer",
    "SwinNDUnet",
    "Swin5DUnet",
]
