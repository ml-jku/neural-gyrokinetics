"""Neural gyrokinetics swin autoencoder."""

import torch

from neugk.models.gk_unet import SwinNDUnet


class SwinAE(SwinNDUnet):
    """N-dimensional shifted window transformer autoecoder implementation (v1/v2)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for i in range(len(self.up_blocks)):
            del self.up_blocks[i].proj_concat

    def forward(
        self, x: torch.Tensor, return_latents: bool = False, **kwargs
    ) -> torch.Tensor:
        lats = {}
        # compress to patch space
        x, pad_axes = self.patch_encode(x)
        if return_latents:
            lats["patch"] = x

        # backbone
        cond = self.condition(kwargs)

        # down path
        for i, blk in enumerate(self.down_blocks):
            x, x_pre = blk(x, return_skip=True, **cond)
            if return_latents:
                lats[f"swin_pre{i}"] = x_pre
                lats[f"swin_down{i}"] = x

        # middle block
        if hasattr(self, "middle_pe"):
            x = self.middle_pe(x)
        x = self.middle(x, **cond)
        if return_latents:
            lats["middle"] = x
        x = self.middle_upscale(x)
        x = x_pre
        if return_latents:
            lats["middle_up"] = x
        # up path
        for i, blk in enumerate(self.up_blocks):
            x = blk(x, **cond)
            if return_latents:
                lats[f"swin_up{i}"] = x

        # expand to original
        x = self.patch_decode(x, cond.get("condition", None), pad_axes)

        if return_latents:
            return x, lats
        else:
            return x
