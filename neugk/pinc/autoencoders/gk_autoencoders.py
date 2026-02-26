"""Module gk_autoencoders.py."""

import warnings
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from neugk.models.layers import MLP
from neugk.models.gk_unet import Swin5DUnet
from neugk.pinc.autoencoders.vector_quantize import VectorQuantize
from neugk.models.nd_vit.vit_layers import ViTLayer
from neugk.models.nd_vit.positional import APE


class Swin5DAE(Swin5DUnet):
    """Swin5D autoencoder using a hierarchical Swin transformer backbone."""

    def __init__(
        self,
        *args,
        conditioning: bool = True,
        normalized_latent: bool = True,
        bottleneck_dim: Optional[int] = None,
        bottleneck_num_heads: int = 2,
        bottleneck_depth: int = 2,
        **kwargs
    ):
        super().__init__(*args, conditioning=[] if conditioning else None, **kwargs)

        self.bottleneck_dim = bottleneck_dim or self.middle.dim
        self.bottleneck_grid_size = self.middle.grid_size
        self.normalized_latent = normalized_latent
        self.middle_dim = self.middle.dim

        # optional latent normalization
        if normalized_latent:
            warnings.warn("LayerNorm on latent might lead to scale problems.")
            self.pre_z_norm = nn.LayerNorm(self.bottleneck_dim)
            self.post_z_norm = nn.LayerNorm(self.bottleneck_dim)

        for i in range(len(self.up_blocks)):
            del self.up_blocks[i].proj_concat

        # bottleneck, project channels down
        self.middle_pre = self.GlobalLayerType(
            self.space,
            dim=self.middle.dim,
            grid_size=self.middle.grid_size,
            depth=bottleneck_depth,
            num_heads=bottleneck_num_heads,
            drop_path=self.middle.drop_path,
            mlp_ratio=self.middle.mlp_ratio,
            use_checkpoint=self.middle.use_checkpoint,
            norm_layer=self.middle.norm_layer,
            act_fn=self.middle.act_fn,
            use_rope=self.use_rope,
            gated_attention=self.gated_attention,
        )
        self.middle_downproj = nn.Linear(self.middle.dim, self.bottleneck_dim)
        # channels up
        self.middle_upproj = nn.Linear(self.bottleneck_dim, self.middle.dim)
        self.middle_post = self.GlobalLayerType(
            self.space,
            dim=self.middle.dim,
            grid_size=self.middle.grid_size,
            depth=bottleneck_depth,
            num_heads=bottleneck_num_heads,
            drop_path=self.middle.drop_path,
            mlp_ratio=self.middle.mlp_ratio,
            use_checkpoint=self.middle.use_checkpoint,
            norm_layer=self.middle.norm_layer,
            act_fn=self.middle.act_fn,
            use_rope=self.use_rope,
            gated_attention=self.gated_attention,
        )
        del self.middle

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)
        kwcond = {"condition": condition} if condition is not None else {}

        zdf, pad_axes = self.patch_encode(df)
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, **kwcond)

        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)
        zdf = self.middle_pre(zdf, **kwcond)
        zdf = self.middle_downproj(zdf)

        if self.normalized_latent:
            zdf = self.pre_z_norm(zdf)
        return zdf, pad_axes

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: List,
        condition: Optional[torch.Tensor] = None,
    ):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)
        kwcond = {"condition": condition} if condition is not None else {}

        if self.normalized_latent:
            zdf = self.post_z_norm(zdf)

        zdf = self.middle_upproj(zdf)
        zdf = self.middle_post(zdf, **kwcond)
        zdf = self.middle_upscale(zdf)

        for blk in self.up_blocks:
            zdf = blk(zdf, **kwcond)

        return {"df": self.patch_decode(zdf, pad_axes, **kwcond)}

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None:
            condition = self.cond_embed(condition)
        zdf, pad_axes = self.encode(df, condition=condition)
        return self.decode(zdf, pad_axes, condition=condition)


class Swin5DVAE(Swin5DAE):
    def __init__(self, beta_vae: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_vae = beta_vae

        if self.normalized_latent:
            del self.pre_z_norm
            del self.post_z_norm
            self.normalized_latent = False

        del self.middle_downproj
        self.middle_vae_downproj = nn.Linear(self.middle_dim, 2 * self.bottleneck_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for vae"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)
        kwcond = {"condition": condition} if condition is not None else {}

        zdf, pad_axes = self.patch_encode(df)
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, **kwcond)

        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)

        zdf = self.middle_pre(zdf, **kwcond)
        mu, logvar = torch.chunk(self.middle_vae_downproj(zdf), 2, dim=-1)
        z = self.reparameterize(mu, logvar)

        self._mu = mu
        self._logvar = logvar
        return z, pad_axes

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None:
            condition = self.cond_embed(condition)
        zdf, pad_axes = self.encode(df, condition=condition)
        outputs = self.decode(zdf, pad_axes, condition=condition)
        outputs["mu"] = self._mu
        outputs["logvar"] = self._logvar
        return outputs

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute kl divergence loss"""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class Swin5DVQVAE(Swin5DAE):
    def __init__(self, *args, vq_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)

        if self.normalized_latent:
            del self.pre_z_norm
            del self.post_z_norm
            self.normalized_latent = False

        embedding_dim = vq_config.get("embedding_dim", 256)
        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=vq_config.get("codebook_size", 8192),
            commitment_weight=vq_config.get("commitment_weight", 0.25),
            decay=vq_config.get("ema_decay", 0.99),
            use_cosine_sim=(vq_config.get("codebook_type", "euclidean") == "cosine"),
            threshold_ema_dead_code=vq_config.get("threshold_ema_dead_code", 2),
        )

        del self.middle_downproj
        del self.middle_upproj
        self.middle_vq_downproj = nn.Linear(self.middle_dim, embedding_dim)
        self.middle_vq_upproj = nn.Linear(embedding_dim, self.middle_dim)

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)
        kwcond = {"condition": condition} if condition is not None else {}

        zdf, pad_axes = self.patch_encode(df)
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, **kwcond)

        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)

        zdf = self.middle_pre(zdf, **kwcond)
        z_continuous = self.middle_vq_downproj(zdf)

        orig_shape = z_continuous.shape
        z_quantized, indices, commit_loss = self.vq(
            z_continuous.view(orig_shape[0], -1, orig_shape[-1])
        )

        self._vq_indices = indices.view((orig_shape[0],) + orig_shape[1:-1])
        self._vq_commit_loss = commit_loss
        return z_quantized.view(orig_shape), pad_axes

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: List,
        condition: Optional[torch.Tensor] = None,
    ):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)
        kwcond = {"condition": condition} if condition is not None else {}

        zdf = self.middle_vq_upproj(zdf)
        zdf = self.middle_post(zdf, **kwcond)
        zdf = self.middle_upscale(zdf)

        for blk in self.up_blocks:
            zdf = blk(zdf, **kwcond)

        return {"df": self.patch_decode(zdf, pad_axes, **kwcond)}

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None:
            condition = self.cond_embed(condition)
        zdf, pad_axes = self.encode(df, condition=condition)
        outputs = self.decode(zdf, pad_axes, condition=condition)
        outputs["vq_commit_loss"] = self._vq_commit_loss
        outputs["vq_indices"] = self._vq_indices
        return outputs

    def get_codebook_usage(self) -> torch.Tensor:
        if hasattr(self, "_vq_indices") and self._vq_indices is not None:
            return self._vq_indices.unique().numel() / self.vq.codebook_size
        return torch.tensor(0.0)

    def get_codebook_vectors(self) -> torch.Tensor:
        if hasattr(self.vq, "embeddings"):
            return self.vq.embeddings.weight.data
        if hasattr(self.vq, "codebook"):
            return self.vq.codebook.data
        raise AttributeError("no codebook vectors available in vq layer.")

    def get_indices(self) -> torch.Tensor:
        if hasattr(self, "_vq_indices") and self._vq_indices is not None:
            return self._vq_indices
        raise RuntimeError("no vq indices available. run encode() or forward() first.")


class Swin5DSimSiam(Swin5DAE):
    """Swin5DSimSiam class."""

    def __init__(
        self,
        *args,
        use_simae_decoder: bool = False,
        vit_predictor: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        predictor_dim = self.bottleneck_dim // 8
        if vit_predictor:
            ape = APE(predictor_dim, self.bottleneck_grid_size, init_weights="sincos")
            backbone = ViTLayer(
                space=len(self.bottleneck_grid_size),
                dim=predictor_dim,
                grid_size=self.bottleneck_grid_size,
                depth=2,
                num_heads=4,
                mlp_ratio=2.0,
                act_fn=self.act_fn,
            )
        else:
            ape = nn.Identity()
            backbone = nn.Identity()

        self.predictor = nn.Sequential(
            MLP([self.bottleneck_dim, predictor_dim], act_fn=self.act_fn),
            ape,
            backbone,
            MLP([predictor_dim, self.bottleneck_dim], act_fn=self.act_fn),
        )

        self.use_simae_decoder = use_simae_decoder
        if not use_simae_decoder:
            for attr in [
                "up_blocks",
                "middle_post",
                "middle_upproj",
                "middle_upscale",
                "unpatch",
            ]:
                delattr(self, attr)

    def forward(
        self,
        df: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        decoder: bool = True,
    ):
        zdf, pad_axes = self.encode(df, condition=condition)
        pdf = self.predictor(zdf)
        if decoder and self.use_simae_decoder:
            return {
                "df": self.decode(zdf, pad_axes, condition)["df"],
                "z": zdf,
                "p": pdf,
            }
        return {"z": zdf, "p": pdf}
