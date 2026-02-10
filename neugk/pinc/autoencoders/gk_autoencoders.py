from typing import Optional, List, Dict

from math import prod
import torch
import torch.nn as nn
from einops import rearrange

from neugk.models.layers import MLP
from neugk.models.gk_unet import Swin5DUnet
from neugk.pinc.autoencoders.vector_quantize import VectorQuantize


class Swin5DAE(Swin5DUnet):
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
        # TODO(diff) make conditioning uniform across models
        super().__init__(*args, conditioning=[] if conditioning else None, **kwargs)

        self.bottleneck_dim = bottleneck_dim if bottleneck_dim else self.middle.dim
        self.bottleneck_grid_size = self.middle.grid_size
        self.normalized_latent = normalized_latent

        # Store middle dim before deleting (needed for VAE/VQ-VAE subclasses)
        self.middle_dim = self.middle.dim
        if normalized_latent:
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
        # compress to patch space
        zdf, pad_axes = self.patch_encode(df)

        if condition is not None:
            condition = self.cond_embed(condition)
        # down path
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, condition=condition)

        # bottleneck
        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)  # TODO(diff) middle layers always need PE

        zdf = self.middle_pre(zdf, condition=condition)
        zdf = self.middle_downproj(zdf)

        # layer norm on latents
        if self.normalized_latent:
            zdf = self.pre_z_norm(zdf)

        return zdf, condition, pad_axes

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: List,
        condition: Optional[torch.Tensor] = None,
    ):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)

        # re-normalize latents before bottleneck
        if self.normalized_latent:
            zdf = self.post_z_norm(zdf)

        # bottleneck up
        zdf = self.middle_upproj(zdf)
        zdf = self.middle_post(zdf, condition=condition)
        zdf = self.middle_upscale(zdf)

        # up path
        for blk in self.up_blocks:
            zdf = blk(zdf, condition=condition)

        # expand to original
        df = self.patch_decode(zdf, pad_axes, condition=condition)

        return {"df": df}

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        zdf, condition, pad_axes = self.encode(df, condition=condition)
        return self.decode(zdf, pad_axes, condition=condition)


class Swin5DVAE(Swin5DAE):
    def __init__(self, beta_vae: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta_vae = beta_vae

        # no normalized latent (incompatible with VAE)
        if self.normalized_latent:
            del self.pre_z_norm
            del self.post_z_norm
            self.normalized_latent = False

        # get mu and log_var 2x bottleneck dim
        del self.middle_downproj
        self.middle_vae_downproj = nn.Linear(self.middle_dim, 2 * self.bottleneck_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        # compress to patch space
        zdf, pad_axes = self.patch_encode(df)

        if condition is not None:
            condition = self.cond_embed(condition)

        # down path
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, condition=condition)

        # bottleneck
        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)

        # first Transformer, then project to 2x dim and split
        zdf = self.middle_pre(zdf, condition=condition)
        mu_logvar = self.middle_vae_downproj(zdf)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)

        # sampling via reparameterization trick
        z = self.reparameterize(mu, logvar)

        # for loss computation
        self._mu = mu
        self._logvar = logvar

        return z, condition, pad_axes

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: List,
        condition: Optional[torch.Tensor] = None,
    ):
        if condition is not None and condition.shape[-1] != self.cond_embed.cond_dim:
            condition = self.cond_embed(condition)

        # bottleneck up
        zdf = self.middle_upproj(zdf)
        zdf = self.middle_post(zdf, condition=condition)
        zdf = self.middle_upscale(zdf)

        # up path
        for blk in self.up_blocks:
            zdf = blk(zdf, condition=condition)

        # expand to original
        df = self.patch_decode(zdf, pad_axes, condition=condition)

        return {"df": df}

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        zdf, condition, pad_axes = self.encode(df, condition=condition)
        outputs = self.decode(zdf, pad_axes, condition=condition)

        # for loss computation
        outputs["mu"] = self._mu
        outputs["logvar"] = self._logvar

        return outputs

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss"""
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        # KL = -0.5 * mean/sum(1 + log(σ²) - μ² - σ²)?
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class Swin5DVQVAE(Swin5DAE):
    def __init__(self, *args, vq_config: Dict, **kwargs):
        super().__init__(*args, **kwargs)

        codebook_size = vq_config.get("codebook_size", 8192)
        embedding_dim = vq_config.get("embedding_dim", 256)
        commitment_weight = vq_config.get("commitment_weight", 0.25)
        codebook_type = vq_config.get("codebook_type", "euclidean")
        ema_decay = vq_config.get("ema_decay", 0.99)
        threshold_ema_dead_code = vq_config.get("threshold_ema_dead_code", 2)
        use_cosine_sim = True if codebook_type == "cosine" else False

        # remove normalized latent
        if self.normalized_latent:
            del self.pre_z_norm
            del self.post_z_norm
            self.normalized_latent = False

        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            decay=ema_decay,
            use_cosine_sim=use_cosine_sim,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

        # bottleneck
        del self.middle_downproj
        del self.middle_upproj
        self.middle_vq_downproj = nn.Linear(self.middle_dim, embedding_dim)
        self.middle_vq_upproj = nn.Linear(embedding_dim, self.middle_dim)

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        # compress to patch space
        zdf, pad_axes = self.patch_encode(df)

        if condition is not None:
            condition = self.cond_embed(condition)

        # down path
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, condition=condition)

        # bottleneck
        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)

        zdf = self.middle_pre(zdf, condition=condition)
        z_continuous = self.middle_vq_downproj(zdf)

        # set original shape for reshaping after VQ
        original_shape = z_continuous.shape
        batch_size = original_shape[0]
        spatial_shape = original_shape[1:-1]  # All spatial dimensions
        embedding_dim = original_shape[-1]

        # reshape for VQ: [B, spatial_dims..., embedding_dim] -> [B, prod(spatial_dims), embedding_dim]
        # codebook lookup is via the embedding dimension (TODO(ae) use 4/5D VQ?)
        z_flat = z_continuous.view(batch_size, -1, embedding_dim)

        # VQ lookup
        z_quantized, indices, commit_loss = self.vq(z_flat)

        # reshape back to original shape
        z_quantized = z_quantized.view(original_shape)

        # handle indices
        indices_shape = (batch_size,) + spatial_shape
        self._vq_indices = indices.view(indices_shape)

        # for loss computation
        self._vq_commit_loss = commit_loss

        return z_quantized, condition, pad_axes

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: List,
        condition: Optional[torch.Tensor] = None,
    ):
        # first post-VQ projection
        zdf = self.middle_vq_upproj(zdf)
        zdf = self.middle_post(zdf, condition=condition)

        zdf = self.middle_upscale(zdf)

        # up path
        for blk in self.up_blocks:
            zdf = blk(zdf, condition=condition)

        # expand to original
        df = self.patch_decode(zdf, pad_axes, condition=condition)

        return {"df": df}

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        zdf, condition, pad_axes = self.encode(df, condition=condition)
        outputs = self.decode(zdf, pad_axes, condition=condition)
        outputs["vq_commit_loss"] = self._vq_commit_loss
        outputs["vq_indices"] = self._vq_indices

        return outputs

    def get_codebook_usage(self) -> torch.Tensor:
        """Get codebook usage statistics"""
        if hasattr(self, "_vq_indices") and self._vq_indices is not None:
            return self._vq_indices.unique().numel() / self.vq.codebook_size
        return torch.tensor(0.0)

    def get_codebook_vectors(self) -> torch.Tensor:
        """Get the codebook vectors"""
        if hasattr(self.vq, "embeddings"):
            return self.vq.embeddings.weight.data
        elif hasattr(self.vq, "codebook"):
            return self.vq.codebook.data
        else:
            raise AttributeError("No codebook vectors available in VQ layer.")

    def get_indices(self) -> torch.Tensor:
        """Get the last computed VQ indices"""
        if hasattr(self, "_vq_indices") and self._vq_indices is not None:
            return self._vq_indices
        else:
            raise RuntimeError(
                "No VQ indices available. Run encode() or forward() first."
            )


class Swin5DSimSiam(Swin5DAE):
    def __init__(self, *args, use_simae_decoder: bool = False, **kwargs):
        # TODO(diff) make conditioning uniform across models
        super().__init__(*args, **kwargs)

        predictor_dim = prod(self.bottleneck_grid_size) * self.bottleneck_dim
        self.predictor = MLP([predictor_dim, predictor_dim // 8, predictor_dim])

        self.use_simae_decoder = use_simae_decoder
        if not use_simae_decoder:
            del self.up_blocks
            del self.middle_post
            del self.middle_upproj
            del self.middle_upscale
            del self.unpatch

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None, decoder: bool = False):
        zdf, condition, pad_axes = self.encode(df, condition=condition)
        pdf = self.predictor(zdf.flatten(start_dim=1))
        pdf = pdf.view(pdf.shape[0], *(*self.bottleneck_grid_size, self.bottleneck_dim))
        if decoder and self.use_simae_decoder:
            pred = self.decode(zdf, pad_axes, condition)
            return (zdf, pdf, pred), condition, pad_axes
        else:
            return (zdf, pdf), condition, pad_axes
