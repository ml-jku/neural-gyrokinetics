"""Module gk_autoencoders.py."""

import warnings
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from neugk.models.layers import MLP
from neugk.models.gk_unet import Swin5DUnet
from neugk.pinc.autoencoders.vector_quantize import (
    VectorQuantize,
    FSQ,
    LFQ,
    ResidualVQ,
)
from neugk.models.nd_vit.vit_layers import ViTLayer
from neugk.models.nd_vit.positional import APE
from neugk.gyroswin.models.x_layers import FluxDecoder


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
        flux_head_config: Optional[Dict] = None,
        mid_norm_learnable: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            conditioning=[] if conditioning else None,
            mid_norm_learnable=mid_norm_learnable,
            **kwargs,
        )

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
        self.middle_pre = self.EncoderGlobalLayerType(
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
        self.middle_post = self.DecoderGlobalLayerType(
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

        # optional heat flux head
        self.eflux_head = None
        if flux_head_config is not None:
            flux_dim = flux_head_config.get("flux_dim", 1)
            head_type = flux_head_config.get("type", "mlp")

            if head_type == "mlp":
                hidden_dim = flux_head_config.get("dim", 128)
                self.eflux_head = MLP(
                    [self.bottleneck_dim, hidden_dim, flux_dim],
                    act_fn=self.act_fn,
                )
            elif head_type == "cross_attn":
                depth = flux_head_config.get("depth", 1)
                num_heads = flux_head_config.get("num_heads", 8)
                mlp_ratio = flux_head_config.get("mlp_ratio", 2.0)
                self.eflux_head = FluxDecoder(
                    left_dims=[self.bottleneck_dim],
                    right_dims=[self.bottleneck_dim],
                    num_heads=num_heads,
                    depth=depth,
                    mlp_ratio=mlp_ratio,
                    act_fn=self.act_fn,
                    init_weights=self.init_weights,
                    reduction="integral",
                )
            else:
                raise ValueError(f"Unknown flux head type: {head_type}")

    def get_compression_info(self):
        """Returns a dictionary with compression-related information."""
        import numpy as np

        input_elements = np.prod(self.base_resolution) * self.problem_dim
        latent_elements = np.prod(self.bottleneck_grid_size) * self.bottleneck_dim
        return {
            "input_elements": int(input_elements),
            "latent_elements": int(latent_elements),
            "input_shape": list(self.base_resolution),
            "input_channels": self.problem_dim,
            "latent_shape": list(self.bottleneck_grid_size),
            "latent_channels": self.bottleneck_dim,
            "rate": input_elements / latent_elements,
            "type": "ae",
        }

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None and condition.shape[-1] != self.enc_cond_dim:
            condition = self.condition(
                {"condition": condition},
                self.enc_cond_embed,
                self.encoder_condition_keys,
                indices=self.enc_indices,
            ).get("condition")

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
        pad_axes: Optional[List] = None,
        condition: Optional[torch.Tensor] = None,
    ):
        if pad_axes is None:
            pad_axes = self.get_pad_axes(self.base_resolution)

        if condition is not None and condition.shape[-1] != self.dec_cond_dim:
            condition = self.condition(
                {"condition": condition},
                self.dec_cond_embed,
                self.decoder_condition_keys,
                indices=self.dec_indices,
            ).get("condition")

        kwcond = {"condition": condition} if condition is not None else {}

        if self.normalized_latent:
            zdf = self.post_z_norm(zdf)

        zdf = self.middle_upproj(zdf)
        zdf = self.middle_post(zdf, **kwcond)
        zdf = self.middle_upscale(zdf)

        for blk in self.up_blocks:
            zdf = blk(zdf, **kwcond)

        return {"df": self.patch_decode(zdf, pad_axes, **kwcond)}

    def forward(
        self,
        df: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ):
        zdf, pad_axes = self.encode(df, condition=condition)
        out = self.decode(zdf, pad_axes, condition=condition)
        if return_latent:
            out["latent"] = zdf

        if self.eflux_head is not None:
            if isinstance(self.eflux_head, MLP):
                z_pooled = zdf.mean(dim=1)
                out["flux"] = self.eflux_head(z_pooled)
            else:
                flux_lat = self.eflux_head.mix(0, zdf, zdf)
                out["flux"] = self.eflux_head([flux_lat])
        return out


class Swin5DVAE(Swin5DAE):
    def __init__(
        self, beta_vae: float = 1.0, logvar_clamp: float = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.beta_vae = beta_vae
        self.logvar_clamp = logvar_clamp

        if self.normalized_latent:
            del self.pre_z_norm
            del self.post_z_norm
            self.normalized_latent = False

        del self.middle_downproj
        self.middle_vae_downproj = nn.Linear(self.middle_dim, 2 * self.bottleneck_dim)

    def get_compression_info(self):
        """Returns a dictionary with compression-related information."""
        import numpy as np

        input_elements = np.prod(self.base_resolution) * self.problem_dim
        latent_elements = np.prod(self.bottleneck_grid_size) * self.bottleneck_dim
        return {
            "input_elements": int(input_elements),
            "latent_elements": int(latent_elements),
            "input_shape": list(self.base_resolution),
            "input_channels": self.problem_dim,
            "latent_shape": list(self.bottleneck_grid_size),
            "latent_channels": self.bottleneck_dim,
            "rate": input_elements / latent_elements,
            "type": "vae",
            "beta_vae": float(self.beta_vae),
        }

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for vae"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None and condition.shape[-1] != self.enc_cond_dim:
            condition = self.condition(
                {"condition": condition},
                self.enc_cond_embed,
                self.encoder_condition_keys,
                indices=self.enc_indices,
            ).get("condition")
        kwcond = {"condition": condition} if condition is not None else {}

        zdf, pad_axes = self.patch_encode(df)
        for blk in self.down_blocks:
            zdf = blk(zdf, return_skip=False, **kwcond)

        if hasattr(self, "middle_pe"):
            zdf = self.middle_pe(zdf)

        zdf = self.middle_pre(zdf, **kwcond)
        mu, logvar = torch.chunk(self.middle_vae_downproj(zdf), 2, dim=-1)
        if self.logvar_clamp is not None:
            logvar = torch.clamp(logvar, min=-self.logvar_clamp, max=self.logvar_clamp)
        z = self.reparameterize(mu, logvar)

        self._mu = mu
        self._logvar = logvar
        return z, pad_axes

    def forward(
        self,
        df: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ):
        zdf, pad_axes = self.encode(df, condition=condition)
        outputs = self.decode(zdf, pad_axes, condition=condition)
        outputs["mu"] = self._mu
        outputs["logvar"] = self._logvar
        if return_latent:
            outputs["latent"] = zdf
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

        # quantizer flavor selector (default "vq" = lucidrains VectorQuantize).
        # all flavors are in-training (quantization in the forward with a STE).
        self.quantizer_type = vq_config.get("quantizer", "vq")
        embedding_dim = vq_config.get("embedding_dim", 256)

        if self.quantizer_type == "vq":
            self.vq = VectorQuantize(
                dim=embedding_dim,
                codebook_size=vq_config.get("codebook_size", 8192),
                commitment_weight=vq_config.get("commitment_weight", 0.25),
                decay=vq_config.get("ema_decay", 0.99),
                use_cosine_sim=(
                    vq_config.get("codebook_type", "euclidean") == "cosine"
                ),
                threshold_ema_dead_code=vq_config.get("threshold_ema_dead_code", 2),
            )
            quantizer_in_dim = embedding_dim
            self.codebook_size = self.vq.codebook_size
        elif self.quantizer_type == "fsq":
            levels = list(vq_config.get("levels", [8, 8, 8, 5, 5, 5]))
            self.vq = FSQ(levels=levels)
            # FSQ operates directly on len(levels) scalar dims
            quantizer_in_dim = self.vq.dim
            self.codebook_size = self.vq.codebook_size
        elif self.quantizer_type == "lfq":
            self.vq = LFQ(
                codebook_size=vq_config.get("codebook_size", 8192),
                entropy_loss_weight=vq_config.get("entropy_loss_weight", 0.1),
                diversity_gamma=vq_config.get("diversity_gamma", 1.0),
                commitment_weight=vq_config.get("commitment_weight", 0.0),
            )
            # LFQ operates on log2(codebook_size) sign dims
            quantizer_in_dim = self.vq.dim
            self.codebook_size = self.vq.codebook_size
        elif self.quantizer_type == "rvq":
            self.vq = ResidualVQ(
                dim=embedding_dim,
                num_quantizers=vq_config.get("num_quantizers", 4),
                codebook_size=vq_config.get("codebook_size", 1024),
                use_cosine_sim=(
                    vq_config.get("codebook_type", "euclidean") == "cosine"
                ),
                decay=vq_config.get("ema_decay", 0.99),
                commitment_weight=vq_config.get("commitment_weight", 0.25),
                threshold_ema_dead_code=vq_config.get("threshold_ema_dead_code", 2),
            )
            quantizer_in_dim = embedding_dim
            self.codebook_size = self.vq.codebook_size
        else:
            raise ValueError(f"Unknown vq.quantizer: {self.quantizer_type}")

        del self.middle_downproj
        del self.middle_upproj
        self.middle_vq_downproj = nn.Linear(self.middle_dim, quantizer_in_dim)
        self.middle_vq_upproj = nn.Linear(quantizer_in_dim, self.middle_dim)

    def get_compression_info(self):
        """Returns a dictionary with compression-related information."""
        import numpy as np

        input_elements = np.prod(self.base_resolution) * self.problem_dim
        num_tokens = np.prod(self.bottleneck_grid_size)
        # codes stored as int16 (2 bytes) in eval; RVQ stores num_quantizers codes
        # per token. Compute CR consistently with the eval accounting.
        n_codes_per_token = getattr(self.vq, "num_quantizers", 1)
        index_bytes = 2  # int16, matches eval reconstructors
        latent_bytes = num_tokens * n_codes_per_token * index_bytes
        rate = (input_elements * 4) / latent_bytes
        return {
            "input_elements": int(input_elements),
            "latent_elements": int(num_tokens * n_codes_per_token),
            "input_shape": list(self.base_resolution),
            "input_channels": self.problem_dim,
            "latent_shape": list(self.bottleneck_grid_size),
            "latent_channels": self.vq.dim,
            "codebook_size": int(self.codebook_size),
            "quantizer": self.quantizer_type,
            "num_codes_per_token": int(n_codes_per_token),
            "rate": float(rate),
            "type": "vqvae",
        }

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if condition is not None and condition.shape[-1] != self.enc_cond_dim:
            condition = self.condition(
                {"condition": condition},
                self.enc_cond_embed,
                self.encoder_condition_keys,
                indices=self.enc_indices,
            ).get("condition")
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

        # indices: (B, tokens) for vq/fsq/lfq, (B, tokens, num_quantizers) for rvq.
        # reshape back to (B, *grid[, num_quantizers]).
        grid = orig_shape[1:-1]
        if indices.dim() == 2:
            self._vq_indices = indices.view((orig_shape[0],) + grid)
        else:
            self._vq_indices = indices.view((orig_shape[0],) + grid + (-1,))
        self._vq_commit_loss = commit_loss
        # z_quantized last dim == quantizer_in_dim == orig_shape[-1]
        return z_quantized.view(orig_shape), pad_axes

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: Optional[List] = None,
        condition: Optional[torch.Tensor] = None,
    ):
        if pad_axes is None:
            pad_axes = self.get_pad_axes(self.base_resolution)

        if condition is not None and condition.shape[-1] != self.dec_cond_dim:
            condition = self.condition(
                {"condition": condition},
                self.dec_cond_embed,
                self.decoder_condition_keys,
                indices=self.dec_indices,
            ).get("condition")
        kwcond = {"condition": condition} if condition is not None else {}

        zdf = self.middle_vq_upproj(zdf)
        zdf = self.middle_post(zdf, **kwcond)
        zdf = self.middle_upscale(zdf)

        for blk in self.up_blocks:
            zdf = blk(zdf, **kwcond)

        return {"df": self.patch_decode(zdf, pad_axes, **kwcond)}

    def decode_from_indices(
        self,
        indices: torch.Tensor,
        pad_axes: Optional[List] = None,
        condition: Optional[torch.Tensor] = None,
    ):
        """Decode from discrete VQ indices back to 5D fields.

        Args:
            indices: (B, seq_len) int64 token indices.
            pad_axes: optional pad axes (defaults to base_resolution pad).
            condition: optional decoder conditioning.
        """
        B = indices.shape[0]
        if self.quantizer_type == "vq":
            codebook = self.vq.codebook.detach()  # (codebook_size, dim)
            z = torch.nn.functional.embedding(indices, codebook)
        elif self.quantizer_type in ("fsq", "lfq"):
            z = self.vq.indices_to_codes(indices)
        else:
            raise NotImplementedError(
                f"decode_from_indices not supported for quantizer {self.quantizer_type}"
            )
        z = z.view(B, *self.bottleneck_grid_size, -1)
        return self.decode(z, pad_axes=pad_axes, condition=condition)

    def get_codebook_usage(self) -> torch.Tensor:
        if hasattr(self, "_vq_indices") and self._vq_indices is not None:
            return self._vq_indices.unique().numel() / self.codebook_size
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

    def forward(
        self,
        df: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ):
        zdf, pad_axes = self.encode(df, condition=condition)
        outputs = self.decode(zdf, pad_axes, condition=condition)
        outputs["vq_commit_loss"] = self._vq_commit_loss
        outputs["vq_indices"] = self._vq_indices
        if return_latent:
            outputs["z"] = zdf
        return outputs


class Swin5DSimSiam(Swin5DAE):
    """Swin5DSimSiam class."""

    def __init__(
        self,
        *args,
        use_simae_decoder: bool = False,
        vit_predictor: bool = True,
        **kwargs,
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
            df = self.decode(zdf, pad_axes, condition=condition)["df"]
            return {"df": df, "z": zdf, "p": pdf}
        else:
            return {"z": zdf, "p": pdf}
