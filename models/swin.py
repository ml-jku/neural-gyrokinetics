import torch
from torch import nn
from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from transformers.models.upernet.modeling_upernet import UperNetHead, UperNetConfig
from transformers.models.swinv2.modeling_swinv2 import Swinv2Config, Swinv2Model


class ConditionerTimestep(nn.Module):
    def __init__(self, dim, num_timesteps):
        super().__init__()
        cond_dim = dim * 4
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.cond_dim = cond_dim
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=num_timesteps, dim=dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, timestep):
        # checks + preprocess
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten()
        # embed
        embed = self.mlp(self.timestep_embed[timestep])
        return embed


class Swin(nn.Module):
    def __init__(
        self,
        input_dim,
        input_window,
        output_dim,
        latent_dim,
        patch_size=1,
        resolution=100,
        condition_ts: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.input_window = input_window
        in_channels = input_dim * input_window

        self.conditioner = None
        cond_dim = 0
        if condition_ts:
            self.conditioner = ConditionerTimestep(latent_dim // 8, 220)
            cond_dim = self.conditioner.cond_dim

        num_heads = [4, 8, 16]
        depths = [2, 6, 2]
        encoder_config = Swinv2Config(
            image_size=resolution,
            patch_size=patch_size,
            num_channels=in_channels,
            embed_dim=latent_dim,
            window_size=4,
            num_heads=num_heads,
            depths=depths,
            qkv_bias=False,
            output_hidden_states=True,
        )
        self.encoder = Swinv2Model(config=encoder_config)

        latent_channels = [h * latent_dim for h in [1, 2, 4, 4]]
        latent_channels[-1] += cond_dim
        decoder_config = UperNetConfig(
            backbone_config=encoder_config,
            hidden_size=4 * latent_dim,
            num_labels=output_dim,
        )
        self.decoder = UperNetHead(decoder_config, in_channels=latent_channels)

    def forward(self, x, grid, ts):
        del grid
        z = list(self.encoder(x).reshaped_hidden_states)
        if self.conditioner:
            cond = self.conditioner(ts)[..., None, None]
            z[-1] = torch.cat([z[-1], cond.repeat(1, 1, 25, 25)], dim=1)
        out = self.decoder(z)
        return out
