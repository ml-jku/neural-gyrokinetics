from typing import Sequence, Union, Optional, Tuple, List, Dict

import torch
from torch import nn
from einops import rearrange

from models.swin_unet import SwinNDUnet, Swin5DUnet
from models.nd_vit.x_layers import MixingBlock, FluxDecoder, VSpaceReduce


class SwinXnet(nn.Module):
    def __init__(
        self,
        dim: int,
        outputs: List[str],
        df_base_resolution: Sequence[int],
        phi_base_resolution: Sequence[int],
        df_patch_size: Union[Sequence[int], int] = 4,
        phi_patch_size: Union[Sequence[int], int] = 4,
        df_window_size: Union[Sequence[int], int] = 5,
        phi_window_size: Union[Sequence[int], int] = 5,
        depth: Union[Sequence[int], int] = 2,
        num_heads: Union[Sequence[int], int] = 4,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 4,
        use_abs_pe: bool = False,
        c_multiplier: int = 2,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
        merging_hidden_ratio: float = 8.0,
        unmerging_hidden_ratio: float = 8.0,
        conditioning: Optional[List[str]] = None,
        cond_embed: Optional[nn.Module] = None,
        modulation: str = "dit",
        act_fn: nn.Module = nn.GELU,
        patch_skip: bool = False,
        decouple_mu: bool = False,
        flux_drop: float = 0.1,
        swin_bottleneck: bool = False,
        separate_zf: bool = False,
        latent_cross_attn: bool = True,
        use_rpb: bool = True,
        use_rope: bool = False,
        detach_flux_latents: bool = False,
        real_potens: bool = False,
        flux_reduce: str = "max",
    ):
        super().__init__()

        self.patch_skip = patch_skip
        self.df_base_resolution = [int(r) for r in df_base_resolution]
        self.df_space = 5
        self.phi_space = 3
        self.problem_dim = in_channels
        self.decouple_mu = decouple_mu
        self.outputs = outputs
        self.latent_dim = dim

        df_in_channels = in_channels
        df_out_channels = out_channels

        phi_in_channels = in_channels if not real_potens else 1
        phi_out_channels = out_channels if not real_potens else 1

        if separate_zf and not real_potens:
            # no separate zf on phi
            phi_in_channels = phi_in_channels - 2
            phi_out_channels = phi_out_channels - 2

        self.df_unet = Swin5DUnet(
            dim=dim,
            base_resolution=self.df_base_resolution,
            patch_size=df_patch_size,
            window_size=df_window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=df_in_channels,
            out_channels=df_out_channels,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            drop_path=drop_path,
            use_abs_pe=use_abs_pe,
            hidden_mlp_ratio=8.0,
            c_multiplier=c_multiplier,
            modulation=modulation,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            cond_embed=cond_embed,
            act_fn=act_fn,
            patch_skip=patch_skip,
            decouple_mu=decouple_mu,
            swin_bottleneck=swin_bottleneck,
        )

        self.phi_unet = SwinNDUnet(
            self.phi_space,
            dim=dim,
            base_resolution=phi_base_resolution,
            patch_size=phi_patch_size,
            window_size=phi_window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=phi_in_channels,
            out_channels=phi_out_channels,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
            drop_path=drop_path,
            use_abs_pe=use_abs_pe,
            conv_patch=True,
            hidden_mlp_ratio=8.0,
            c_multiplier=2,
            modulation=modulation,
            merging_hidden_ratio=merging_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            conditioning=conditioning,
            cond_embed=cond_embed,
            act_fn=act_fn,
            patch_skip=patch_skip,
            swin_bottleneck=swin_bottleneck,
            use_rpb=use_rpb,
            use_rope=use_rope,
        )
        self.phi_up_blocks = self.phi_unet.up_blocks
        self.phi_down_blocks = self.phi_unet.down_blocks

        self.df_up_blocks = self.df_unet.up_blocks
        self.df_down_blocks = self.df_unet.down_blocks

        self.flux_head = True if "flux" in outputs else None
        if self.flux_head:
            self.flux_head = FluxDecoder(
                self.phi_unet.down_dims[::-1],
                self.df_unet.down_dims[::-1],
                num_heads=8,
                drop=flux_drop,
                attn_drop=0.1,
                detach_latents=detach_flux_latents,
                init_weights="xavier_uniform",
                reduction=flux_reduce,
            )

        if latent_cross_attn:
            # down/middle directio
            df_mix = []
            phi_mix = []
            phi_down_dims = self.phi_unet.down_dims
            df_down_dims = self.df_unet.down_dims
            for df_dim, phi_dim in zip(df_down_dims, phi_down_dims):
                df_mix.append(MixingBlock(df_dim, phi_dim, num_heads=8, attn_drop=0.1,
                                          init_weights="xavier_uniform"))
                phi_mix.append(MixingBlock(phi_dim, df_dim, num_heads=8, attn_drop=0.1,
                                           init_weights="xavier_uniform"))
            self.df_mix = nn.ModuleList(df_mix)
            self.phi_mix = nn.ModuleList(phi_mix)

            # up direction
            df_mix_up = []
            phi_mix_up = []
            for df_blk, phi_blk in zip(self.df_up_blocks, self.phi_up_blocks):
                df_mix_up.append(
                    MixingBlock(
                        left_dim=df_blk.dim,
                        right_dim=phi_blk.dim,
                        num_heads=8,
                        attn_drop=0.1,
                        init_weights="xavier_uniform",
                    )
                )
                phi_mix_up.append(
                    MixingBlock(
                        left_dim=phi_blk.dim,
                        right_dim=df_blk.dim,
                        num_heads=8,
                        attn_drop=0.1,
                        init_weights="xavier_uniform",
                    )
                )
            self.df_mix_up = nn.ModuleList(df_mix_up)
            self.phi_mix_up = nn.ModuleList(phi_mix_up)

            df_patch_dim = self.df_unet.unpatch.dim * (2 if patch_skip else 1)
            phi_patch_dim = self.phi_unet.unpatch.dim * (2 if patch_skip else 1)
            self.df_mix_unpatch = MixingBlock(
                left_dim=df_patch_dim, right_dim=phi_patch_dim, num_heads=8, attn_drop=0.1,
                init_weights="xavier_uniform"
            )
            self.phi_mix_unpatch = MixingBlock(
                left_dim=phi_patch_dim, right_dim=df_patch_dim, num_heads=8, attn_drop=0.1,
                init_weights="xavier_uniform"
            )

    def forward(
        self, df: torch.Tensor, phi: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        use_phi = hasattr(self, "phi_unet") and phi is not None
        df, phi, df_pad_axes, phi_pad_axes = self.patch_encode(df, phi, use_phi)

        if self.patch_skip:
            df0 = df.clone()
            phi0 = phi.clone()

        # parameter conditioning
        # TODO why have two?
        df_cond = self.df_unet.condition(kwargs)
        phi_cond = self.phi_unet.condition(kwargs)

        # down paths
        df_features = []
        phi_features = []
        for i, df_blk in enumerate(self.df_down_blocks):
            if hasattr(self, "df_mix"):
                # mix latents
                df, phi = self.df_mix[i](df, phi), self.phi_mix[i](phi, df)
            # down blocks
            df, df_pre = df_blk(df, **df_cond)
            phi, phi_pre = self.phi_down_blocks[i](phi, **phi_cond)
            phi_features.append(phi_pre)
            df_features.append(df_pre)

        # middle blocks + latent mixing
        flux_lats = []
        if hasattr(self.df_unet, "middle_pe"):
            df = self.df_unet.middle_pe(df)
        if use_phi and hasattr(self.phi_unet, "middle_pe"):
            phi = self.phi_unet.middle_pe(phi)

        if hasattr(self, "df_mix"):
            # mix latents
            df, phi = self.df_mix[-1](df, phi), self.phi_mix[-1](phi, df)

        df = self.df_unet.middle(df, **df_cond)
        phi = self.phi_unet.middle(phi, **phi_cond)

        if self.flux_head is not None:
            flux_lats.append(self.flux_head.mix(0, phi, df))

        df = self.df_unet.middle_upscale(df)
        phi = self.phi_unet.middle_upscale(phi)

        # up path
        df_features = df_features[::-1]
        phi_features = phi_features[::-1]
        for i, df_blk in enumerate(self.df_up_blocks):
            if hasattr(self, "df_mix_up"):
                # mix latents
                df, phi = self.df_mix_up[i](df, phi), self.phi_mix_up[i](phi, df)
            # up blocks
            df, df_ = df_blk(df, s=df_features[i], return_skip=True, **df_cond)
            phi, phi_ = self.phi_up_blocks[i](phi, s=phi_features[i], return_skip=True, **phi_cond)
            # multiscale flux latents
            if self.flux_head is not None:
                flux_lats.append(self.flux_head.mix(i + 1, phi_, df_))

        # expand to original
        if self.patch_skip:
            df = torch.cat([df, df0], -1)
            phi = torch.cat([phi, phi0], -1)

        df, phi = self.patch_decode(
            df,
            phi,
            df_pad_axes=df_pad_axes,
            phi_pad_axes=phi_pad_axes,
            use_phi=use_phi,
            df_cond=df_cond.get("condition"),
            phi_cond=None if not use_phi else phi_cond.get("condition"),
        )

        flux = None
        if self.flux_head is not None:
            flux = self.flux_head(flux_lats)

        outputs = {}
        for key, pred in zip(["df", "phi", "flux"], [df, phi, flux]):
            if pred is not None:
                outputs[key] = pred

        return outputs

    def patch_encode(self, df: torch.Tensor, phi: torch.Tensor, use_phi: bool = True):
        # decouple mu and add positional information
        if self.decouple_mu:
            df = rearrange(df, "b c ... -> b ... c")
            df = self.vel_pe(df)
            df = rearrange(df, "b vp mu ... c -> b (c mu) vp ...")
        # compress to patch space
        df, df_pad_axes = self.df_unet.patch_encode(df)
        if use_phi:
            phi, phi_pad_axes = self.phi_unet.patch_encode(phi)
        else:
            phi = phi_pad_axes = None
        return df, phi, df_pad_axes, phi_pad_axes

    def patch_decode(
        self,
        zdf: torch.Tensor,
        zphi: torch.Tensor,
        df_pad_axes: Sequence,
        phi_pad_axes: Sequence,
        use_phi: bool = True,
        df_cond: Optional[torch.Tensor] = None,
        phi_cond: Optional[torch.Tensor] = None,
    ):
        # patch-space mixing
        if hasattr(self, "df_mix_unpatch"):
            # final mixing
            zdf = self.df_mix_unpatch(zdf, zphi)
            if use_phi:
                zphi = self.phi_mix_unpatch(zphi, zdf)
        # expand to original
        df = self.df_unet.patch_decode(zdf, df_pad_axes, cond=df_cond)
        if use_phi:
            phi = self.phi_unet.patch_decode(zphi, phi_pad_axes, cond=phi_cond)
        else:
            phi=None
        return df, phi


class SwinXNetMultitask(SwinXnet):
    def __init__(
        self,
        *args,
        df_base_resolution: Sequence[int],
        df_patch_size: Union[Sequence[int], int] = 4,
        df_window_size: Union[Sequence[int], int] = 5,
        detach_flux_latents: bool = False,
        **kwargs
    ):
        # TODO check if deleting all the unused members
        # NOTE: must use df grids (s, x, y), otherwise shape mismatch
        phi_base_resolution = df_base_resolution[2:]
        phi_patch_size = df_patch_size[2:]
        phi_window_size = df_window_size[2:]
        kwargs.pop("phi_base_resolution")
        kwargs.pop("phi_patch_size")
        kwargs.pop("phi_window_size")
        super().__init__(
            *args,
            df_base_resolution=df_base_resolution,
            df_patch_size=df_patch_size,
            df_window_size=df_window_size,
            phi_base_resolution=phi_base_resolution,
            phi_patch_size=phi_patch_size,
            phi_window_size=phi_window_size,
            detach_flux_latents=detach_flux_latents,
            **kwargs
        )

        # remove down path for phi unet
        self.use_phi = "phi" in self.outputs # check if we use phi
        # remove down-mixing
        self.df_mix_middle = self.df_mix[-1]
        self.phi_mix_middle = self.phi_mix[-1]
        del self.df_mix[:-1]
        del self.phi_mix[:-1]
        del self.phi_unet.patch_embed
        del self.phi_unet.down_blocks
        del self.phi_down_blocks
        if not self.use_phi:
            del self.phi_unet.unpatch
            del self.phi_mix_unpatch
            del self.phi_up_blocks
            del self.phi_unet.up_blocks

        self.phi_middle = self.phi_unet.middle
        self.phi_middle_upscale = self.phi_unet.middle_upscale

        if self.use_phi:
            # phi integrator weights
            vspace_attn_down = []
            for df_blk, phi_blk in zip(self.df_down_blocks, self.phi_up_blocks[::-1]):
                vspace_attn_down.append(
                    VSpaceReduce(
                        dim=df_blk.dim,
                        out_dim=phi_blk.dim,
                        num_heads=8,
                        attn_drop=0.1,
                        decouple_mu=self.df_unet.decouple_mu,
                        init_weights="xavier_uniform",
                    )
                )
            self.vspace_attn_down = nn.ModuleList(vspace_attn_down)

        self.vspace_attn_middle = VSpaceReduce(
            dim=self.df_unet.middle.dim,
            out_dim=self.phi_middle.dim,
            num_heads=8,
            attn_drop=0.1,
            decouple_mu=self.df_unet.decouple_mu,
            init_weights="xavier_uniform",
        )

        if self.patch_skip:
            self.vspace_attn_patch_skip = VSpaceReduce(
                dim=self.df_unet.unpatch.dim,
                out_dim=self.latent_dim,
                num_heads=8,
                attn_drop=0.1,
                decouple_mu=self.df_unet.decouple_mu,
                init_weights="xavier_uniform",
            )

    def forward(self, df: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        df, df_pad_axes = self.df_unet.patch_encode(df)
        phi_pad_axes = df_pad_axes[:6]

        if self.patch_skip:
            df0 = df.clone()
            phi0 = self.vspace_attn_patch_skip(df0)

        # parameter conditioning
        # TODO why have two?
        df_cond = self.df_unet.condition(kwargs)
        phi_cond = self.phi_unet.condition(kwargs)

        # down paths
        df_features = []
        phi_features = []
        for i, df_blk in enumerate(self.df_down_blocks):
            # down blocks
            df, df_pre = df_blk(df, **df_cond)
            df_features.append(df_pre)
            if self.use_phi:
                # integrate out velocity space
                phi_features.append(self.vspace_attn_down[i](df_pre))
            else:
                phi_features.append(None)

        # middle blocks + latent mixing
        flux_lats = []
        if hasattr(self.df_unet, "middle_pe"):
            df = self.df_unet.middle_pe(df)

        # integrate out velocity space
        phi = self.vspace_attn_middle(df)
        if hasattr(self.phi_unet, "middle_pe"):
            phi = self.phi_unet.middle_pe(phi)

        df, phi = self.df_mix_middle(df, phi), self.phi_mix_middle(phi, df)

        df = self.df_unet.middle(df, **df_cond)
        phi = self.phi_middle(phi, **phi_cond)

        if self.flux_head is not None:
            flux_lats.append(self.flux_head.mix(0, phi, df))

        df = self.df_unet.middle_upscale(df)
        phi = self.phi_middle_upscale(phi)

        # up path
        df_features = df_features[::-1]
        for i, (df_blk, df_mix) in enumerate(
            zip(self.df_up_blocks, self.df_mix_up)
        ):
            # mix latents
            df = df_mix(df, phi)
            phi = self.phi_mix_up[i](phi, df)
            # up blocks
            df, df_ = df_blk(df, s=df_features[i], return_skip=True, **df_cond)
            if self.use_phi:
                phi, phi_ = self.phi_up_blocks[i](phi, s=phi_features[i], return_skip=True, **phi_cond)
            else:
                phi_ = phi
            # multiscale flux latents
            if self.flux_head is not None:
                flux_lats.append(self.flux_head.mix(i + 1, phi_, df_))
        # expand to original
        if self.patch_skip:
            df = torch.cat([df, df0], -1)
            phi = torch.cat([phi, phi0], -1)

        df, phi = self.patch_decode(
            df,
            phi,
            df_pad_axes=df_pad_axes,
            phi_pad_axes=phi_pad_axes,
            use_phi=self.use_phi,
            df_cond=df_cond.get("condition"),
            phi_cond=phi_cond.get("condition"),
        )

        out = [df]
        if self.use_phi:
            phi = rearrange(phi, "b c s x y -> b c x s y")
            out += [phi.squeeze()]

        flux = None
        if self.flux_head is not None:
            flux = self.flux_head(flux_lats)
        out += [flux]

        outputs = {}
        for key, pred in zip(self.outputs, out):            
            outputs[key] = pred
        return outputs
