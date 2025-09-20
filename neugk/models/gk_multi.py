"""Neural gyrokinetics multitarget models."""

from typing import Sequence, Union, Optional, Tuple, List, Dict

import torch
from torch import nn
from einops import rearrange
from functools import partial

from neugk.models.gk_unet import SwinNDUnet, Swin5DUnet, SwinBlockUp
from neugk.models.nd_vit.x_layers import MixingBlock, FluxDecoder, VSpaceReduce, BidirectionalMixingBlock
from neugk.models.nd_vit.swin_layers import FilmSwinLayer, DiTSwinLayer, SwinLayer
from neugk.models.nd_vit.vit_layers import LayerModes
from neugk.models.nd_vit.patching import PatchUnmerging, unpad


class NeuGK(nn.Module):
    """Neural Gyrokinetics model for predicting the 5d density and the 3d potential."""
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
        flux_num_heads: int = 8,
        flux_depth: int = 1,
        flux_cond_embed: Optional[nn.Module] = None,
        init_weights: str = "xavier_uniform",
        patching_init_weights: str = "xavier_uniform",
        cond_init_weights: str = "normal_smallvar",
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
            init_weights=init_weights,
            cond_init_weights=cond_init_weights,
            patching_init_weights=patching_init_weights
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
            init_weights=init_weights,
            cond_init_weights=cond_init_weights,
            patching_init_weights=patching_init_weights
        )
        self.phi_up_blocks = self.phi_unet.up_blocks
        self.phi_down_blocks = self.phi_unet.down_blocks

        self.df_up_blocks = self.df_unet.up_blocks
        self.df_down_blocks = self.df_unet.down_blocks

        use_flux_head = "flux" in outputs or "fluxavg" in outputs or "fluxfield" in outputs
        self.flux_head = True if use_flux_head else None
        self.decode_fluxfield = False
        if self.flux_head:
            if "flux" in outputs or "fluxavg" in outputs:
                # scalar flux head
                self.flux_head = FluxDecoder(
                    self.phi_unet.down_dims[::-1],
                    self.df_unet.down_dims[::-1],
                    num_heads=flux_num_heads,
                    depth=flux_depth,
                    drop=flux_drop,
                    attn_drop=0.1,
                    detach_latents=detach_flux_latents,
                    init_weights=init_weights,
                    reduction=flux_reduce,
                    cond_embed=flux_cond_embed,
                )
            else:
                self.decode_fluxfield = True
                self.flux_cond_embed = flux_cond_embed
                self.condition_keys = ['dg', 'itg', 'q', 's_hat', 'timestep']
                flux_space = 2
                self.flux_mix = BidirectionalMixingBlock(
                    self.phi_unet.down_dims[-1],
                    self.df_unet.down_dims[-1],
                    self.phi_unet.down_dims[-1],
                    num_heads=8,
                    attn_drop=0.1,
                    init_weights=init_weights
                )
                # TODO: change based on flux_space
                space_diff = self.df_space - flux_space
                up_dims = self.df_unet.down_dims[::-1][1:]
                up_grid_sizes = self.df_unet.grid_sizes[::-1][1:]
                middle_grid_size = self.df_unet.grid_sizes[-1]
                if decouple_mu:
                    middle_grid_size = middle_grid_size[space_diff-1:]
                    up_grid_sizes = [gs[space_diff-1:] for gs in up_grid_sizes]
                else:
                    middle_grid_size = middle_grid_size[space_diff:]
                    up_grid_sizes = [gs[space_diff:] for gs in up_grid_sizes]
                up_depth = depth[::-1]
                up_num_heads = num_heads[::-1]
                out_channels = 2
                if decouple_mu and flux_space == self.df_space:
                    df_window_size = [df_window_size[0]] + df_window_size[2:]
                    flux_space = self.df_space - 1
                    df_patch_size = [df_patch_size[0]] + df_patch_size[2:]
                    out_channels = out_channels * self.df_unet.decoupled_dim
                up_blocks = []
                if cond_embed is not None:
                    if modulation == "dit":
                        ModulatedSwinLayer = DiTSwinLayer
                    if modulation == "film":
                        ModulatedSwinLayer = FilmSwinLayer
                    ModulatedSwinLayer = partial(ModulatedSwinLayer, cond_dim=cond_embed.cond_dim)
                    LocalLayer = ModulatedSwinLayer
                else:
                    LocalLayer = SwinLayer
                LocalLayer = partial(
                    LocalLayer,
                    use_rpb=use_rpb,
                    use_rope=use_rope,
                )
                
                # self.flux_middle_first = LocalLayer(
                #     flux_space,
                #     self.df_unet.down_dims[-1],
                #     grid_size=middle_grid_size,
                #     window_size=df_window_size[space_diff:],
                #     depth=2,
                #     num_heads=8,
                #     drop_path=drop_path,
                #     mlp_ratio=8.0,
                #     use_checkpoint=use_checkpoint,
                #     norm_layer=nn.LayerNorm,
                #     act_fn=act_fn,
                # )

                self.flux_middle_two = LocalLayer(
                    flux_space,
                    self.df_unet.down_dims[-1],
                    grid_size=middle_grid_size,
                    window_size=df_window_size[space_diff:],
                    depth=2,
                    num_heads=8,
                    drop_path=drop_path,
                    mlp_ratio=8.0,
                    use_checkpoint=use_checkpoint,
                    norm_layer=nn.LayerNorm,
                    act_fn=act_fn,
                )

                self.flux_middle_upscale = PatchUnmerging(
                    space=flux_space,
                    dim=self.df_unet.down_dims[-1],
                    grid_size=middle_grid_size,
                    target_grid_size=up_grid_sizes[0],
                    c_multiplier=c_multiplier,
                    use_conv=False,
                    mlp_depth=1,
                )

                for i in range(num_layers - 1):
                    up_blocks.append(
                        SwinBlockUp(
                            flux_space,
                            up_dims[i],
                            grid_size=up_grid_sizes[i],
                            target_grid_size=up_grid_sizes[i + 1],
                            window_size=df_window_size[space_diff:],
                            num_heads=up_num_heads[i],
                            depth=up_depth[i],
                            use_abs_pe=use_abs_pe,
                            drop_path=drop_path,
                            hidden_mlp_ratio=8.0,
                            c_multiplier=c_multiplier,
                            use_checkpoint=use_checkpoint,
                            act_fn=act_fn,
                            norm_layer=nn.LayerNorm,
                            LayerType=LocalLayer,
                            conv_upsample=False,
                        )
                    )
                # last up block (no upsample)
                up_blocks.append(
                    SwinBlockUp(
                        flux_space,
                        up_dims[-1],
                        grid_size=up_grid_sizes[-1],
                        window_size=df_window_size[space_diff:],
                        num_heads=up_num_heads[-1],
                        depth=up_depth[-1],
                        use_abs_pe=use_abs_pe,
                        drop_path=drop_path,
                        hidden_mlp_ratio=8.0,
                        use_checkpoint=use_checkpoint,
                        act_fn=act_fn,
                        norm_layer=nn.LayerNorm,
                        LayerType=LocalLayer,
                        mode=LayerModes.SEQUENCE,
                    )
                )
                self.flux_up_blocks = nn.ModuleList(up_blocks)

                # unpatch
                self.flux_unpatch = PatchUnmerging(
                    flux_space,
                    up_dims[-1],
                    grid_size=up_grid_sizes[-1],
                    expand_by=df_patch_size[space_diff:],
                    out_channels=out_channels,
                    flatten=False,
                    use_conv=False,
                    norm_layer=None,
                    mlp_ratio=unmerging_hidden_ratio,
                    act_fn=nn.LeakyReLU,
                    patch_skip=patch_skip,
                    cond_dim=flux_cond_embed.cond_dim if flux_cond_embed else None,
                )

                # if self.patch_skip:
                #     self.flux_patch_skip_mix =  MixingBlock(
                #         self.df_unet.unpatch.dim,
                #         self.phi_unet.unpatch.dim,
                #         num_heads=8,
                #         attn_drop=0.1,
                #         init_weights=init_weights
                #     )

        if latent_cross_attn:
            # down/middle directio
            df_mix = []
            phi_mix = []
            flux_down_mix = []
            phi_down_dims = self.phi_unet.down_dims
            df_down_dims = self.df_unet.down_dims
            for df_dim, phi_dim in zip(df_down_dims, phi_down_dims):
                df_mix.append(MixingBlock(df_dim, phi_dim, num_heads=8, attn_drop=0.1,
                                          init_weights=init_weights))
                phi_mix.append(MixingBlock(phi_dim, df_dim, num_heads=8, attn_drop=0.1,
                                           init_weights=init_weights))
                if self.decode_fluxfield:
                    flux_down_mix.append(MixingBlock(df_dim, phi_dim,
                                         num_heads=8, attn_drop=0.1, 
                                         init_weights=init_weights))
            self.df_mix = nn.ModuleList(df_mix)
            self.phi_mix = nn.ModuleList(phi_mix)
            self.flux_down_mix = nn.ModuleList(flux_down_mix)

            # up direction
            df_mix_up = []
            phi_mix_up = []
            flux_mix_up = []
            for i, (df_blk, phi_blk) in enumerate(zip(self.df_up_blocks, self.phi_up_blocks)):
                df_mix_up.append(
                    MixingBlock(
                        left_dim=df_blk.dim,
                        right_dim=phi_blk.dim,
                        num_heads=8,
                        attn_drop=0.1,
                        init_weights=init_weights,
                    )
                )
                phi_mix_up.append(
                    MixingBlock(
                        left_dim=phi_blk.dim,
                        right_dim=df_blk.dim,
                        num_heads=8,
                        attn_drop=0.1,
                        init_weights=init_weights,
                    )
                )
                if self.decode_fluxfield:
                    flux_mix_up.append(
                        BidirectionalMixingBlock(
                            left_dim=self.flux_up_blocks[i].dim,
                            middle_dim=df_blk.dim,
                            right_dim=phi_blk.dim,
                            num_heads=8,
                            attn_drop=0.1,
                            init_weights=init_weights,
                        )
                    )
            self.df_mix_up = nn.ModuleList(df_mix_up)
            self.phi_mix_up = nn.ModuleList(phi_mix_up)
            self.flux_mix_up = nn.ModuleList(flux_mix_up)

            df_patch_dim = self.df_unet.unpatch.dim * (2 if patch_skip else 1)
            phi_patch_dim = self.phi_unet.unpatch.dim * (2 if patch_skip else 1)
            flux_patch_dim = self.flux_unpatch.dim * (2 if patch_skip else 1)
            self.df_mix_unpatch = MixingBlock(
                left_dim=df_patch_dim, right_dim=phi_patch_dim, num_heads=8, attn_drop=0.1,
                init_weights=init_weights
            )
            self.phi_mix_unpatch = MixingBlock(
                left_dim=phi_patch_dim, right_dim=df_patch_dim, num_heads=8, attn_drop=0.1,
                init_weights=init_weights
            )
            if self.decode_fluxfield:
                self.flux_mix_unpatch = BidirectionalMixingBlock(
                    left_dim=flux_patch_dim,
                    middle_dim=df_patch_dim,
                    right_dim=phi_patch_dim, 
                    num_heads=8, 
                    attn_drop=0.1,
                    init_weights=init_weights
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
            flux_lats.append(self.flux_head.mix(0, phi, df, kwargs))

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
                flux_lats.append(self.flux_head.mix(i + 1, phi_, df_, kwargs))

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
        zflux: torch.Tensor,
        df_pad_axes: Sequence,
        phi_pad_axes: Sequence,
        use_phi: bool = True,
        df_cond: Optional[torch.Tensor] = None,
        phi_cond: Optional[torch.Tensor] = None,
        flux_cond: Optional[torch.Tensor] = None,
    ):
        # patch-space mixing
        if hasattr(self, "df_mix_unpatch"):
            # final mixing
            zdf = self.df_mix_unpatch(zdf, zphi)
            if use_phi:
                zphi = self.phi_mix_unpatch(zphi, zdf)
        if hasattr(self, "flux_mix_unpatch"):
            zflux = self.flux_mix_unpatch(zflux, zdf, zphi)
        # expand to original
        df = self.df_unet.patch_decode(zdf, df_pad_axes, cond=df_cond)
        if use_phi:
            phi = self.phi_unet.patch_decode(zphi, phi_pad_axes, cond=phi_cond)
        else:
            phi=None
        if self.decode_fluxfield:
            flux = self.flux_unpatch(zflux, cond=flux_cond)
            flux = unpad(flux, df_pad_axes, self.df_unet.base_resolution)
            flux = rearrange(flux, "b ... c -> b c ...")
            # if self.decouple_mu:
            #     flux = rearrange(
            #         flux, "b (c mu) vp ... -> b c vp mu ...", mu=self.df_unet.decoupled_dim
            #     )
        else:
            flux = None
        return df, phi, flux

    def condition(self, kwconds: Dict[str, torch.Tensor]) -> Dict:
        # drop input fields
        kwconds = {k: v for k, v in kwconds.items() if k in self.condition_keys}
        if len(kwconds) == 0:
            return {}

        assert self.condition_keys == sorted(list(kwconds.keys())), (
            "Mismatch in conditioning keys "
            f"{self.condition_keys} != {sorted(list(kwconds.keys()))}"
        )
        cond = torch.cat(
            [kwconds[k].unsqueeze(-1) for k in self.condition_keys], dim=-1
        )
        if self.flux_cond_embed is not None:
            # embed conditioning is e.g. sincos
            return {"condition": self.flux_cond_embed(cond)}
        else:
            return {}

class NeuGKMultitask(NeuGK):
    """Neural Gyrokinetics model for multitask predictions from the 5d density."""

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
            flux_attn_down = []
            for i, (df_blk, phi_blk) in enumerate(zip(self.df_down_blocks, self.phi_up_blocks[::-1])):
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

                if self.decode_fluxfield:
                    flux_attn_down.append(
                            VSpaceReduce(
                            dim=df_blk.dim,
                            out_dim=self.flux_up_blocks[::-1][i].dim,
                            num_heads=8,
                            attn_drop=0.1,
                            decouple_mu=self.df_unet.decouple_mu,
                            init_weights="xavier_uniform",
                        )
                    )

            self.vspace_attn_down = nn.ModuleList(vspace_attn_down)
            if self.decode_fluxfield:
                self.flux_attn_down = nn.ModuleList(flux_attn_down)

        self.vspace_attn_middle = VSpaceReduce(
            dim=self.df_unet.middle.dim,
            out_dim=self.phi_middle.dim,
            num_heads=8,
            attn_drop=0.1,
            decouple_mu=self.df_unet.decouple_mu,
            init_weights="xavier_uniform",
        )

        if self.decode_fluxfield:
            self.flux_vspace_attn_middle = VSpaceReduce(
                dim=self.df_unet.middle.dim,
                out_dim=self.flux_middle_two.dim,
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

            if self.decode_fluxfield:
                self.flux_vspace_attn_patch_skip = VSpaceReduce(
                    dim=self.flux_unpatch.dim,
                    out_dim=self.latent_dim,
                    num_heads=8,
                    attn_drop=0.1,
                    decouple_mu=self.df_unet.decouple_mu,
                    init_weights="xavier_uniform",
                )

    def forward(self, df: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        df, df_pad_axes = self.df_unet.patch_encode(df)
        phi_pad_axes = df_pad_axes[:6]
        flux = None
        flux0 = None

        if self.patch_skip:
            df0 = df.clone()
            phi0 = self.vspace_attn_patch_skip(df0)
            if self.decode_fluxfield and hasattr(self, "flux_vspace_attn_patch_skip"):
                # TODO: add VSPaceReduce for fluxfield with integrate_s=True
                flux0 = self.flux_vspace_attn_patch_skip(df0, integrate_s=True)
                # flux0 = self.flux_patch_skip_mix(df0.sum((1,2)), phi0)

        # parameter conditioning
        # TODO why have two?
        df_cond = self.df_unet.condition(kwargs)
        phi_cond = self.phi_unet.condition(kwargs)
        if self.decode_fluxfield:
            flux_cond = self.condition(kwargs)

        # down paths
        df_features = []
        phi_features = []
        flux_features = []
        phi = None
        for i, df_blk in enumerate(self.df_down_blocks):
            # down blocks
            df, df_pre = df_blk(df, **df_cond)
            df_features.append(df_pre)
            if self.use_phi:
                # integrate out velocity space
                phi = self.vspace_attn_down[i](df_pre)
                phi_features.append(phi)
            else:
                phi_features.append(None)
            if self.decode_fluxfield and phi is not None and hasattr(self, "flux_down_mix"):
                # mix df + phi
                flux = self.flux_attn_down[i](df_pre, integrate_s=True)
                flux_features.append(self.flux_down_mix[i](flux, phi))
            else:
                flux_features.append(None)

        # middle blocks + latent mixing
        flux_lats = []
        if hasattr(self.df_unet, "middle_pe"):
            df = self.df_unet.middle_pe(df)

        # integrate out velocity space
        phi = self.vspace_attn_middle(df)
        if hasattr(self.phi_unet, "middle_pe"):
            phi = self.phi_unet.middle_pe(phi)

        if self.decode_fluxfield:
            # TODO: add VSPaceReduce for fluxfield with integrate_s=True
            flux = self.flux_vspace_attn_middle(df, integrate_s=True)
            #flux = self.flux_middle_first(df.sum((1,2)), **flux_cond)

        df, phi = self.df_mix_middle(df, phi), self.phi_mix_middle(phi, df)
        if self.decode_fluxfield:
            flux = self.flux_mix(flux, df, phi)

        df = self.df_unet.middle(df, **df_cond)
        phi = self.phi_middle(phi, **phi_cond)

        if self.decode_fluxfield:
            flux = self.flux_middle_two(flux, **flux_cond)

        if self.flux_head is not None and not self.decode_fluxfield:
            flux_lats.append(self.flux_head.mix(0, phi, df, kwargs))

        df = self.df_unet.middle_upscale(df)
        phi = self.phi_middle_upscale(phi)
        if self.decode_fluxfield:
            flux = self.flux_middle_upscale(flux)

        # up path
        df_features = df_features[::-1]
        for i, (df_blk, df_mix) in enumerate(
            zip(self.df_up_blocks, self.df_mix_up)
        ):
            # mix latents
            df = df_mix(df, phi)
            phi = self.phi_mix_up[i](phi, df)
            if self.decode_fluxfield and hasattr(self, "flux_mix_up"):
                flux = self.flux_mix_up[i](flux, df, phi)
            # up blocks
            df, df_ = df_blk(df, s=df_features[i], return_skip=True, **df_cond)
            if self.use_phi:
                phi, phi_ = self.phi_up_blocks[i](phi, s=phi_features[i], return_skip=True, **phi_cond)
            else:
                phi_ = phi
            # multiscale flux latents
            if self.flux_head is not None and not self.decode_fluxfield:
                flux_lats.append(self.flux_head.mix(i + 1, phi_, df_, kwargs))
            if self.decode_fluxfield:
                flux = self.flux_up_blocks[i](flux, s=flux_features[i], return_skip=False, **flux_cond)
        
        # expand to original
        if self.patch_skip:
            df = torch.cat([df, df0], -1)
            phi = torch.cat([phi, phi0], -1)
            if self.decode_fluxfield and flux0 is not None:
                flux = torch.cat([flux, flux0], -1)

        df, phi, flux = self.patch_decode(
            df,
            phi,
            flux,
            df_pad_axes=df_pad_axes,
            phi_pad_axes=phi_pad_axes,
            use_phi=self.use_phi,
            df_cond=df_cond.get("condition"),
            phi_cond=phi_cond.get("condition"),
            flux_cond=flux_cond.get("condition")
        )

        out = [df]
        if self.use_phi:
            phi = rearrange(phi, "b c s x y -> b c x s y")
            out += [phi.squeeze()]

        if hasattr(self, "flux_head") and not self.decode_fluxfield:
            flux = self.flux_head(flux_lats)
        out += [flux]

        outputs = {}
        for key, pred in zip(self.outputs, out):            
            outputs[key] = pred
        return outputs
