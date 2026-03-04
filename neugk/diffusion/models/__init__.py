from typing import Optional, List

import torch
from torch import nn

from neugk.diffusion.models.dit import DiT
from neugk.diffusion.models.mlp import LatentMLP
from neugk.models.layers import ContinuousConditionEmbed, IntegerConditionEmbed


class DummyAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        _ = condition
        return {"df": df}

    def encode(self, df: torch.Tensor, condition: Optional[torch.Tensor] = None):
        _ = condition
        return df, None, None

    def decode(
        self,
        zdf: torch.Tensor,
        pad_axes: List,
        condition: Optional[torch.Tensor] = None,
    ):
        _ = pad_axes, condition
        return {"df": zdf}


def get_diffusion_model(cfg, autoencoder):
    diff_cfg = cfg.model

    # time conditioning
    diffusion_settings = diff_cfg.get("diffusion", {})
    if not diffusion_settings:
        diffusion_settings = diff_cfg  # fallback

    scheduler_cfg = diffusion_settings.get("scheduler", {})
    diff_steps = scheduler_cfg.get("num_train_timesteps", 1000)

    if diffusion_settings.get("continuous_time", True):
        time_fn = ContinuousConditionEmbed(32, 1)
    else:
        time_fn = IntegerConditionEmbed(128, 1, max_size=diff_steps + 1, use_mlp=False)
    # parameter conditioning
    cond_fn = None
    n_cond = len(diff_cfg.conditioning)
    if n_cond > 0:
        cond_fn = ContinuousConditionEmbed(32, n_cond)

    if "dit" in diff_cfg.model_type:
        model = DiT(
            space=5 - int(autoencoder.decouple_mu),
            dim=diff_cfg.latent_dim,
            z_dim=autoencoder.bottleneck_dim,
            base_resolution=autoencoder.bottleneck_grid_size,
            patch_size=None,
            depth=diff_cfg.vit.depth,
            num_heads=diff_cfg.vit.num_heads,
            use_checkpoint=diff_cfg.vit.gradient_checkpoint,
            drop_path=diff_cfg.vit.drop_path,
            hidden_mlp_ratio=2.0,
            time_embed=time_fn,
            cond_embed=cond_fn,
            use_rope=diff_cfg.vit.use_rope,
            gated_attention=diff_cfg.vit.gated_attention,
        )

    if "mlp" in diff_cfg.model_type:
        model = LatentMLP(
            z_dim=autoencoder.bottleneck_dim,
            dim=diff_cfg.latent_dim,
            base_resolution=autoencoder.bottleneck_grid_size,
            time_embed=time_fn,
            cond_embed=cond_fn,
        )

    if "unet" in cfg.model.model_type:
        assert cfg.dataset.input_fields == [
            "df"
        ], "No more inputs than df supported for simple 5D-Swin"
        from neugk.diffusion.models.diff_unet import Swin5DDiffUnet

        patch_size = cfg.model.swin.patch_size
        window_size = cfg.model.swin.window_size
        base_resolution = (32, 8, 16, 85, 32)
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.merging_hidden_ratio
        unmerging_hidden_ratio = cfg.model.swin.unmerging_hidden_ratio
        c_multiplier = cfg.model.swin.c_multiplier
        norm_output = cfg.model.swin.norm_output
        use_abs_pe = cfg.model.swin.use_abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        modulation = cfg.model.swin.modulation
        swin_bottleneck = cfg.model.swin.swin_bottleneck
        use_rpb = cfg.model.swin.use_rpb
        use_rope = cfg.model.swin.use_rope
        decouple_mu = cfg.model.decouple_mu
        problem_dim = 2 + (2 * int(cfg.dataset.separate_zf))

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + tuple(base_resolution)

        model = Swin5DDiffUnet(
            dim=diff_cfg.latent_dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            window_size=window_size,
            depth=depth,
            num_heads=num_heads,
            in_channels=problem_dim,
            out_channels=problem_dim,
            num_layers=num_layers,
            use_checkpoint=gradient_checkpoint,
            drop_path=0.1,
            use_abs_pe=use_abs_pe,
            conv_patch=False,
            hidden_mlp_ratio=2.0,
            c_multiplier=c_multiplier,
            merging_hidden_ratio=patching_hidden_ratio,
            unmerging_hidden_ratio=unmerging_hidden_ratio,
            merging_depth=1,
            unmerging_depth=1,
            time_embed=time_fn,
            cond_embed=cond_fn,
            norm_output=norm_output,
            act_fn=act_fn,
            patch_skip=False,
            modulation=modulation,
            swin_bottleneck=swin_bottleneck,
            decouple_mu=decouple_mu,
            use_rpb=use_rpb,
            use_rope=use_rope,
            conditioning={},
        )

    if "phi" in cfg.model.model_type:
        from neugk.diffusion.models.diff_unet import Swin3DDiffUnet

        patch_size = [
            cfg.model.swin.patch_size[3],
            cfg.model.swin.patch_size[2],
            cfg.model.swin.patch_size[4],
        ]
        window_size = [
            cfg.model.swin.window_size[3],
            cfg.model.swin.window_size[2],
            cfg.model.swin.window_size[4],
        ]
        base_resolution = (85, 16, 32)
        num_heads = cfg.model.swin.num_heads
        depth = cfg.model.swin.depth
        num_layers = cfg.model.num_layers
        gradient_checkpoint = cfg.model.swin.gradient_checkpoint
        patching_hidden_ratio = cfg.model.swin.merging_hidden_ratio
        unmerging_hidden_ratio = cfg.model.swin.unmerging_hidden_ratio
        c_multiplier = cfg.model.swin.c_multiplier
        norm_output = cfg.model.swin.norm_output
        use_abs_pe = cfg.model.swin.use_abs_pe
        act_fn = getattr(torch.nn, cfg.model.swin.act_fn)
        modulation = cfg.model.swin.modulation
        swin_bottleneck = cfg.model.swin.swin_bottleneck
        use_rpb = cfg.model.swin.use_rpb
        use_rope = cfg.model.swin.use_rope
        problem_dim = 1

        bundle_steps = cfg.model.bundle_seq_length
        if bundle_steps > 1:
            # extend patching for time dimension
            patch_size = [1] + patch_size
            window_size = [bundle_steps] + window_size
            base_resolution = (bundle_steps,) + tuple(base_resolution)

        model = Swin3DDiffUnet(
            #     dim=diff_cfg.latent_dim,
            #     base_resolution=base_resolution,
            #     patch_size=patch_size,
            #     window_size=window_size,
            #     depth=depth,
            #     num_heads=num_heads,
            #     in_channels=problem_dim,
            #     out_channels=problem_dim,
            #     num_layers=num_layers,
            #     use_checkpoint=gradient_checkpoint,
            #     drop_path=0.1,
            #     use_abs_pe=use_abs_pe,
            #     conv_patch=False,
            #     hidden_mlp_ratio=2.0,
            #     c_multiplier=c_multiplier,
            #     merging_hidden_ratio=patching_hidden_ratio,
            #     unmerging_hidden_ratio=unmerging_hidden_ratio,
            #     merging_depth=1,
            #     unmerging_depth=1,
            #     time_embed=time_fn,
            #     cond_embed=cond_fn,
            #     norm_output=norm_output,
            #     act_fn=act_fn,
            #     patch_skip=patch_skip,
            #     modulation=modulation,
            #     swin_bottleneck=swin_bottleneck,
            #     use_rpb=use_rpb,
            #     use_rope=use_rope,
            #     conditioning={},
            # )
            dim=diff_cfg.latent_dim,
            base_resolution=base_resolution,
            max_tsteps=diff_steps + 1,
            in_channels=problem_dim,
            out_channels=problem_dim,
        )

    if "fno" in cfg.model.model_type:
        assert cfg.dataset.input_fields == [
            "df"
        ], "No more inputs than df supported for simple 5D-Swin"
        from neugk.diffusion.models.diff_unet import FNO5DDiff

        patch_size = cfg.model.swin.patch_size
        base_resolution = (32, 8, 16, 85, 32)
        problem_dim = 2 + (2 * int(cfg.dataset.separate_zf))

        model = FNO5DDiff(
            dim=diff_cfg.latent_dim,
            base_resolution=base_resolution,
            max_tsteps=diff_steps + 1,
            in_channels=problem_dim,
            out_channels=problem_dim,
        )

    if "basic" in cfg.model.model_type:
        assert cfg.dataset.input_fields == [
            "df"
        ], "No more inputs than df supported for simple 5D-Swin"
        from neugk.diffusion.models.diff_unet import Basic5DDiff

        patch_size = cfg.model.swin.patch_size
        base_resolution = (32, 8, 16, 85, 32)
        problem_dim = 2 + (2 * int(cfg.dataset.separate_zf))

        model = Basic5DDiff(
            dim=diff_cfg.latent_dim,
            base_resolution=base_resolution,
            patch_size=patch_size,
            max_tsteps=diff_steps + 1,
            in_channels=problem_dim,
            out_channels=problem_dim,
        )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model
