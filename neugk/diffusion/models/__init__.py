from neugk.diffusion.models.dit import DiT
from neugk.models.layers import ContinuousConditionEmbed, IntegerConditionEmbed


def get_diffusion_model(cfg, autoencoder):
    diffuser_cfg = cfg.model

    # time conditioning
    diff_steps = cfg.model.scheduler.num_train_timesteps
    time_fn = IntegerConditionEmbed(128, 1, max_size=diff_steps, use_mlp=False)
    # parameter conditioning
    cond_fn = None
    n_cond = len(diffuser_cfg.conditioning)
    if n_cond > 0:
        cond_fn = ContinuousConditionEmbed(32, n_cond)

    if diffuser_cfg.model_type == "dit":
        model = DiT(
            space=5 - int(autoencoder.decouple_mu),
            dim=diffuser_cfg.latent_dim,
            z_dim=autoencoder.bottleneck_dim,
            base_resolution=autoencoder.bottleneck_grid_size,
            patch_size=None,
            depth=diffuser_cfg.vit.depth,
            num_heads=diffuser_cfg.vit.num_heads,
            use_checkpoint=diffuser_cfg.vit.gradient_checkpoint,
            drop_path=diffuser_cfg.vit.drop_path,
            hidden_mlp_ratio=2.0,
            time_embed=time_fn,
            cond_embed=cond_fn,
        )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model
