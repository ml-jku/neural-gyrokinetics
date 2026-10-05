"""Conditioned GyroSwin flux decoder: torch module translated to JAX, forward parity."""

from __future__ import annotations

import jax.random as jr


def test_conditioned_flux_decoder_forward_parity():
    import jax.numpy as jnp
    import numpy as np
    import torch
    from neugk.gyroswin.models.x_layers import FluxDecoder as TorchFluxDecoder

    from neugk_jax.gyroswin.models.x_layers import FluxDecoder
    from neugk_jax.translate import translate_gyroswin

    torch.manual_seed(0)
    left_dims, right_dims, n_cond = [32, 16], [64, 32], 3
    tmod = TorchFluxDecoder(
        left_dims,
        right_dims,
        num_heads=4,
        depth=1,
        n_cond=n_cond,
        cond_embed_dim=128,
        drop=0.1,
        attn_drop=0.1,
    ).eval()
    sd = {k: v.detach().numpy() for k, v in tmod.state_dict().items()}
    jmod = FluxDecoder(left_dims, right_dims, 4, 1, key=jr.PRNGKey(0), n_cond=n_cond)
    jmod, missing, unused = translate_gyroswin(jmod, sd)
    assert not missing and all(k.startswith("cond_embed.") for k in unused), (missing, unused)

    rng = np.random.default_rng(0)
    grids = [(2, 3, 2), (4, 6, 4)]
    cond = rng.standard_normal(n_cond).astype(np.float32)
    lefts = [rng.standard_normal((*g, d)).astype(np.float32) for g, d in zip(grids, left_dims)]
    rights = [rng.standard_normal((*g, d)).astype(np.float32) for g, d in zip(grids, right_dims)]
    with torch.no_grad():
        tc = torch.from_numpy(cond)[None]
        tl = [
            tmod.mix(i, torch.from_numpy(a)[None], torch.from_numpy(b)[None], cond=tc)
            for i, (a, b) in enumerate(zip(lefts, rights))
        ]
        tflux = tmod(tl).numpy()
    jl = [
        jmod.mix(i, jnp.asarray(a), jnp.asarray(b), jnp.asarray(cond))
        for i, (a, b) in enumerate(zip(lefts, rights))
    ]
    jflux = np.asarray(jmod(jl))
    np.testing.assert_allclose(jflux, tflux, rtol=1e-4, atol=1e-5)
