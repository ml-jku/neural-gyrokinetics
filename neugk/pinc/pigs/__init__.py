"""pigs — Physics-Inspired Gaussian Splats for 5D gyrokinetic compression.

The method LADDER (all at a fixed Gaussian budget, all returned fp16-rounded -> on-disk-fair):

    import neugk.pinc.pigs as pigs

    m, info = pigs.compress_base(data, n=1000)    # VANILLA GS (slow joint-AdamW MSE) -- reference baseline
    m, info = pigs.compress_fast(data, n=1000)    # fast density: amp warm-start + separable warmup + polish
    m, info = pigs.compress_gpinc(data, n=1000)   # fast + gPINC physics (separable, ~10x faster) -- recommended
    m, info = pigs.compress_pinc(data, n=1000)    # fast + DENSE PINC -- same quality, the speed reference
    m, info = pigs.compress_pigs(data, n_total=1000, flux_lambda=3.0)   # + flux graft + POST refine

    pigs.evaluate(m, data, "cuda")                # PSNR(f)/PSNR(phi)/kyspec/qspec + flux dom/tail/all
    n = pigs.n_for_cr(data, target_cr=1000)       # pick N for a target compression ratio

Modules: model (5D Gaussian/Gabor primitives + quantization), fast (separable contraction + warm-starts),
train (train_default / train_fast / train_pinc / refine_flux), flux (graft + evaluation), recipe (the ladder).
"""
from neugk.pinc.pigs.model import (
    GaussianSplat5D, GaborSplat5D, from_gaussian, build_gs_5d, quantize_, flat_to_lower_tri,
)
from neugk.pinc.pigs.fast import (
    solve_amplitudes, build_subgrids, sep_field, gabor_sep_field, gabor_denorm, reconstruct,
    ky_per_bin, solve_new_amps_complex,
)
from neugk.pinc.pigs.train import train_default, train_fast, train_pinc, refine_flux
from neugk.pinc.pigs.flux import residual_centers, add_atoms, evaluate, TAIL_BINS
from neugk.pinc.pigs.recipe import (
    compress_base, compress_fast, compress_gpinc, compress_pinc, compress_pigs,
    model_bytes, compression_ratio, n_for_cr, tied_model_bytes, tied_centers,
)

__all__ = [
    "compress_base", "compress_fast", "compress_gpinc", "compress_pinc", "compress_pigs",
    "evaluate", "reconstruct", "model_bytes", "compression_ratio", "n_for_cr", "quantize_", "TAIL_BINS",
    "GaussianSplat5D", "GaborSplat5D", "from_gaussian", "build_gs_5d", "flat_to_lower_tri",
    "solve_amplitudes", "build_subgrids", "sep_field", "gabor_sep_field", "gabor_denorm",
    "ky_per_bin", "solve_new_amps_complex", "residual_centers", "add_atoms",
    "train_default", "train_fast", "train_pinc", "refine_flux",
]
