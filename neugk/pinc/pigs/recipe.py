"""The PIGS method ladder (self-contained), all at a FIXED Gaussian budget, all returned fp16-rounded
(on-disk-fair) with timing in info:

  compress_base   : VANILLA Gaussian splatting (joint AdamW MSE minibatches; no warm-start, no separable
                    trick). The slow reference we started from.
  compress_fast   : fast density fit = amp warm-start -> separable warmup (Huber) -> fused polish.
  compress_gpinc  : fast + gPINC physics fine-tune on the SEPARABLE reconstruction (~10x faster than dense,
                    bit-identical losses). The recommended reconstruction model.
  compress_pinc   : fast + DENSE PINC physics fine-tune. Same losses/quality as gpinc -- kept ONLY as the
                    speed reference (shows what gPINC's separable trick buys).
  compress_pigs   : fast + gPINC + flux graft (Gabor carriers at the heat-flux tail, residual-placed,
                    complex amp warm-start) + frozen-base flux refine (POST). Budget-neutral:
                    n_total = n_base + n_flux. The flux-accurate model.

Density-fit kwargs (warmup_steps, polish_epochs, loss, ...) pass through to train_fast for ablations.
"""
import time

from neugk.pinc.pigs.model import from_gaussian, quantize_, BYTES_PER_PARAM as _BPP
from neugk.pinc.pigs.train import train_default, train_fast, train_pinc, refine_flux
from neugk.pinc.pigs.flux import residual_centers, add_atoms, TAIL_BINS
from neugk.pinc.pigs.fast import solve_new_amps_complex


def model_bytes(n_base, n_flux=0, quant="fp16"):
    """base Gaussian = 16 params, flux Gabor atom = 17 (the +1 carrier ky)."""
    return int((n_base * 16 + n_flux * 17) * _BPP[quant])


def compression_ratio(data, n_base, n_flux=0, quant="fp16"):
    return data.full_df.nbytes / model_bytes(n_base, n_flux, quant)


def n_for_cr(data, target_cr, quant="fp16"):
    """#Gaussians for a target compression ratio (file-size control up front)."""
    return max(1, int(data.full_df.nbytes / (target_cr * 16 * _BPP[quant])))


def tied_model_bytes(n_base, m_env, k, quant="fp16"):
    """tied carrier group = 1 envelope (mu 5 + L_phys 6 + L_vel 3 = 14) + k carriers (ky, Re, Im = 3k)."""
    return int((n_base * 16 + m_env * (14 + 3 * k)) * _BPP[quant])


def tied_centers(data, m_env, device, placement="stratified", seed=2):
    """Envelope centers for the tied graft. MEASURED: COVERAGE placement beats residual hotspots ~6x on
    flux AND +4-6 dB phi at identical bytes (residual clusters carriers in high-energy phi-sensitive
    regions; uniform coverage gives POST independent flux DOF across all (s,x)). 'stratified' (one voxel
    per equal grid chunk) = production default; 'random' ~equal flux, ~-1 dB phi."""
    import torch
    grid = data.grid.reshape(-1, 5).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    if placement == "stratified":
        chunk = grid.shape[0] // m_env
        offs = torch.randint(0, chunk, (m_env,), generator=g, device=device)
        return grid[torch.arange(m_env, device=device) * chunk + offs].float()
    if placement == "random":
        return grid[torch.randperm(grid.shape[0], generator=g, device=device)[:m_env]].float()
    raise ValueError(placement)


def _reset_data(data, seed):
    """Free the CUDA cache (running several compress_* in sequence accumulates cached blocks; the dense
    PINC forward peaks near the full GPU, so start clean). With seed: full determinism -- the polish
    loader's dataset.shuffle() permutes f_df/f_grid IN PLACE with the global RNG, so runs are only
    reproducible after resetting that hysteresis AND seeding the RNG."""
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()
    if seed is not None:
        from einops import rearrange
        data.f_df = rearrange(data.df, "c ... -> c (...)").to(data.f_df.device, data.f_df.dtype)
        data.f_grid = rearrange(data.grid, "... d -> (...) d").to(data.f_grid.device, data.f_grid.dtype)
        torch.manual_seed(seed)


def _finish(m, data, n_base, n_flux, quant, t0, extra=None, verbose=True):
    quantize_(m, quant)
    nbytes = model_bytes(n_base, n_flux, quant)
    info = {"n": n_base + n_flux, "quant": quant, "bytes": nbytes,
            "CR": round(data.full_df.nbytes / nbytes, 1), "time_s": round(time.time() - t0, 1),
            **(extra or {})}
    if verbose:
        print("done:", info, flush=True)
    return m, info


def compress_base(data, n, device="cuda", quant="fp16", epochs=15, seed=None, verbose=True):
    """VANILLA GS baseline (train_default): slow joint-AdamW MSE; no fast tricks, no physics.
    epochs=15 matches the documented 'default GS (vanilla)' table row (~90 s at N~1200, 32.1 dB)."""
    _reset_data(data, seed)
    t0 = time.time()
    m, _ = train_default(data, n, device, epochs=epochs)
    if verbose:
        print(f"[1/1] vanilla density  N={n}  ({time.time()-t0:.0f}s)", flush=True)
    return _finish(m, data, n, 0, quant, t0, verbose=verbose)


def compress_fast(data, n, device="cuda", quant="fp16", seed=None, verbose=True, **fit_kw):
    """FAST density (train_fast): amp warm-start -> separable warmup -> fused polish. No physics.
    fit_kw -> train_fast (e.g. warmup_steps=, polish_epochs=, loss=)."""
    _reset_data(data, seed)
    t0 = time.time()
    m, _ = train_fast(data, n, device, **fit_kw)
    if verbose:
        print(f"[1/1] fast density  N={n}  ({time.time()-t0:.0f}s)", flush=True)
    return _finish(m, data, n, 0, quant, t0, verbose=verbose)


def _fast_plus_pinc(data, n, device, mode, pinc_epochs, seed, verbose, fit_kw):
    _reset_data(data, seed)
    t0 = time.time()
    md, _ = train_fast(data, n, device, **fit_kw)
    if verbose:
        print(f"[1/2] fast density  N={n}  ({time.time()-t0:.0f}s)", flush=True)
    m, _ = train_pinc(md, data, device, epochs=pinc_epochs, mode=mode)
    if verbose:
        print(f"[2/2] {'gPINC (separable)' if mode == 'gpinc' else 'PINC (dense)'}  ({time.time()-t0:.0f}s)", flush=True)
    return m, t0


def compress_gpinc(data, n, device="cuda", quant="fp16", pinc_epochs=40, seed=None, verbose=True, **fit_kw):
    """fast + gPINC: physics fine-tune on the SEPARABLE reconstruction (recommended; ~10x faster than dense)."""
    m, t0 = _fast_plus_pinc(data, n, device, "gpinc", pinc_epochs, seed, verbose, fit_kw)
    return _finish(m, data, n, 0, quant, t0, verbose=verbose)


def compress_pinc(data, n, device="cuda", quant="fp16", pinc_epochs=40, seed=None, verbose=True, **fit_kw):
    """fast + DENSE PINC: same losses/quality as compress_gpinc -- the SPEED REFERENCE for the separable trick."""
    m, t0 = _fast_plus_pinc(data, n, device, "dense", pinc_epochs, seed, verbose, fit_kw)
    return _finish(m, data, n, 0, quant, t0, verbose=verbose)


def compress_pigs(data, n_total=1000, flux_frac=0.25, n_flux=None, device="cuda", quant="fp16",
                  flux_mode="raw", flux_lambda=3.0, pinc_epochs=40, post_steps=500,
                  tied=True, tied_k=9, placement="stratified", seed=None, verbose=True,
                  patience=25, **fit_kw):
    """Full PIGS: fast density -> gPINC -> graft flux carriers -> frozen-base flux refine (POST).
    Budget-neutral: n_total = n_base + n_flux atom-equivalents (round(flux_frac*n_total) unless n_flux given).

    tied=True (PRODUCTION default, measured ~20x better flux than free carriers): the flux budget is spent
    on TIED CARRIER GROUPS -- m_env envelopes (stratified-coverage placed, narrow sigma_y=1.5), each
    carrying tied_k carriers at the tail bins with SHARED mu/Sigma (14+3k params vs 17k -> ~3.7x carriers
    per byte). POST trains amps ONLY, so the tying and on-bin ky stay exact (stored-bytes accounting real).
    tied=False: the older free-carrier graft (one ky+amp per atom, residual-placed, broad sigma_y).

    flux_mode: 'raw' (min total flux, ~3 dB phi cost) | 'wnorm' (phi-preserving, fixes only the tail).
    flux_lambda sweeps the phi-vs-flux Pareto. With seed=0 + tied: ~f 34.5 / phi ~16 / flux ~0.003-0.01."""
    import torch
    _reset_data(data, seed)
    n_flux = n_flux if n_flux is not None else max(1, round(flux_frac * n_total))
    n_base = n_total - n_flux
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    md, _ = train_fast(data, n_base, device, patience=patience, **fit_kw)         # 1. fast density
    log(f"[1/4] fast density  N_base={n_base}  ({time.time()-t0:.0f}s)")
    mg, _ = train_pinc(md, data, device, epochs=pinc_epochs, mode="gpinc")        # 2. gPINC physics
    log(f"[2/4] gPINC  ({time.time()-t0:.0f}s)")
    gb = from_gaussian(mg)
    if tied:                                                                      # 3. graft TIED groups
        m_env = max(1, round(n_flux * 17 / (14 + 3 * tied_k)))
        n_carr = m_env * tied_k
        centers = (residual_centers(gb, data, device, m_env) if placement == "residual"
                   else tied_centers(data, m_env, device, placement=placement))
        shared = centers.repeat_interleave(tied_k, dim=0)
        m, mask = add_atoms(gb, data, device, n_carr, TAIL_BINS, sigma_y_cells=1.5, shared_mu=shared)
        with torch.no_grad():                                                     #    tie Sigma per group
            for t_ in ("L_phys_raw", "L_vel_raw"):
                blk = getattr(m, t_).data[gb.N:]
                blk.copy_(blk.reshape(m_env, tied_k, -1)[:, :1]
                          .expand(m_env, tied_k, blk.shape[-1]).reshape(n_carr, -1))
        solve_new_amps_complex(m, mask, data, device)
        log(f"[3/4] graft  +{n_carr} TIED carriers ({m_env} envelopes x {tied_k})  ({time.time()-t0:.0f}s)")
        m = refine_flux(m, data, device, mask=mask, train=("amps",),              # 4. POST (amps only ->
                        flux_mode=flux_mode, flux_lambda=flux_lambda, steps=post_steps, patience=patience)  # tying stays exact)
        nbytes = tied_model_bytes(n_base, m_env, tied_k, quant)
        extra = {"n_base": n_base, "n_flux": n_carr, "tied": True, "m_env": m_env, "tied_k": tied_k,
                 "placement": placement, "flux_mode": flux_mode, "flux_lambda": flux_lambda}
    else:                                                                         # 3. graft FREE carriers
        centers = residual_centers(gb, data, device, n_flux)
        m, mask = add_atoms(gb, data, device, n_flux, TAIL_BINS, sigma_y_cells=8.0, shared_mu=centers)
        solve_new_amps_complex(m, mask, data, device)
        log(f"[3/4] graft  +{n_flux} free flux atoms  ({time.time()-t0:.0f}s)")
        m = refine_flux(m, data, device, mask=mask, flux_mode=flux_mode,
                        flux_lambda=flux_lambda, steps=post_steps, patience=patience)
        nbytes = model_bytes(n_base, n_flux, quant)
        extra = {"n_base": n_base, "n_flux": n_flux, "tied": False,
                 "flux_mode": flux_mode, "flux_lambda": flux_lambda}
    log(f"[4/4] POST {flux_mode}  ({time.time()-t0:.0f}s)")
    quantize_(m, quant)
    info = {"n": n_total, "quant": quant, "bytes": nbytes,
            "CR": round(data.full_df.nbytes / nbytes, 1), "time_s": round(time.time() - t0, 1), **extra}
    if verbose:
        log("done:", info)
    return m, info
