"""Verify the FIXED pigs ladder: 5 distinct compress functions, base!=fast (vanilla is genuinely the slow
baseline), gPINC vs dense PINC (same losses, different speed), PIGS flux graft works, and the warmup
ablation ACTUALLY varies (kwargs passthrough). Small/fast settings -- trends matter, not absolute numbers."""
import sys; sys.path.insert(0, "/home/u6eb/gutenbru.u6eb/plasmamodelling")
for _m in list(sys.modules):
    if _m.startswith("neugk.pinc."): del sys.modules[_m]
import torch
import neugk.pinc.pigs as pigs
from neugk.pinc.neural_fields import CycloneNFDataset

dev = "cuda"
data = CycloneNFDataset("iteration_13", timesteps=100, path="/projects/u6eb/gyrokinetics/preprocessed_kvikio",
                        backend="kvikio", realpotens=True, normalize="zscore", normalize_coords=True)
data.to(torch.device(dev))

def row(tag, m, info):
    e = pigs.evaluate(m, data, dev)
    print(f"{tag:22s} t={info['time_s']:>6.1f}s CR={info['CR']:>7.1f} f={e['PSNR(f)']:6.2f} "
          f"phi={e['PSNR(phi)']:6.2f} flux_all={e['flux_relL1']:.3f}", flush=True)
    return e

N = 500
print("=== LADDER (N=%d, small settings) ===" % N, flush=True)
e_base = row("base (vanilla GS)", *pigs.compress_base(data, N, dev, epochs=6, verbose=False))
e_fast = row("fast (density)", *pigs.compress_fast(data, N, dev, verbose=False))
e_gp   = row("gpinc (fast+sep)", *pigs.compress_gpinc(data, N, dev, verbose=False))         # full 40 epochs
e_pn   = row("pinc (fast+dense)", *pigs.compress_pinc(data, N, dev, pinc_epochs=4, verbose=False))
e_pg   = row("pigs (full)", *pigs.compress_pigs(data, n_total=N, flux_frac=0.3, post_steps=300,
                                                 verbose=False))                            # full 40 epochs
assert e_fast["PSNR(f)"] > e_base["PSNR(f)"] + 0.5, "fast should beat short vanilla baseline"
assert e_gp["PSNR(phi)"] > e_fast["PSNR(phi)"] + 5, "gPINC must fix phi vs density-only"
assert e_pg["flux_relL1"] < e_gp["flux_relL1"] - 0.05, "PIGS must improve flux over gpinc"
print("ladder assertions PASSED", flush=True)

print("=== WARMUP ABLATION reality check (must differ in time AND quality) ===", flush=True)
for s in (100, 500, 1000):
    m, info = pigs.compress_fast(data, N, dev, warmup_steps=s, verbose=False)
    e = pigs.evaluate(m, data, dev)
    print(f"warmup={s:5d}: t={info['time_s']:5.1f}s f={e['PSNR(f)']:.2f}", flush=True)

print("=== n_for_cr sanity ===", flush=True)
for cr in (100, 1000, 2000):
    print(f"CR={cr}: n={pigs.n_for_cr(data, cr)}", flush=True)
print("LADDER_VERIFY_DONE", flush=True)
