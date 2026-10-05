"""Detect per-trajectory transition windows [t0, t1] from the spectral free-energy
proxy E(t) = sum_ky kyspec[t] (the GT k_y potential-energy spectrum, stored per
timestep in the trajectory metadata -- cheap, no df reads, no FFT).

Rationale: the heat flux Q lags the linear mode growth (Q is ~quadratic and only
rises once turbulence has developed), so a flux-based onset starts the window after
the energy cascade has already set up. The spectral energy rises *with* the linear
growth, so it brackets the cascade; the k_y spectral centroid (which marches to
lower modes during the inverse cascade) is kept as a validation overlay.

  t0 = onset of exponential growth (E first exceeds a small fraction of the saturated level)
  t1 = post-overshoot saturation (E settles back into the saturated band after its peak)

Outputs transition_windows.json (per traj: t0, t1, S, energies, centroids, flag).
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neugk.dataset.backend import KvikIOBackend

DATA = "/local00/bioinf/galletti/preprocessed_kvikio"


def _smooth(x, w=3):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def detect(energy, centroid, sat_tail=0.4, onset_frac=0.03, n_snap=10, dt_idx=2):
    win = (n_snap - 1) * dt_idx  # 10 snapshots at dt=2.4 R/Vr (every 2nd frame) -> 18 idx
    """t0 = spectral-energy onset; t1 = t0 + a short fixed window (~10 dts).

    The spectral energy gives a clean, robust *onset* (it rises with the linear mode
    growth, before the heat flux). The energy *peak/end* is unreliable (it plateaus and
    fluctuates in saturation), so we do not detect t1 -- we take a fixed window of
    ``win`` index steps (= win/2 sampling dts at dt = 2.4 R/Vr, default 20 idx ~ 24 R/Vr
    ~ 10 dts), which spans the growth/overshoot where the cascade is sharpest. The k_y
    centroid (reported) confirms the peak marches to lower modes across the window.
    """
    n = len(energy)
    e = _smooth(np.asarray(energy, float), 3)
    S = float(np.median(e[int((1 - sat_tail) * n) :]))
    if S <= 0:
        return 0, min(n - 1, win), S, "no_saturation"
    up = np.where(e > onset_frac * S)[0]  # first rise above the quiescent floor
    t0 = int(up[0]) if len(up) else 0
    t1 = min(t0 + win, n - 1)
    flag = "" if t1 - t0 >= win - 1 else "truncated"  # onset too late to fit the window
    return t0, t1, S, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset/pinc.yaml")
    ap.add_argument("--trajectories", nargs="*", default=None)
    ap.add_argument("--out", default="transition_windows.json")
    args = ap.parse_args()

    if args.trajectories:
        trajs = args.trajectories
    else:
        import yaml
        import re

        c = yaml.safe_load(open(args.config))
        trajs = sorted(
            {re.match(r"(iteration_\d+)", t).group(1) for t in c["test_trajectories"]},
            key=lambda s: int(s.split("_")[1]),
        )

    be = KvikIOBackend(rank=0, use_kvikio=False)
    out = {}
    for tr in trajs:
        meta = be.read_metadata(f"{DATA}/{tr}_ifft_realpotens", lightweight=True)
        ky = np.asarray(meta["kyspec"], float)  # (T, 32) GT k_y spectrum
        E = ky.sum(axis=1)  # spectral energy per timestep
        modes = np.arange(ky.shape[1])
        cen = (modes[None] * ky).sum(1) / (ky.sum(1) + 1e-12)
        t0, t1, S, flag = detect(E, cen)
        out[tr] = dict(
            t0=int(t0),
            t1=int(t1),
            n_timesteps=int(len(E)),
            S=float(S),
            E_t0=float(E[t0]),
            E_t1=float(E[t1]),
            E_peak=float(E.max()),
            cen_t0=float(cen[t0]),
            cen_t1=float(cen[t1]),
            cen_min=float(cen[t0 : t1 + 1].min()) if t1 > t0 else float(cen[t0]),
            dt=1.2,
            flag=flag,
        )
    json.dump(out, open(args.out, "w"), indent=2)
    lens = [v["t1"] - v["t0"] for v in out.values()]
    flagged = [k for k, v in out.items() if v["flag"]]
    print(
        f"wrote {args.out}: {len(out)} trajs | window len idx median={int(np.median(lens))} "
        f"min={min(lens)} max={max(lens)} | flagged={flagged or 'none'}"
    )
    # cascade sanity: centroid should drop from t0 to its min within the window
    drops = [out[k]["cen_t0"] - out[k]["cen_min"] for k in out]
    print(
        f"centroid drop t0->min within window: median={np.median(drops):.2f} "
        f"(positive = peak marches to lower modes)"
    )


if __name__ == "__main__":
    main()
