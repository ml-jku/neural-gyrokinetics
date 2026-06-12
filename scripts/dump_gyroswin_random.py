"""Dump a randomly-initialised torch GyroSwinMultitask state_dict to /tmp/.

Used as a translation target while we don't have a trained GyroSwin
checkpoint — instantiates the torch model with the production config,
saves ``state_dict()``, and writes ``/tmp/gyroswin_random.pth``. The
JAX-side ``translate_gyroswin`` then exercises the rename table on real
shapes without needing actual training.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf


def _stub_dataset(active_keys, resolution=(32, 8, 16, 85, 32)):
    """Minimal stand-in for the torch dataset object that ``get_model`` consults."""

    class _StubDS:
        pass

    s = _StubDS()
    s.active_keys = list(active_keys)
    s.resolution = tuple(resolution)
    s.phi_resolution = (resolution[3], resolution[2], resolution[4])
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/model/gyroswin/gyroswin_large_pretraining.yaml",
                   help="path to a gyroswin model config (under upstream torch repo)")
    p.add_argument("--out", default="/tmp/gyroswin_random.pth")
    p.add_argument("--torch-repo", default="/system/user/publicwork/galletti/git/neural-gyrokinetics-gitlab",
                   help="root of the upstream torch repo (for ``neugk.*`` imports)")
    args = p.parse_args()

    if args.torch_repo not in sys.path:
        sys.path.insert(0, args.torch_repo)

    cfg_path = (
        args.config if os.path.isabs(args.config) else os.path.join(args.torch_repo, args.config)
    )
    mcfg = OmegaConf.load(cfg_path)
    # the torch ``get_model`` reads cfg.{model,dataset} — wire them up
    cfg = OmegaConf.create({
        "model": mcfg,
        "dataset": {
            "input_fields": ["df"],
            "separate_zf": True,
            "real_potens": True,
            "active_keys": ["re", "im"],
        },
        "logging": {"model_summary": False},
    })

    from neugk.gyroswin.models import get_model
    ds = _stub_dataset(cfg.dataset.active_keys)
    model = get_model(cfg, dataset=ds)
    state = model.state_dict()
    torch.save({"model_state_dict": state, "epoch": 0}, args.out)
    sz = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({sz / 1e6:.1f} MB, {len(state)} keys)")
    # print a small sample of the keys so the JAX translator can be eyeballed
    sample = sorted(state.keys())[:6] + ["...(elided)"] + sorted(state.keys())[-6:]
    for k in sample:
        if k.startswith("..."):
            print(f"  {k}")
        else:
            print(f"  {k:<70s}  shape={tuple(state[k].shape)}")


if __name__ == "__main__":
    main()
