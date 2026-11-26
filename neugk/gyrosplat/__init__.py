"""GyroSplat: compress gyrokinetic distribution functions with Gaussian atoms while
keeping the physics readouts (the potential, the flux) exact to the byte budget.

The package is organised around one idea: a distribution function on a
(velocity x space) grid is written as a sum of separable Gaussian atoms with complex
amplitudes, optionally modulated by a plane-wave carrier.  Because the field is linear in
the amplitudes, three things are cheap and exact: the amplitude solve (variable
projection), the physics stage (a Gauss-Newton on the flux readout) and the byte
accounting.  Everything physics-specific lives behind a small `Backend` interface, so the
codec runs unchanged on any dataset that provides one.

Typical use::

    from neugk.gyrosplat import load_config, compress, score
    cfg = load_config("configs/gyrosplat/recipe/gkw_302x.yaml")
    model = compress(backend, cfg)
    row = score(model, backend, cfg)
"""
from neugk.gyrosplat.config import RecipeConfig, load_config
from neugk.gyrosplat.data import Snapshot, Backend
from neugk.gyrosplat.recipe import compress, decode, score

__all__ = [
    "RecipeConfig",
    "load_config",
    "Snapshot",
    "Backend",
    "compress",
    "decode",
    "score",
]
__version__ = "0.1.0"
