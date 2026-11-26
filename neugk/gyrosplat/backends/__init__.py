"""Physics backends.  A backend turns the stored array into the spectral object the field
equations use, solves the potentials, integrates the fluxes and scores a reconstruction.

    synthetic  -- a toy readout on a small grid, for tests and as a template for new data
    gkw        -- the GKW readout through the host `neugk` package (optional dependency)
"""
from neugk.gyrosplat.backends.synthetic import SyntheticBackend

__all__ = ["SyntheticBackend"]
