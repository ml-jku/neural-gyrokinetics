"""Gyrokinetic physics: field/flux integrals and turbulence diagnostics.

Single source of truth for the physics operators and physics-informed losses,
shared by the autoencoder, neural-field and diffusion paths.
"""

from neugk.physics.integrals import FluxIntegral, get_integrals, GEOM_KEYS
from neugk.physics.diagnostics import (
    phi_fft,
    diagnostics,
    monotonicity_loss,
    mass_loss,
    velocity_moment_errors,
    integral_losses,
    spectra_losses,
    compute_data_loss,
    compute_integral_loss,
    compute_spectral_loss,
    served_spectral_loss,
)

__all__ = [
    "FluxIntegral",
    "get_integrals",
    "GEOM_KEYS",
    "phi_fft",
    "diagnostics",
    "monotonicity_loss",
    "mass_loss",
    "velocity_moment_errors",
    "integral_losses",
    "spectra_losses",
    "compute_data_loss",
    "compute_integral_loss",
    "compute_spectral_loss",
    "served_spectral_loss",
]
