"""Real-space <-> spectral transforms of the distribution function and the potential.

The transforms take numpy or jax arrays and run in their namespace. The spectral df is
``(..., s, kx, ky)`` complex with a zero-centred kx axis; the real-space df stacks its real
and imaginary parts, ``(2, ..., s, x, y)``. The spectral potential is ``(kx, s, ky)`` with a
zero-centred kx axis; the real potential is ``(x, s, y)``.
"""

from __future__ import annotations

from typing import Optional, Sequence


def df_to_spec(df):
    xp = df.__array_namespace__()
    spec = xp.fft.fftn(df[0] + 1j * df[1], axes=(-2, -1), norm="forward")
    return xp.fft.ifftshift(spec, axes=-2)


def spec_to_df(spec):
    xp = spec.__array_namespace__()
    phys = xp.fft.ifftn(xp.fft.fftshift(spec, axes=-2), axes=(-2, -1), norm="forward")
    return xp.stack([phys.real, phys.imag]).astype(xp.float32)


def phi_to_spec(phi, shape: Optional[Sequence[int]] = None):
    """Spectrum of the real potential.

    With ``shape`` the potential lives on a finer grid (the GKW ``Poten`` dump): its
    one-sided ky half, cropped around kx = 0 to ``shape``, is returned (the ``Spc3d``
    layout).
    """
    xp = phi.__array_namespace__()
    spec = xp.fft.fftshift(xp.fft.fftn(phi, axes=(0, 2), norm="forward"), axes=0)
    if shape is None:
        return spec
    spec = xp.fft.fftshift(spec, axes=2)[..., spec.shape[-1] // 2 :]
    nkx, _, nky = shape
    x0 = (spec.shape[0] - nkx) // 2 + (1 if spec.shape[0] % 2 == 0 else 0)
    return spec[x0 : x0 + nkx, :, :nky]


def spec_to_phi(spec, shape: Optional[Sequence[int]] = None):
    """Real potential of a spectrum, zero-padded first to the spectral ``shape`` when given.

    Only the one-sided ky half of ``spec`` enters (``irfftn``).
    """
    xp = spec.__array_namespace__()
    if shape is not None and tuple(spec.shape) != tuple(shape):
        (nkx, _, nky), (nx, _, ny) = shape, spec.shape
        x0 = (nkx - nx) // 2 + 1
        spec = xp.pad(spec, ((x0, nkx - nx - x0), (0, 0), (0, nky - ny)))
    nkx, _, nky = spec.shape
    spec = xp.fft.ifftshift(spec, axes=0)
    return xp.fft.irfftn(spec, axes=(0, 2), norm="forward", s=(nkx, nky))
