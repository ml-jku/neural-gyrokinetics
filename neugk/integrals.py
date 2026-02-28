"""Python implementation of gyrokinetic potentials and flux phase-space integrals."""

from typing import Dict, Optional, Tuple, Sequence
import torch
from torch.special import bessel_j0 as j0, i0
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torch.utils._pytree import tree_map


GEOM_KEYS = [
    "krho",
    "ints",
    "intmu",
    "intvp",
    "vpgr",
    "mugr",
    "bn",
    "efun",
    "rfun",
    "bt_frac",
    "parseval",
    "mas",
    "tmp",
    "d2X",
    "signz",
    "signB",
    "bessel",
    "gamma",
    "adiabatic",
    "de",
    "vthrat",
]


def get_integrals(
    pred: torch.Tensor,
    geom: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    flux_fields: bool = False,
    spectral_df: bool = False,
    spectral_potens: bool = False,
):
    if pred.ndim == 6 and pred.shape[0] != 2:
        pred = pred[[0, 1]] + pred[[2, 3]]
    elif pred.ndim == 7 and pred.shape[1] != 2:
        pred = pred[:, [0, 1]] + pred[:, [2, 3]]
    geom = {k: g.unsqueeze(0).to(pred.device) for k, g in geom.items()}
    integrator = FluxIntegral(
        real_potens=False,
        flux_fields=flux_fields,
        spectral_df=spectral_df,
        spectral_potens=spectral_potens,
    )
    integrator.to(pred.device)
    if phi is not None:
        phi = phi.unsqueeze(0)
    phi, (pflux, eflux, vflux) = integrator(geom, df=pred.unsqueeze(0), phi=phi)
    phi = phi.squeeze()
    return phi, (pflux, eflux, vflux)


class FluxIntegral(nn.Module):
    """
    Computes physical fluxes and self-consistent potential from the distribution function.

    Physics Context (derived from GKW):
    - The field solver operates in a local flux-tube geometry (kx, ky, s).
    - Quasi-neutrality sums the charge density responses of all species.
    - Potential phi is macroscopic (no species axis).
    - Fluxes are computed via cross-correlations: Im(f * phi^*).
    - Standard GKW normalization: length to major radius R, velocity to thermal velocity.
    """

    def __init__(
        self,
        real_potens: bool = False,
        spectral_potens: bool = False,
        flux_fields: bool = False,
        spectral_df: bool = False,
    ):
        super().__init__()

        self.real_potens = real_potens
        self.spectral_potens = spectral_potens
        self.flux_fields = flux_fields
        self.spectral_df = spectral_df

        self._vmap_fwd_df = torch.vmap(
            self.forward_single,
            in_dims=({k: 0 for k in sorted(GEOM_KEYS)}, 0),
        )
        self._vmap_fwd_df_phi = torch.vmap(
            self.forward_single,
            in_dims=({k: 0 for k in sorted(GEOM_KEYS)}, 0, 0),
        )

    def _geom_tensors(
        self, geometry: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float32
    ) -> Dict[str, torch.Tensor]:
        # use float64 for stability
        geometry = tree_map(lambda g: g.to(dtype=torch.float64), geometry)
        geom_ = {}

        # grid expansion for broadcasting
        geom_["krho"] = rearrange(geometry["krho"], "b y -> b 1 1 1 1 y")
        geom_["ints"] = rearrange(geometry["ints"], "b s -> b 1 1 s 1 1")
        geom_["intmu"] = rearrange(geometry["intmu"], "b mu -> b 1 mu 1 1 1")
        geom_["intvp"] = rearrange(geometry["intvp"], "b par -> b par 1 1 1 1")
        geom_["vpgr"] = rearrange(geometry["vpgr"], "b par -> b par 1 1 1 1")
        geom_["mugr"] = rearrange(geometry["mugr"], "b mu -> b 1 mu 1 1 1")

        # settings expansion
        geom_["bn"] = rearrange(geometry["bn"], "b s -> b 1 1 s 1 1")
        geom_["efun"] = rearrange(geometry["efun"], "b s -> b 1 1 s 1 1")
        geom_["rfun"] = rearrange(geometry["rfun"], "b s -> b 1 1 s 1 1")
        geom_["bt_frac"] = rearrange(geometry["bt_frac"], "b s -> b 1 1 s 1 1")
        geom_["parseval"] = rearrange(geometry["parseval"], "b y -> b 1 1 1 1 y")

        def expand_scalar(t):
            return t.view(*t.shape, 1, 1, 1, 1, 1)

        geom_["mas"] = expand_scalar(geometry["mas"])
        geom_["tmp"] = expand_scalar(geometry["tmp"])
        geom_["d2X"] = expand_scalar(geometry["d2X"])
        geom_["signz"] = expand_scalar(geometry["signz"])
        geom_["signB"] = expand_scalar(geometry["signB"])
        geom_["adiabatic"] = expand_scalar(geometry["adiabatic"])
        geom_["de"] = expand_scalar(geometry["de"])
        geom_["vthrat"] = expand_scalar(geometry["vthrat"])
        vthrat = geom_["vthrat"]

        # gyroaverage bessel
        kxrh = rearrange(geometry["kxrh"], "b x -> b 1 1 1 x 1")
        little_g = rearrange(geometry["little_g"], "b s three -> three b 1 1 s 1 1")
        krloc = torch.sqrt(
            geom_["krho"] ** 2 * little_g[0]
            + 2 * geom_["krho"] * kxrh * little_g[1]
            + kxrh**2 * little_g[2]
        )
        bessel = torch.sqrt(2.0 * geom_["mugr"] / geom_["bn"]) / geom_["signz"]
        bessel = geom_["mas"] * vthrat * krloc * bessel
        geom_["bessel"] = j0(bessel)

        # scaled i0 for zonal response
        gamma = geom_["mas"] * vthrat * krloc
        gamma = 0.5 * (gamma / (geom_["signz"] * geom_["bn"])) ** 2
        geom_["gamma"] = i0(gamma) * torch.exp(-gamma)
        return tree_map(lambda g: g.to(dtype=dtype), geom_)

    def _df_fft(self, df: torch.Tensor, norm: str = "forward"):
        c_dim = 1 if df.ndim == 7 else 0
        df = df.movedim(c_dim, -1).contiguous()
        df = torch.view_as_complex(df)
        df = torch.fft.fftn(df, dim=(-2, -1), norm=norm)
        return torch.fft.ifftshift(df, dim=(-2,))

    def _phi_to_spc(
        self,
        phi: torch.Tensor,
        out_shape: Tuple,
        shift_axes: Sequence[int] = (0,),
        norm: str = "forward",
    ):
        if not self.real_potens:
            phi = phi.movedim(0, -1).contiguous()
            phi = torch.view_as_complex(phi)
        phi = torch.fft.fftn(phi, dim=(0, 2), norm=norm)
        phi = torch.fft.fftshift(phi, dim=shift_axes)
        if phi.shape != out_shape:
            nx, _, ny = out_shape
            phi = phi[..., phi.shape[-1] // 2 :]
            xpad = (phi.shape[0] - nx) // 2
            xpad = xpad + 1 if (phi.shape[0] % 2 == 0) else xpad
            phi = phi[xpad : nx + xpad, :, :ny]
        return rearrange(phi, "x s y -> s x y")

    def _spc_to_phi(
        self,
        spc: torch.Tensor,
        original_shape: Tuple = (392, 16, 96),
        repad: bool = False,
        shift_axes: Sequence[int] = (0,),
        norm: str = "forward",
    ):
        spc = rearrange(spc, "s x y -> x s y")
        spc_nx, _, spc_ny = spc.shape
        if repad:
            nx, _, ny = original_shape
            x_pad_total = nx - spc_nx
            x_pad_left = x_pad_total // 2
            x_pad_right = x_pad_total - x_pad_left
            spc = F.pad(spc, (0, 0, 0, 0, x_pad_left, x_pad_right))
            spc_flipped_y = torch.flip(spc, dims=[-1])
            spc = torch.cat([spc_flipped_y, spc], dim=-1)
            y_pad_total = ny - spc_ny * 2
            y_pad_left = y_pad_total // 2
            y_pad_right = y_pad_total - y_pad_left
            spc = F.pad(spc, (y_pad_left, y_pad_right, 0, 0))
        phi = torch.fft.ifftshift(spc, dim=shift_axes)
        if self.real_potens:
            phi = torch.fft.irfftn(
                phi, dim=(0, 2), norm=norm, s=[spc_nx, spc_ny]
            ).float()
        else:
            phi = torch.fft.ifftn(phi, dim=(0, 2), norm=norm)
            phi = torch.view_as_real(phi).movedim(-1, 0).contiguous()
        return phi

    def pev_fluxes(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: torch.Tensor,
        magnitude: bool = False,
    ):
        """
        Computes particle, heat and momentum fluxes.

        df shape: (sp, vpar, vmu, s, x, y).
        phi shape: (s, x, y).
        """
        bn, bt_frac, parseval = geom["bn"], geom["bt_frac"], geom["parseval"]
        rfun, efun, d2X, signB = geom["rfun"], geom["efun"], geom["d2X"], geom["signB"]
        ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
        vpgr, mugr, krho = geom["vpgr"], geom["mugr"], geom["krho"]
        bessel = geom["bessel"]

        # broadcast phi to match df
        phi_view = [1] * (df.ndim - 3) + list(phi.squeeze().shape)
        phi = phi.squeeze().view(*phi_view)

        # gyroaverage
        phi_gyro = bessel * phi
        if magnitude:
            df = -1j * torch.abs(df)
            phi_gyro = torch.abs(phi_gyro)
        dum = parseval * ints * (efun * krho) * df
        dum1 = dum * torch.conj(phi_gyro)
        dum2 = dum1 * bn
        d3v = ints * d2X * intmu * bn * intvp
        dum1 = torch.imag(dum1)
        dum2 = torch.imag(dum2)

        # physical normalizations
        pflux = d3v * dum1 * geom["de"] * geom["vthrat"]
        eflux = (
            d3v
            * (vpgr**2 * dum1 + 2 * mugr * dum2)
            * geom["de"]
            * geom["tmp"]
            * geom["vthrat"]
        )
        vflux = (
            d3v
            * (dum1 * vpgr * rfun * bt_frac * signB)
            * geom["de"]
            * geom["mas"]
            * (geom["vthrat"] ** 2)
        )

        if self.flux_fields:
            return pflux, eflux, vflux
        else:
            sum_dims = (
                tuple(range(df.ndim)) if df.ndim == 5 else tuple(range(1, df.ndim))
            )
            return (
                pflux.sum(dim=sum_dims),
                eflux.sum(dim=sum_dims),
                vflux.sum(dim=sum_dims),
            )

    def phi(self, geom: Dict[str, torch.Tensor], df: torch.Tensor):
        """
        Solves for the self-consistent electrostatic potential phi.
        """
        de, signz, tmp, bn = geom["de"], geom["signz"], geom["tmp"], geom["bn"]
        ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
        bessel, gamma = geom["bessel"], geom["gamma"]
        adiabatic = geom["adiabatic"]
        cfen = torch.zeros_like(ints)

        poisson_int = signz * de * intmu * intvp * bessel * bn
        poisson_int = torch.where(torch.abs(intvp) < 1e-9, 0.0, poisson_int)

        # macroscopic response summed over species
        poisson_diag_s = torch.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
        if poisson_diag_s.ndim == 6 and poisson_diag_s.shape[0] > 1:
            poisson_diag = poisson_diag_s.sum(dim=0, keepdim=True)
        else:
            poisson_diag = poisson_diag_s

        poisson_diag[..., 0, 0] = 0.0

        # optional adiabatic background
        adiabatic_correction = -(-1.0) * torch.exp(-cfen) * 1.0 / 1.0 * adiabatic
        poisson_diag = poisson_diag - adiabatic_correction
        poisson_diag = -1 / poisson_diag

        phi = (1 + 0j) * poisson_int * df

        # integrate velocity and species
        sum_dims = tuple(range(df.ndim - 3))
        phi = phi.sum(sum_dims, keepdim=True)

        # zonal flow correction (ions only)
        s_idx = 0 if signz.shape[0] > 1 else slice(None)
        ion_signz, ion_gamma, ion_tmp, ion_de = (
            signz[s_idx],
            gamma[s_idx],
            tmp[s_idx],
            de[s_idx],
        )

        diagz = ion_signz * (ion_gamma - 1.0) * torch.exp(-cfen) / ion_tmp
        matz = -ints / (ion_signz * ion_de * (diagz - torch.exp(-cfen) / ion_tmp))
        matz[..., 1:] = 0.0
        maty = (-matz * torch.exp(-cfen)).sum((-3,), keepdim=True)
        maty = ion_tmp / (ion_de * torch.exp(-cfen)) + maty / torch.exp(-cfen)
        maty[..., 0, :] = 1 + 0j
        maty = torch.where(maty == 0, 1.0, maty)
        maty = 1 / maty
        maty[..., 1:] = 0.0

        bufphi = (1 + 0j) * matz * phi
        bufphi = bufphi.sum((-3, -1), keepdim=True)
        phi = phi + (1 + 0j) * maty * bufphi * adiabatic

        phi = phi * poisson_diag
        return phi.view(phi.shape[-3:])

    def forward_single(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ):
        ns, nx, ny = df.shape[-3:]
        if not self.spectral_df:
            df = self._df_fft(df)
        else:
            c_dim = 1 if df.ndim == 7 else 0
            df = df.movedim(c_dim, -1).contiguous()
            df = torch.view_as_complex(df)
        phi_int = self.phi(geom, df)
        phi_ = phi_int.clone()
        if phi is not None:
            phi_ = self._phi_to_spc(phi, out_shape=(nx, ns, ny))
        pflux, eflux, vflux = self.pev_fluxes(geom, df, phi_)
        if not self.spectral_potens:
            phi_int = self._spc_to_phi(phi_int, original_shape=(nx, ns, ny))
        return phi_int, (pflux, eflux, vflux)

    def forward(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ):
        """Integrals for physical fluxes and self-consistent potential."""
        geom = self._geom_tensors(geom, df.dtype)
        geom = dict(sorted(geom.items()))
        if phi is None:
            return self._vmap_fwd_df(geom, df)
        else:
            return self._vmap_fwd_df_phi(geom, df, phi)
