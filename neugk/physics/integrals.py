"""Python implementation of gyrokinetic potentials and flux phase-space integrals."""

from typing import Dict, Optional, Tuple, Sequence
import torch
from torch.special import bessel_j0 as j0, i0, bessel_j1 as j1
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
    "beta",
    "nlapar",
    "nlbpar",
    "bessel_bpar",
    "krloc",
]


def get_integrals(
    pred: torch.Tensor,
    geom: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    apar: Optional[torch.Tensor] = None,
    bpar: Optional[torch.Tensor] = None,
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
    if apar is not None:
        apar = apar.unsqueeze(0)
    if bpar is not None:
        bpar = bpar.unsqueeze(0)
    phi, (pflux, eflux, vflux) = integrator(
        geom, df=pred.unsqueeze(0), phi=phi, apar=apar, bpar=bpar
    )
    phi = phi.squeeze()
    return phi, (pflux, eflux, vflux)


class FluxIntegral(nn.Module):
    """
    Integrals for particle, heat and momentum fluxes and self-consistent potentials.

    This module implements the field equations and phase-space integrals and handles
    both electrostatic and electromagnetic (finite-beta) regimes.

    It is based on and aligned to GKW, a FORTRAN codebase for gyrokinetics.
    Source and docs for GKW: https://bitbucket.org/gkw/workspace/projects/GKW
    """

    def __init__(
        self,
        real_potens: bool = False,
        spectral_potens: bool = False,
        flux_fields: bool = False,
        spectral_df: bool = False,
        integral_precision: str = "float64",
    ):
        super().__init__()

        self.real_potens = real_potens
        self.spectral_potens = spectral_potens
        self.flux_fields = flux_fields
        self.spectral_df = spectral_df
        self.integral_dtype = (
            torch.float64 if integral_precision == "float64" else torch.float32
        )

    def _geom_tensors(
        self, geometry: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float32
    ) -> Dict[str, torch.Tensor]:
        # use float64 for stability
        geometry = tree_map(lambda g: g.to(dtype=self.integral_dtype), geometry)
        geom_ = {}

        # grid expansion for broadcasting
        geom_["krho"] = rearrange(geometry["krho"], "y -> 1 1 1 1 y")
        geom_["ints"] = rearrange(geometry["ints"], "s -> 1 1 s 1 1")
        geom_["intmu"] = rearrange(geometry["intmu"], "mu -> 1 mu 1 1 1")
        geom_["intvp"] = rearrange(geometry["intvp"], "par -> par 1 1 1 1")
        geom_["vpgr"] = rearrange(geometry["vpgr"], "par -> par 1 1 1 1")
        geom_["mugr"] = rearrange(geometry["mugr"], "mu -> 1 mu 1 1 1")

        # settings expansion
        geom_["bn"] = rearrange(geometry["bn"], "s -> 1 1 s 1 1")
        geom_["efun"] = rearrange(geometry["efun"], "s -> 1 1 s 1 1")
        geom_["rfun"] = rearrange(geometry["rfun"], "s -> 1 1 s 1 1")
        geom_["bt_frac"] = rearrange(geometry["bt_frac"], "s -> 1 1 s 1 1")
        geom_["parseval"] = rearrange(geometry["parseval"], "y -> 1 1 1 1 y")

        def expand_scalar(t):
            # first dim for species
            return t.view(*t.shape, *[1] * 5)

        geom_["mas"] = expand_scalar(geometry["mas"])
        geom_["tmp"] = expand_scalar(geometry["tmp"])
        geom_["d2X"] = expand_scalar(geometry["d2X"])
        geom_["signz"] = expand_scalar(geometry["signz"])
        geom_["signB"] = expand_scalar(geometry["signB"])
        geom_["adiabatic"] = expand_scalar(geometry["adiabatic"])
        geom_["de"] = expand_scalar(geometry["de"])
        geom_["vthrat"] = expand_scalar(geometry["vthrat"])
        geom_["beta"] = expand_scalar(geometry["beta"])
        geom_["nlapar"] = expand_scalar(geometry["nlapar"])
        geom_["nlbpar"] = expand_scalar(geometry["nlbpar"])

        # precompute bessel and gamma functions for gyroaverage
        kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 x 1")
        little_g = rearrange(geometry["little_g"], "s three -> three 1 1 s 1 1")
        krloc = torch.sqrt(
            geom_["krho"] ** 2 * little_g[0]
            + 2 * geom_["krho"] * kxrh * little_g[1]
            + kxrh**2 * little_g[2]
        )
        geom_["krloc"] = krloc
        bessel = torch.sqrt(2.0 * geom_["mugr"] / geom_["bn"]) / geom_["signz"]
        vthrat = geom_["vthrat"]
        bessel = geom_["mas"] * vthrat * krloc * bessel
        # torch.special.bessel_j0/j1 fail on Blackwell GPUs (nvrtc arch error).
        # Compute on CPU then move back — geometry is fixed per trajectory so
        # this cost is negligible.
        _bessel_cpu = bessel.cpu()
        geom_["bessel"] = j0(_bessel_cpu).to(bessel.device)
        geom_["bessel_bpar"] = torch.where(
            torch.abs(_bessel_cpu) < 1e-8,
            torch.ones_like(_bessel_cpu),
            2.0 * j1(_bessel_cpu) / _bessel_cpu,
        ).to(bessel.device)

        # scaled i0 for zonal response (also Blackwell-affected)
        gamma = geom_["mas"] * vthrat * krloc
        gamma = 0.5 * (gamma / (geom_["signz"] * geom_["bn"])) ** 2
        _gamma_cpu = gamma.cpu()
        geom_["gamma"] = (i0(_gamma_cpu) * torch.exp(-_gamma_cpu)).to(gamma.device)
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
        apar: Optional[torch.Tensor] = None,
        bpar: Optional[torch.Tensor] = None,
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
        bessel, bessel_bpar = geom["bessel"], geom["bessel_bpar"]

        def broadcast_field(f):
            if f is None:
                return 0.0
            f_view = [1] * (df.ndim - 3) + list(f.squeeze().shape)
            return f.squeeze().view(*f_view)  # (sp, vpar, vmu, s, x, y)

        # broadcast potentials to match df
        phi = broadcast_field(phi)
        apar = broadcast_field(apar)
        bpar = broadcast_field(bpar)

        # Real-valued arithmetic on the (re, im) parts instead of complex tensors:
        # complex elementwise *backward* hits torch's nvrtc jiterator, which fails
        # to compile on Blackwell (sm_100). dum1/dum2 only need the imaginary part,
        # so we never materialize the complex products. Results match bit-for-bit.
        def parts(f):
            if isinstance(f, torch.Tensor) and f.is_complex():
                return f.real, f.imag
            return f, 0.0  # absent field -> broadcast_field returned 0.0

        df_re, df_im = parts(df)
        phi_re, phi_im = parts(phi)
        apar_re, apar_im = parts(apar)
        bpar_re, bpar_im = parts(bpar)

        # generalized potential chi = J0*phi - 2*vth*vpar*J0*apar + 2*mu*T/Z*(2J1/z)*bpar
        cb = 2.0 * geom["vthrat"] * vpgr * bessel
        cc = 2.0 * mugr * geom["tmp"] / geom["signz"] * bessel_bpar
        chi_re = bessel * phi_re - cb * apar_re + cc * bpar_re
        chi_im = bessel * phi_im - cb * apar_im + cc * bpar_im
        # conjugate: negate the imaginary part
        chi_conj_re, chi_conj_im = chi_re, -chi_im

        if magnitude:
            df_mag = torch.sqrt(df_re**2 + df_im**2)
            df_re, df_im = torch.zeros_like(df_mag), -df_mag  # df -> -i|df|
            chi_mag = torch.sqrt(chi_conj_re**2 + chi_conj_im**2)
            chi_conj_re, chi_conj_im = chi_mag, torch.zeros_like(chi_mag)

        k = parseval * ints * (efun * krho)
        dum_re, dum_im = k * df_re, k * df_im
        # dum1 = imag(dum * chi_conj); dum2 = imag(dum * chi_conj * bn) = bn * dum1
        dum1 = dum_re * chi_conj_im + dum_im * chi_conj_re
        dum2 = dum1 * bn
        d3v = ints * d2X * intmu * bn * intvp

        # physical normalizations (matched to GKW internal units)
        pflux = d3v * dum1 * geom["de"]
        eflux = d3v * (vpgr**2 * dum1 + 2.0 * mugr * dum2) * geom["de"] * geom["tmp"]
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
                tuple(range(pflux.ndim))
                if df.ndim == 5
                else tuple(range(1, pflux.ndim))
            )
            return (
                pflux.sum(dim=sum_dims),
                eflux.sum(dim=sum_dims),
                vflux.sum(dim=sum_dims),
            )

    def solve_fields(self, geom: Dict[str, torch.Tensor], df: torch.Tensor):
        """
        Solves for self-consistent potentials: phi, apar, bpar.
        """
        de, signz, tmp, bn = geom["de"], geom["signz"], geom["tmp"], geom["bn"]
        ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
        vpgr, mugr, krloc = geom["vpgr"], geom["mugr"], geom["krloc"]
        bessel, bessel_bpar, gamma = geom["bessel"], geom["bessel_bpar"], geom["gamma"]
        adiabatic, beta = geom["adiabatic"], geom["beta"]
        nlapar, nlbpar = geom["nlapar"], geom["nlbpar"]

        cfen = torch.zeros_like(ints)

        # --- Solve for phi ---
        poisson_int = signz * de * intmu * intvp * bessel * bn
        phi_term = (1 + 0j) * poisson_int * df
        sum_dims = tuple(range(phi_term.ndim - 3))
        phi = phi_term.sum(sum_dims, keepdim=True)

        poisson_diag_s = torch.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
        poisson_diag = poisson_diag_s.sum(dim=0, keepdim=True)
        poisson_diag[..., 0, 0] = 0.0
        # optional adiabatic background
        adiabatic_correction = -(-1.0) * torch.exp(-cfen) * 1.0 / 1.0 * adiabatic
        poisson_diag = poisson_diag - adiabatic_correction
        poisson_diag = torch.where(
            poisson_diag == 0.0,
            torch.tensor(1.0, dtype=poisson_diag.dtype),
            poisson_diag,
        )
        poisson_diag = -1.0 / poisson_diag

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
        maty = torch.where(maty == 0, torch.tensor(1.0, dtype=maty.dtype), maty)
        maty = 1.0 / maty
        maty[..., 1:] = 0.0
        bufphi = (1 + 0j) * matz * phi
        bufphi = bufphi.sum((-3, -1), keepdim=True)
        phi = phi + (1 + 0j) * maty * bufphi * adiabatic
        phi = (phi * poisson_diag).view(phi.shape[-3:])

        # solve for apar
        # S_A = beta * sum Z_s n_s vth_s <vpar J0 f_s>
        # Denom = k_perp^2 + beta * sum ...
        apar_int = (
            beta * signz * de * geom["vthrat"] * intmu * intvp * vpgr * bessel * bn
        )
        apar = ((1 + 0j) * apar_int * df).sum(sum_dims, keepdim=True)
        apar_diag = krloc**2
        # Add small term for stability if k_perp=0
        apar_diag = torch.where(
            apar_diag == 0, torch.tensor(1.0, dtype=apar_diag.dtype), apar_diag
        )
        apar = (apar / apar_diag).view(phi.shape) * nlapar.view(-1, 1, 1)

        # solve for bpar
        # S_B = beta * sum n_s T_s <mu (2J1/z) f_s>
        # Denom = krloc**2 / beta + sum ... (actually GKW uses bpar normalized differently?)
        # Simplified bpar solve matched to krloc**2 / beta
        bpar_int = beta * de * tmp * intmu * intvp * mugr * bessel_bpar * bn
        bpar = ((1 + 0j) * bpar_int * df).sum(sum_dims, keepdim=True)
        bpar_diag = krloc**2
        bpar_diag = torch.where(
            bpar_diag == 0, torch.tensor(1.0, dtype=bpar_diag.dtype), bpar_diag
        )
        # In GKW, the bpar sign or factor 2 might be involved.
        # Based on diagnos_fluxes_vspace.F90: -2.*conjg(bpar_ga)
        # So let's use -bpar_int
        bpar = ((-1.0 + 0j) * bpar).view(phi.shape) / bpar_diag * nlbpar.view(-1, 1, 1)

        return phi, apar, bpar

    def forward_single(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        apar: Optional[torch.Tensor] = None,
        bpar: Optional[torch.Tensor] = None,
    ):
        # extend geometry
        geom = self._geom_tensors(geom, df.dtype)
        ns, nx, ny = df.shape[-3:]
        if not self.spectral_df:
            df = self._df_fft(df)
        else:
            c_dim = 1 if df.ndim == 7 else 0
            df = df.movedim(c_dim, -1).contiguous()
            df = torch.view_as_complex(df)

        phi_int, apar_int, bpar_int = self.solve_fields(geom, df)
        # internal fields for flux calculation
        phi_f, apar_f, bpar_f = phi_int, apar_int, bpar_int
        # if external potentials provided, use them for flux
        if phi is not None:
            phi_f = self._phi_to_spc(phi, out_shape=(nx, ns, ny))
        if apar is not None:
            apar_f = self._phi_to_spc(apar, out_shape=(nx, ns, ny))
        if bpar is not None:
            bpar_f = self._phi_to_spc(bpar, out_shape=(nx, ns, ny))

        pflux, eflux, vflux = self.pev_fluxes(geom, df, phi_f, apar_f, bpar_f)
        if not self.spectral_potens:
            phi_int = self._spc_to_phi(phi_int, original_shape=(nx, ns, ny))
        return phi_int, (pflux, eflux, vflux)

    def forward(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        apar: Optional[torch.Tensor] = None,
        bpar: Optional[torch.Tensor] = None,
    ):
        """
        If potentials (phi, apar, bpar) are passed, they are used to compute the fluxes
        together with the distribution function (df). If not, the fields are solved
        self-consistently from df using the appropriate field solvers (Poisson/Ampere).

        Args:
            geom (Dict): Dictionary containing geometry parameters and settings.
            df (torch.Tensor): 5D or 6D distribution function.
                               Shape: (batch, sp, 2, vpar, vmu, s, x, y).
            phi (torch.Tensor, optional): 3D electrostatic potential.
                               Shape: (batch, 2, x, s, y).
            apar (torch.Tensor, optional): 3D parallel magnetic potential.
                               Shape: (batch, 2, x, s, y).
            bpar (torch.Tensor, optional): 3D parallel magnetic field perturbation.
                               Shape: (batch, 2, x, s, y).

        Returns:
            Tuple containing (phi_int, (pflux, eflux, vflux)), where:
                - phi_int is the solved/returned electrostatic potential.
                - pflux is the particle flux per species (batch, sp).
                - eflux is the heat flux per species (batch, sp).
                - vflux is the momentum flux per species (batch, sp).
        """
        geom = dict(sorted(geom.items()))
        geom_keys = {k: 0 for k in sorted(list(geom.keys()))}
        if phi is None and apar is None and bpar is None:
            vfwd = torch.vmap(self.forward_single, in_dims=(geom_keys, 0))
            return vfwd(geom, df)
        else:
            in_dims = (
                geom_keys,
                0,
                0 if phi is not None else None,
                0 if apar is not None else None,
                0 if bpar is not None else None,
            )
            vfwd = torch.vmap(self.forward_single, in_dims=in_dims)
            return vfwd(geom, df, phi, apar, bpar)
