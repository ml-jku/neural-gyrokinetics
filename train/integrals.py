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
]


class FluxIntegral(nn.Module):
    def __init__(self):
        super().__init__()

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
        geom_ = {}
        # expand geometry constants for broadcasting
        # grids
        geom_["krho"] = rearrange(geometry["krho"], "b y -> b 1 1 1 1 y")
        geom_["ints"] = rearrange(geometry["ints"], "b s -> b 1 1 s 1 1")
        geom_["intmu"] = rearrange(geometry["intmu"], "b mu -> b 1 mu 1 1 1")
        geom_["intvp"] = rearrange(geometry["intvp"], "b par -> b par 1 1 1 1")
        geom_["vpgr"] = rearrange(geometry["vpgr"], "b par -> b par 1 1 1 1")
        geom_["mugr"] = rearrange(geometry["mugr"], "b mu -> b 1 mu 1 1 1")
        # settings
        geom_["bn"] = rearrange(geometry["bn"], "b s -> b 1 1 s 1 1")
        geom_["efun"] = rearrange(geometry["efun"], "b s -> b 1 1 s 1 1")
        geom_["rfun"] = rearrange(geometry["rfun"], "b s -> b 1 1 s 1 1")
        geom_["bt_frac"] = rearrange(geometry["bt_frac"], "b s -> b 1 1 s 1 1")
        geom_["parseval"] = rearrange(geometry["parseval"], "b y -> b 1 1 1 1 y")
        geom_["mas"] = rearrange(geometry["mas"], "b -> b 1 1 1 1 1")
        geom_["tmp"] = rearrange(geometry["tmp"], "b -> b 1 1 1 1 1")
        geom_["d2X"] = rearrange(geometry["d2X"], "b -> b 1 1 1 1 1")
        geom_["signz"] = rearrange(geometry["signz"], "b -> b 1 1 1 1 1")
        geom_["signB"] = rearrange(geometry["signB"], "b -> b 1 1 1 1 1")
        # bessel for gyroaverage
        vthrat = rearrange(geometry["vthrat"], "b -> b 1 1 1 1 1")
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
        # exponentially scaled bessel i0 function
        gamma = geom_["mas"] * vthrat * krloc
        gamma = 0.5 * (gamma / (geom_["signz"] * geom_["bn"])) ** 2
        geom_["gamma"] = i0(gamma) * torch.exp(-gamma)
        return tree_map(lambda g: g.to(dtype=dtype), geom_)

    def _df_fft(self, df: torch.Tensor, norm: str = "forward"):
        df = df.movedim(0, -1).contiguous()
        df = torch.view_as_complex(df)
        df = torch.fft.fftn(df, dim=(3, 4), norm=norm)
        return torch.fft.ifftshift(df, dim=(3,))

    def _phi_to_spc(
        self,
        phi: torch.Tensor,
        out_shape: Tuple,
        shift_axes: Sequence[int] = (0,),
        norm: str = "forward",
    ):
        phi = phi.movedim(0, -1).contiguous()
        phi = torch.view_as_complex(phi)
        phi = torch.fft.fftn(phi, dim=(0, 2), norm=norm)
        phi = torch.fft.fftshift(phi, dim=shift_axes)
        # unpad (and positive half of spectra)
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
            # pad x
            nx, _, ny = original_shape
            x_pad_total = nx - spc_nx
            x_pad_left = x_pad_total // 2
            x_pad_right = x_pad_total - x_pad_left
            spc = F.pad(spc, (0, 0, 0, 0, x_pad_left, x_pad_right))
            # y full spectrum and pad
            spc_flipped_y = torch.flip(spc, dims=[-1])
            spc = torch.cat([spc_flipped_y, spc], dim=-1)
            y_pad_total = ny - spc_ny * 2
            y_pad_left = y_pad_total // 2
            y_pad_right = y_pad_total - y_pad_left
            spc = F.pad(spc, (y_pad_left, y_pad_right, 0, 0))
        # ifft
        phi = torch.fft.ifftshift(spc, dim=shift_axes)
        phi = torch.fft.ifftn(phi, dim=(0, 2), norm=norm)
        phi = torch.view_as_real(phi).movedim(-1, 0).contiguous()
        return phi  # (c, x, s, y)

    def pev_fluxes(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: torch.Tensor,
        magnitude: bool = False,
    ):
        """
        Computes particle, heat and momentum fluxes based on the distribution function
        and electrostatic potential.

        Args:
            geom (Dict): Dictionary containing geometry parameters and settings.
            df (torch.Tensor): 5D density function. Shape: (2, vpar, vmu, s, x, y).
            phi (torch.Tensor): 3D electrostatic potential. Shape: (2, x, s, y).
            magnitude (bool, optional): Use df and phi absolutes. Default: False.
        """
        bn, bt_frac, parseval = geom["bn"], geom["bt_frac"], geom["parseval"]
        rfun, efun, d2X, signB = geom["rfun"], geom["efun"], geom["d2X"], geom["signB"]
        ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
        vpgr, mugr, krho = geom["vpgr"], geom["mugr"], geom["krho"]
        bessel = geom["bessel"]
        phi = rearrange(phi.squeeze(), "s x y -> 1 1 s x y")
        # prepare potential for gyroaverage
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
        pflux = d3v * dum1
        eflux = d3v * (vpgr**2 * dum1 + 2 * mugr * dum2)
        vflux = d3v * (dum1 * vpgr * rfun * bt_frac * signB)
        return pflux.sum(), eflux.sum(), vflux.sum()

    def phi(self, geom: Dict[str, torch.Tensor], df: torch.Tensor):
        """
        Computes electrostatic potential integral from the distribution function.

        Args:
            geom (Dict): Dictionary containing geometry parameters and settings.
            df (torch.Tensor): 5D density function. Shape: (2, vpar, vmu, s, x, y).
        """
        # density of the species
        de = 1.0
        signz, tmp, bn = geom["signz"], geom["tmp"], geom["bn"]
        ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
        bessel, gamma = geom["bessel"], geom["gamma"]
        cfen = torch.zeros_like(ints)
        # poisson integral term
        poisson_int = signz * de * intmu * intvp * bessel * bn
        poisson_int = torch.where(torch.abs(intvp) < 1e-9, 0.0, poisson_int)
        # diagonal zonal flow corrections
        diagz = signz * (gamma - 1.0) * torch.exp(-cfen) / tmp
        matz = -ints / (signz * de * (diagz - torch.exp(-cfen) / tmp))
        matz[..., 1:] = 0.0
        maty = (-matz * torch.exp(-cfen)).sum((2,), keepdim=True)
        maty = tmp / (de * torch.exp(-cfen)) + maty / torch.exp(-cfen)
        maty[..., 0, :] = 1 + 0j
        maty = torch.where(maty == 0, 1.0, maty)
        maty = 1 / maty
        maty[..., 1:] = 0.0
        # diagonal poisson (normalization)
        poisson_diag = torch.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
        poisson_diag[..., 0, 0] = 0.0
        poisson_diag = poisson_diag + signz * torch.exp(-cfen) * de / tmp
        # prepare phi integral form
        phi = (1 + 0j) * poisson_int * df
        # 1st: integrate vspace
        phi = phi.sum((0, 1), keepdim=True)
        # 2nd: zonal flow corrections on correct kxky axes
        bufphi = (1 + 0j) * matz * phi
        bufphi = bufphi.sum((2, 4), keepdim=True)
        phi = phi + (1 + 0j) * maty * bufphi
        # 3rd: poisson normalization
        phi = phi * poisson_diag
        return phi.squeeze()

    def forward_single(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ):
        ns, nx, ny = df.shape[3:]
        # df to fourier
        df = self._df_fft(df)  # (par, mu, s, x, y)
        phi_int = self.phi(geom, df)  # (s, x, y)
        phi_ = phi_int.clone()
        if phi is not None:
            phi_ = self._phi_to_spc(phi, out_shape=(nx, ns, ny))  # (s, x, y)
        pflux, eflux, vflux = self.pev_fluxes(geom, df, phi_)
        # integrated phi repad and back to real
        phi_int = self._spc_to_phi(phi_int)
        return phi_int, (pflux, eflux, vflux)

    def forward(
        self,
        geom: Dict[str, torch.Tensor],
        df: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ):
        """
        Integrals for particle, heat and momentum fluxes and electrostatic potential.

        The implementation is based on GKW, a FORTRAN codebase for gyrokinetics.
        Source and docs for GKW: https://bitbucket.org/gkw/workspace/projects/GKW

        If a potential phi is passed, then it is ultimately used to compute the fluxes,
        together with the distribution function df. If not, then df is used to compute
        both integrals (warning: unreliable in some geometries).

        Args:
            geom (Dict): Dictionary containing geometry parameters and settings.
            df (torch.Tensor): 5D density function. Shape: (b, 2, vpar, vmu, s, x, y).
            phi (torch.Tensor, opt): 3D electrostatic potential. Shape: (b, 2, x, s, y).

        Returns:
            Tuple contraining (phi_int, (pflux, eflux, vflux)), where
                - phi_int is the electrostatic potential
                - pflux is the summed particle flux
                - eflux is the summed heat flux
                - vflux is the summed momentum flux
        """
        geom = self._geom_tensors(geom, df.dtype)
        geom = dict(sorted(geom.items()))  # NOTE: order matters in torch vmap...
        if phi is None:
            return self._vmap_fwd_df(geom, df)
        else:
            return self._vmap_fwd_df_phi(geom, df, phi)
