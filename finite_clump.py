from numba import cfunc
from numba.types import float64
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar, brentq
from scipy.special import gamma as Gamma
from scipy.special import beta
from scipy.special import betainc

from constants import kpc_to_cm, rho_critical, GeV_to_m_sun
from constants import dampe_excess_bin_low, dampe_excess_bin_high
from constants import dampe_excess_iflux, dn_de_gamma_AP, _constrain_ep_spec
from nfw_clump import dphi2_de_dr as dphi2_de_dr_nfw
from nfw_clump import dJ_dr as dJ_dr_nfw
from nfw_clump import NFW_params
from tt_clump import dphi2_de_dr as dphi2_de_dr_tt
from tt_clump import dJ_dr as dJ_dr_tt
from tt_clump import TT_params


def rho(r, halo_params):
    """Computes halo density.

    Parameters
    ----------
    r : float
        Distance from halo center to point of interest (kpc). Must be positive.
    halo_params
        A set of halo parameters.

    Returns
    -------
    rho : float
        Halo density at specified point in GeV / cm^3.
    """
    if halo_params.__class__.__name__ == "NFW_params":
        return halo_params.rhos * (halo_params.rs / r)**halo_params.gamma * \
                (1. + r / halo_params.rs)**(halo_params.gamma - 3.)
    elif halo_params.__class__.__name__ == "TT_params":
        return halo_params.rho0 * (halo_params.Rb / r)**halo_params.gamma * \
                np.exp(-r / halo_params.Rb)


def mass(halo_params):
    """Computes total mass of halo.

    Parameters
    ----------
    halo_params
        A set of halo parameters.

    Returns
    -------
    M : float
        Total halo mass (for TT profile) or virial mass (for NFW profile)
        (number of solar masses)
    """
    if halo_params.__class__.__name__ == "NFW_params":
        # Find virial mass numerically
        # NOTE: the "magic number" of 10000 should be fine for this project.
        try:
            r_vir = brentq(lambda r: rho(r, halo_params) - rho_critical,
                           halo_params.rs, halo_params.rs * 10000, xtol=1e-200)
        except RuntimeError:
            r_vir = np.nan

        # Have to integrate numerically: analytic result has an imaginary part
        factor = 4.*np.pi * GeV_to_m_sun * kpc_to_cm**3
        return factor * quad(lambda r: r**2 * rho(r, halo_params),
                             0, r_vir, epsabs=0, epsrel=1e-5)[0]
    elif halo_params.__class__.__name__ == "TT_params":
        return (4. * np.pi * halo_params.rho0 *
                (halo_params.Rb * kpc_to_cm)**3 *
                Gamma(3. - halo_params.gamma)) * GeV_to_m_sun


def luminosity(halo_params, mx, sv=3e-26, fx=2.):
    factor = sv / (2*fx*mx**2)

    if halo_params.__class__.__name__ == "NFW_params":
        rs_cm = kpc_to_cm * halo_params.rs
        L_clump = (4*np.pi*halo_params.rhos**2*rs_cm**3) / \
            (30 - 47*halo_params.gamma + 24*halo_params.gamma**2 -
             4*halo_params.gamma**3)
    elif halo_params.__class__.__name__ == "TT_params":
        Rb_cm = kpc_to_cm * halo_params.Rb
        L_clump = 2**(-1 + 2*halo_params.gamma) * np.pi * Rb_cm**3 * \
            halo_params.rho0**2 * Gamma(3 - 2*halo_params.gamma)

    return factor * L_clump


def lum_to_rho_norm(lum, halo_params, mx, sv=3e-26, fx=2):
    return np.sqrt(lum / luminosity(halo_params, mx, sv, fx))


def dphi_de_e(e, d, halo_params, mx, sv=3e-26, fx=2):
    """Computes dphi/dE|_{e-} for a DM clump.

    Parameters
    ----------
    e : float or float array
        Electron energies, GeV.
    d : float
        Distance to the clump's center, kpc.
    halo_params : namedtuple
    mx : float
        DM mass, GeV.
    sv : float
        DM self-annihilation cross section, cm^3/s.
    gamma : float
        NFW power index.
    """
    if e > mx:
        return 0.
    else:  # perform integration over r
        if halo_params.__class__.__name__ == "NFW_params":
            args = (e, d, halo_params.rs, halo_params.rhos, halo_params.gamma,
                    mx, sv, fx)
            dphi2_de_dr = dphi2_de_dr_nfw
        elif halo_params.__class__.__name__ == "TT_params":
            args = (e, d, halo_params.Rb, halo_params.rho0, halo_params.gamma,
                    mx, sv, fx)
            dphi2_de_dr = dphi2_de_dr_tt

        return quad(dphi2_de_dr, 0, 100.*d,
                    args, points=[d], epsabs=0, epsrel=1e-5)[0]


def J_factor(th_max, d, halo_params):
    """Computes J factor for a target region.

    Parameters
    ----------
    th_max : float
        Angular radius of the target.
    d : float
        Distance to center of DM clump in kpc.
    halo_params : namedtuple

    Returns
    -------
    J : float
        J factor in GeV^2/cm^5
    """
    if halo_params.__class__.__name__ == "NFW_params":
        args = (th_max, d, halo_params.rs, halo_params.rhos, halo_params.gamma)
        dJ_dr = dJ_dr_nfw
    elif halo_params.__class__.__name__ == "TT_params":
        args = (th_max, d, halo_params.Rb, halo_params.rho0, halo_params.gamma)
        dJ_dr = dJ_dr_tt

    # Split integral around center of clump
    int_near = quad(dJ_dr, 0., d,
                    args, points=[d], epsabs=0, epsrel=1e-5)[0]
    int_far = quad(dJ_dr, d, 100.*d,
                   args, points=[d], epsabs=0, epsrel=1e-5)[0]

    return (int_near + int_far) * kpc_to_cm


def dphi_de_gamma(e, th_max, d, halo_params, mx, sv=3e-26, fx=2):
    """Computes dphi/dE|_gamma for a DM clump.

    Parameters
    ----------
    e : list of floats
        Photon energies in GeV
    th_max : float
        Angular radius of observing region (rad)
    d : float
        Distance to clump center in kpc
    mx : float
        DM mass in GeV
    sv : float
        DM self-annihilation cross section in cm^3 / s
    fx : int
        1 if DM is self-conjugate, 2 if not.

    Returns
    -------
        Photon flux at earth from target region in (GeV cm^2 s sr)^{-1}
    """
    if e >= mx:
        return 0.
    else:
        dOmega = 2*np.pi*(1.-np.cos(th_max))
        J = J_factor(th_max, d, halo_params)  # GeV^2 / cm^5

        ret_val = dOmega/(4*np.pi) * J * sv / (2.*fx*mx**2) * \
            dn_de_gamma_AP(e, mx)

        return ret_val


def normalize_clump_dampe(d, halo_params, bg_flux, sv=3e-26, fx=2):
    """Get density normalization giving best fit to excess.

    Parameters
    ----------
    d : float
        Distance to the clump's center, kpc.
    bg_flux : float -> float
        Background flux in (GeV cm^2 s sr)^-1
    sv : float
        DM self-annihilation cross section, cm^3/s.

    Returns
    -------
    rhos : float
        Density normalization in GeV / cm^3.
    """
    mx = dampe_excess_bin_high

    # Residual integrated flux
    @cfunc(float64(float64))
    def bg_flux_cf(e):
        return bg_flux(e)

    bg_flux_LLC = LowLevelCallable(bg_flux_cf.ctypes)
    residual_iflux = dampe_excess_iflux - quad(bg_flux_LLC,
                                               dampe_excess_bin_low,
                                               dampe_excess_bin_high,
                                               epsabs=0,
                                               epsrel=1e-5)[0]

    # Integrated DM flux with density normalization set to 1
    if halo_params.__class__.__name__ == "NFW_params":
        dphi2_de_dr = dphi2_de_dr_nfw
        args = (d, halo_params.rs, 1., halo_params.gamma, mx, sv, fx)
    elif halo_params.__class__.__name__ == "TT_params":
        dphi2_de_dr = dphi2_de_dr_tt
        args = (d, halo_params.Rb, 1., halo_params.gamma, mx, sv, fx)

    # DM integrated flux
    dm_iflux = 2.*dblquad(dphi2_de_dr,
                          dampe_excess_bin_low, dampe_excess_bin_high,
                          lambda e: 0, lambda e: 100.*d,
                          args, epsabs=0)[0]

    # Set normalization in halo parameters
    rho_norm = np.sqrt(residual_iflux / dm_iflux)
    if halo_params.__class__.__name__ == "NFW_params":
        return NFW_params(halo_params.rs, rho_norm, halo_params.gamma)
    elif halo_params.__class__.__name__ == "TT_params":
        return TT_params(halo_params.Rb, rho_norm, halo_params.gamma)


def luminosity_dampe(d, halo_params, bg_flux, sv=3e-26, fx=2):
    mx = dampe_excess_bin_high
    halo_params = normalize_clump_dampe(d, halo_params, bg_flux, sv, fx)

    return luminosity(halo_params, mx, sv, fx)


def dphi_de_e_dampe(e, d, halo_params, bg_flux, sv=3e-26, fx=2):
    """Gets differential electron flux by fitting rho_s to the DAMPE excess.
    """
    mx = dampe_excess_bin_high
    halo_params = normalize_clump_dampe(d, halo_params, bg_flux, sv, fx)

    return dphi_de_e(e, d, halo_params, mx, sv, fx)


def dphi_de_gamma_dampe(e, th_max, d, halo_params, bg_flux, sv=3e-26, fx=2):
    """Gets differential photon flux by fitting rho_s to the DAMPE excess.
    """
    mx = dampe_excess_bin_high
    halo_params = normalize_clump_dampe(d, halo_params, bg_flux, sv, fx)

    return dphi_de_gamma(e, th_max, d, halo_params, mx, sv, fx)


def gamma_ray_extent(e, d, mx, halo_params, thresh=0.5, sv=3e-26, fx=2):
    """Computes the angular extent of the subhalo at a specific gamma ray
    energy.

    Parameters
    ----------
    e : float
        Gamma ray energy (GeV).
    d : float
        Distance to subhalo (kpc).
    mx : float
        DM mass (GeV).
    rs : float
        Subhalo scale radius (kpc).
    rhos : float
        Subhalo density normalization (GeV / cm^3).

    Notes
    -----
    Throws many ZeroDivisionError exceptions. It's unclear why, but the results
    seem reliable.

    Returns
    -------
    th_ext : float
        Radius of observing region such that the flux in the region is equal to
        thresh times the total flux.
    """
    # Compute flux integrating over the whole sky
    total_flux = dphi_de_gamma(e, np.pi, d, halo_params, mx, sv, fx)

    def fn(th):
        return (dphi_de_gamma(e, th, d, halo_params, mx, sv, fx) /
                total_flux - thresh)**2

    # The optimization will fail badly if rho_norm is nan for some reason
    if halo_params.__class__.__name__ == "NFW_params":
        rho_norm = halo_params.rhos
    elif halo_params.__class__.__name__ == "TT_params":
        rho_norm = halo_params.rho0

    if not np.isnan(rho_norm):
        return minimize_scalar(fn, bounds=(0, np.pi)).x
    else:
        return np.nan


def constrain_ep_spec(d, mx, halo_params, bg_flux, sv=3e-26, fx=2,
                      excluded_idxs=[]):
    def dm_flux(e):
        return dphi_de_e(e, d, halo_params, mx, sv, fx)

    return _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs)
