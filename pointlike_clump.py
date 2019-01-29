import numpy as np
from numba import cfunc, jit
from numba.types import float64
from scipy.integrate import quad
from scipy import LowLevelCallable

from constants import kpc_to_cm, lambda_prop, b, speed_of_light
from constants import dn_de_gamma_AP
from constants import dampe_excess_bin_low, dampe_excess_bin_high
from constants import dampe_excess_iflux, _constrain_ep_spec
from background_models import bg_dampe


def lum_dampe_pt(d, bg_flux=bg_dampe):
    """Returns the luminosity such that xx->e+e- fits the DAMPE excess.

    Returns
    -------
    L : float
        Luminosity in s^-1
    """
    mx = dampe_excess_bin_high
    @cfunc(float64(float64))
    def bg_flux_cf(e):
        return bg_flux(e)
    bg_flux_LLC = LowLevelCallable(bg_flux_cf.ctypes)
    bg_iflux = quad(bg_flux_LLC, dampe_excess_bin_low, dampe_excess_bin_high, epsabs=0.)[0]

    def helper(d):
        @cfunc(float64(float64))
        def dm_flux_cf(e):
            return dphi_de_e_pt(e, d, mx, lum=1.)
        dm_flux_LLC = LowLevelCallable(dm_flux_cf.ctypes)
        # Factor of 2 to count e+ and e-
        dm_iflux = 2 * quad(dm_flux_LLC, dampe_excess_bin_low,
                            dampe_excess_bin_high, points=[mx], epsabs=0)[0]
        residual_iflux = dampe_excess_iflux - bg_iflux
        return residual_iflux / dm_iflux

    return np.vectorize(helper)(d)


@jit(nopython=True)
def dphi_de_e_pt(e, d, mx, lum=None):
    """Flux of e- from a point-like DM clump after propagation.

    Parameters
    ----------
    e : float
        e- energy (GeV).
    d : float
        Distance to clump (kpc).
    mx : float
        DM mass in GeV
    lum : float
        Clump luminosity (s^-1).

    Returns
    -------
    dphi_de_e : numpy array
        e- flux in (GeV cm^2 s sr)^-1
    """
    d_cm = kpc_to_cm * d  # convert distance to cm

    if e < mx:
        lam = lambda_prop(e, mx)

        return 1. / b(e) * np.exp(-d_cm**2 / (4.*lam)) / \
            (4.*np.pi*lam)**1.5 * lum * speed_of_light/(4*np.pi)
    else:
        return 0


@jit(nopython=True)
def dphi_de_gamma_pt(e, d, mx, lum):
    """Photon flux from FSR for DM annihilating into e+ e-

    To-do
    -----
    Make function to get integrated flux

    Parameters
    ----------
    e : numpy array
        e- energy (GeV).
    d : float
        Distance to clump (kpc).
    mx : float
        DM mass in GeV
    lum : float
        Clump luminosity (s^-1).

    Returns
    -------
    dphi_de_gamma : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    d_cm = kpc_to_cm * d

    return lum/(4.*np.pi*d_cm**2) * dn_de_gamma_AP(e, mx)


def dphi_de_e_dampe_pt(e, d, bg_flux):
    """e+/e- flux, with DM mass and clump luminosity set to fit DAMPE excess.

    Parameters
    ----------
    e : numpy array
        e- energy (GeV).
    d : float
        Distance to clump (kpc).
    bg_flux : float -> float
        Background flux in (GeV cm^2 s sr)^-1

    Returns
    -------
    dphi_de_e : numpy array
        e+/e- flux in (GeV cm^2 s sr)^-1
    """
    mx = dampe_excess_bin_high
    lum = lum_dampe_pt(d, bg_flux)

    return np.vectorize(dphi_de_e_pt)(e, d, mx, lum)


def dphi_de_gamma_dampe_pt(e, d, bg_flux):
    """Photon flux, with DM mass and clump luminosity set to fit DAMPE excess.

    Parameters
    ----------
    e : numpy array
        Photon energy (GeV).
    d : float
        Distance to clump (kpc).
    bg_flux : float -> float
        Background flux in (GeV cm^2 s sr)^-1

    Returns
    -------
    dphi_de_gamma : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    mx = dampe_excess_bin_high
    lum = lum_dampe_pt(d, bg_flux)

    return np.vectorize(dphi_de_gamma_pt)(e, d, mx, lum)


def constrain_ep_spec_pt(d, mx, lum, bg_flux, excluded_idxs=[]):
    def dm_flux(e):
        return dphi_de_e_pt(e, d, mx, lum)

    return _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs)
