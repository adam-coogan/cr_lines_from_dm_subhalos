import numpy as np
from numba import cfunc, jit
from numba.types import float64
from scipy.integrate import quad
from scipy import LowLevelCallable

from utilities import kpc_to_cm, lambda_prop, b, speed_of_light
from utilities import dn_de_g_ap
from utilities import e_low_excess, e_high_excess
from utilities import Phi_excess
from background_models import Phi_e_bg, phi_e_bg_dampe, phi_e_bg_alt


def lum_dampe_pt(d):
    """Returns the luminosity such that xx->e+e- fits the DAMPE excess.

    Returns
    -------
    L : float
        Luminosity in s^-1
    """
    bg_iflux = Phi_e_bg(e_low_excess, e_high_excess)
    mx = e_high_excess

    def helper(d):
        @cfunc(float64(float64))
        def dm_flux_cf(e):
            return dphi_de_e_pt(e, d, mx, lum=1.)
        dm_flux_LLC = LowLevelCallable(dm_flux_cf.ctypes)
        # Factor of 2 to count e+ and e-
        dm_iflux = 2 * quad(dm_flux_LLC, e_low_excess,
                            e_high_excess, points=[mx], epsabs=0)[0]
        residual_iflux = Phi_excess - bg_iflux
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
def phi_g_pt(e, d, mx, lum):
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
    phi_g : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    mx = e_high_excess
    d_cm = kpc_to_cm * d

    return lum/(4.*np.pi*d_cm**2) * dn_de_g_ap(e, mx)


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
    mx = e_high_excess
    lum = lum_dampe_pt(d)
    return np.vectorize(dphi_de_e_pt)(e, d, mx, lum)


def phi_g_dampe_pt(e, d, bg_flux):
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
    phi_g : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    mx = e_high_excess
    lum = lum_dampe_pt(d)
    return np.vectorize(phi_g_pt)(e, d, mx, lum)


def _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs=[], debug_msgs=False):
    """Determines the significance of the e-+e+ flux from DM annihilation in
    other bins.

    Parameters
    ----------
    dm_flux : float -> float
        A function returning the e-+e+ flux from DM annihilation as a function
        of the lepton's energy in GeV.
    bg_flux : float -> float
        A function giving the background flux.
    excluded_idxs : list of ints
        A list specifying indices of bins to ignore. This is useful because Ge
        et al treat several other bins as also having an excess from DM
        annihilating, and thus exclude them from the background fit.

    Returns
    -------
    n_sigma_max : float

        The statistical significance for the DM e-+e+ flux for the bin with the
        most significant excess from DM.
    """
    def dm_iflux(e_low, e_high):
        return quad(lambda e: 2. * dm_flux(e), e_low, e_high, epsabs=0)[0]

    idxs = list(set(range(len(dampe_bins))) - set(excluded_idxs))
    n_sigma_max = 0.

    # Track bin containing largest excess
    e_low_max, e_high_max = np.nan, np.nan

    for (e_low, e_high), flux, err in reversed(zip(dampe_bins[idxs],
                                                   dampe_dflux[idxs],
                                                   dampe_dflux_err[idxs])):
        if e_low < e_high_excess:
            # Residual flux
            obs_iflux = (e_high - e_low) * flux
            bg_iflux = quad(bg_flux, e_low, e_high, epsabs=0.)[0]
            residual_iflux = obs_iflux - bg_iflux

            # Error on integrated flux
            obs_iflux_err = (e_high - e_low) * err

            # Compare with flux from DM annihilations in the bin
            n_sigma_bin = (dm_iflux(e_low, e_high) - residual_iflux) / \
                obs_iflux_err

            if n_sigma_bin > n_sigma_max:
                n_sigma_max = n_sigma_bin
                e_low_max, e_high_max = e_low, e_high

    if debug_msgs:
        print("Bin with largest excess: ", e_low_max, e_high_max)

    return n_sigma_max


def constrain_ep_spec_pt(d, mx, lum, bg_flux, excluded_idxs=[]):
    def dm_flux(e):
        return dphi_de_e_pt(e, d, mx, lum)

    return _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs)
