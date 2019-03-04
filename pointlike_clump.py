import numpy as np
from numba import cfunc, jit
from numba.types import float64
from scipy.integrate import quad
from scipy import LowLevelCallable

from background_models import Phi_e_bg, phi_e_bg_dampe, phi_e_bg_alt
from dm_params import mx, fx, sv
from utilities import kpc_to_cm, lambda_prop, b, speed_of_light, dn_de_g_ap
from utilities import e_low_excess, e_high_excess, D
from utilities import Phi_excess, bins_dampe, phis_dampe, phi_errs_dampe


def lum_dampe(dist, bg_model="dampe"):
    """Returns the luminosity such that xx->e+e- fits the DAMPE excess.

    Returns
    -------
    L : float
        Luminosity in s^-1
    """
    Phi_bg = Phi_e_bg(e_low_excess, e_high_excess, bg_model)

    @np.vectorize
    def _lum_dampe(dist):
        # Factor of 2 to count e+ and e-
        # Set lum to 1 and use that the flux is proportional to lum**2
        points_e = e_high_excess * (1 - np.logspace(-6, -1, 10))
        Phi_clump = 2 * quad(phi_e, e_low_excess, e_high_excess, args=(dist, 1.),
                             points=points_e, epsabs=0, epsrel=1e-5)[0]
        return (Phi_excess - Phi_bg) / Phi_clump

    return _lum_dampe(dist)


@np.vectorize
def phi_e(e, dist, lum):
    """Flux of e- from a point-like DM clump after propagation.

    Parameters
    ----------
    e : float
        e- energy (GeV).
    dist : float
        Distance to clump (kpc).
    lum : float
        Clump luminosity (s^-1).

    Returns
    -------
    phi_e : numpy array
        e- flux in (GeV cm^2 s sr)^-1
    """
    d_cm = kpc_to_cm * dist

    if e < mx:
        lam = lambda_prop(e, mx)
        return 1. / b(e) * np.exp(-d_cm**2 / (4.*lam)) / (4.*np.pi*lam)**1.5 * lum * speed_of_light/(4*np.pi)
    else:
        return 0


#@jit(nopython=True)
@np.vectorize
def phi_g(e, dist, lum):
    """Photon flux from FSR for DM annihilating into e+ e-

    To-do
    -----
    Make function to get integrated flux

    Parameters
    ----------
    e : numpy array
        e- energy (GeV).
    dist : float
        Distance to clump (kpc).
    lum : float
        Clump luminosity (s^-1).

    Returns
    -------
    phi_g : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    d_cm = kpc_to_cm * dist
    return lum/(4.*np.pi*d_cm**2) * dn_de_g_ap(e, mx)


def phi_e_dampe(e, dist):
    """e+/e- flux, with DM mass and clump luminosity set to fit DAMPE excess.

    Parameters
    ----------
    e : numpy array
        e- energy (GeV).
    dist : float
        Distance to clump (kpc).

    Returns
    -------
    dphi_de_e : numpy array
        e+/e- flux in (GeV cm^2 s sr)^-1
    """
    lum = lum_dampe(dist)
    return phi_e(e, dist, lum)


def phi_g_dampe(e, dist):
    """Photon flux, with DM mass and clump luminosity set to fit DAMPE excess.

    Parameters
    ----------
    e : numpy array
        Photon energy (GeV).
    dist : float
        Distance to clump (kpc).

    Returns
    -------
    phi_g : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    lum = lum_dampe(dist)
    return phi_g(e, dist, lum)


def line_width_constraint(dist, lum, n_sigma=3., bg_model="dampe", excluded_idxs=[]):
    """Returns significance of largest excess in a DAMPE bin aside from the one
    with the true excess. Assumes the clump can be treated as a point source.
    """
    idxs = set(range(len(bins_dampe))) - set(excluded_idxs)
    # Get index of bin containing excess
    excess_bin_idx = np.where([e_low_excess, e_high_excess] == bins_dampe)[0][0]
    # Ignore all bins above this -- since m_DM = e_high_excess, the flux is
    # zero in all bins above this, and known in this bin.
    idxs = idxs - set(range(excess_bin_idx, bins_dampe.shape[0]))
    # Reverse the list since constraint is likely to be set by bin closest to
    # the excess.
    idxs = sorted(list(idxs))
    idxs.reverse()

    # Select the bins that were not excluded
    bins = bins_dampe[idxs]
    phis = phis_dampe[idxs]
    phi_errs = phi_errs_dampe[idxs]

    Phi_residual = []
    Phi_errs = []
    for (e_low, e_high), phi, err in zip(bins, phis, phi_errs):
        Phi_dampe = (e_high - e_low) * phi
        Phi_bg = Phi_e_bg(e_low, e_high, bg_model)
        Phi_residual.append(Phi_dampe - Phi_bg)
        Phi_errs.append((e_high - e_low) * err)

    @np.vectorize
    def _line_width_constraint(dist, lum):
        n_sigma_max = 0.
        for (e_low, e_high), Phi_res, Phi_err in zip(bins, Phi_residual, Phi_errs):
            # Factor of 2 is needed because DAMPE measures e+ and e-
            points_e = np.logspace(np.log10(e_low), np.log10(e_high), 10)
            Phi_clump = 2.*quad(phi_e, e_low, e_high, args=(dist, lum),
                                points=points_e, epsabs=0, epsrel=1e-5)[0]
            # Determine significance of DM contribution
            n_sigma_bin = (Phi_clump - Phi_res) / Phi_err
            n_sigma_max = max(n_sigma_bin, n_sigma_max)
            if n_sigma_max >= n_sigma:  # stop if threshold was exceeded
                return n_sigma_max

        return n_sigma_max

    return _line_width_constraint(dist, lum)


@np.vectorize
def anisotropy_differential(e, dist, lum):
    if e < mx:
        phi_ep_clump = 2*phi_e(e, dist, lum)
        phi_ep_tot = phi_ep_clump + phi_e_bg_dampe(e)
        return np.abs((3*D(e)/speed_of_light * 2*dist/lambda_prop(e, mx) *
                       phi_ep_clump * kpc_to_cm / phi_ep_tot))
    else:
        return 0.


@np.vectorize
def anisotropy_integrated(e_low, e_high, dist, lum):
    points_e = mx * (1 - np.logspace(-6, -1, 10))
    return quad(anisotropy_differential, e_low, e_high, args=(dist, lum),
                points=points_e, epsabs=1e-99)[0] / (e_high - e_low)
