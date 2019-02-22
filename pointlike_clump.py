import numpy as np
from numba import cfunc, jit
from numba.types import float64
from scipy.integrate import quad
from scipy import LowLevelCallable

from background_models import Phi_e_bg, phi_e_bg_dampe, phi_e_bg_alt
from dm_params import mx, fx, sv
from utilities import kpc_to_cm, lambda_prop, b, speed_of_light, dn_de_g_ap
from utilities import e_low_excess, e_high_excess
from utilities import Phi_excess, bins_dampe, phis_dampe, phi_errs_dampe


def lum_dampe(d, bg_model="dampe"):
    """Returns the luminosity such that xx->e+e- fits the DAMPE excess.

    Returns
    -------
    L : float
        Luminosity in s^-1
    """
    Phi_bg = Phi_e_bg(e_low_excess, e_high_excess, bg_model)

    @np.vectorize
    def _lum_dampe(d):
        # Factor of 2 to count e+ and e-
        # Set lum to 1 and use that the flux is proportional to lum**2
        points_e = e_high_excess * (1 - np.logspace(-6, -1, 10))
        Phi_clump = 2 * quad(phi_e, e_low_excess, e_high_excess, args=(d, 1.),
                             points=points_e, epsabs=0, epsrel=1e-5)[0]
        return (Phi_excess - Phi_bg) / Phi_clump

    return _lum_dampe(d)


@np.vectorize
def phi_e(e, d, lum):
    """Flux of e- from a point-like DM clump after propagation.

    Parameters
    ----------
    e : float
        e- energy (GeV).
    d : float
        Distance to clump (kpc).
    lum : float
        Clump luminosity (s^-1).

    Returns
    -------
    phi_e : numpy array
        e- flux in (GeV cm^2 s sr)^-1
    """
    d_cm = kpc_to_cm * d

    if e < mx:
        lam = lambda_prop(e, mx)
        return 1. / b(e) * np.exp(-d_cm**2 / (4.*lam)) / (4.*np.pi*lam)**1.5 * lum * speed_of_light/(4*np.pi)
    else:
        return 0


#@jit(nopython=True)
@np.vectorize
def phi_g(e, d, lum):
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
    lum : float
        Clump luminosity (s^-1).

    Returns
    -------
    phi_g : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    d_cm = kpc_to_cm * d
    return lum/(4.*np.pi*d_cm**2) * dn_de_g_ap(e, mx)


def phi_e_dampe(e, d):
    """e+/e- flux, with DM mass and clump luminosity set to fit DAMPE excess.

    Parameters
    ----------
    e : numpy array
        e- energy (GeV).
    d : float
        Distance to clump (kpc).

    Returns
    -------
    dphi_de_e : numpy array
        e+/e- flux in (GeV cm^2 s sr)^-1
    """
    lum = lum_dampe(d)
    return phi_e(e, d, lum)


def phi_g_dampe(e, d):
    """Photon flux, with DM mass and clump luminosity set to fit DAMPE excess.

    Parameters
    ----------
    e : numpy array
        Photon energy (GeV).
    d : float
        Distance to clump (kpc).

    Returns
    -------
    phi_g : numpy array
        Photon flux in (GeV cm^2 s sr)^-1
    """
    lum = lum_dampe(d)
    return phi_g(e, d, lum)


def line_width_constraint(dist, lum, n_sigma=3., bg_model="dampe", excluded_idxs=[]):
    """Returns significance of largest excess in a DAMPE bin aside from the one
    with the true excess. Assumes the clump can be treated as a point source.
    """
    # Get index of bin containing excess
    excess_bin_idx = np.where([e_low_excess, e_high_excess] == bins_dampe)[0][0]
    # Ignore all bins above this -- since m_DM = e_high_excess, the flux is
    # zero in all bins above this, and known in this bin.
    idxs = idxs - set(range(excess_bin_idx, bins_dampe.shape[0]))
    # Reverse the list since constraint is likely to be set by bin closest to
    # the excess.
    idxs = set(range(len(bins_dampe))) - set(excluded_idxs)
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
            print(e_low, e_high)
            Phi_clump = 2.*quad(phi_e, e_low, e_high, args=(dist, lum),
                                points=points_e, epsabs=0, epsrel=1e-5)[0]
            # Determine significance of DM contribution
            n_sigma_bin = (Phi_clump - Phi_res) / Phi_err
            n_sigma_max = max(n_sigma_bin, n_sigma_max)
            if n_sigma_max >= n_sigma:  # stop if threshold was exceeded
                return n_sigma_max

        return n_sigma_max

    return _line_width_constraint(dist, lum)


# def _constrain_ep_spec(dm_flux, excluded_idxs=[], debug_msgs=False):
#     """Determines the significance of the e-+e+ flux from DM annihilation in
#     other bins.
#
#     Parameters
#     ----------
#     dm_flux : float -> float
#         A function returning the e-+e+ flux from DM annihilation as a function
#         of the lepton's energy in GeV.
#     bg_flux : float -> float
#         A function giving the background flux.
#     excluded_idxs : list of ints
#         A list specifying indices of bins to ignore. This is useful because Ge
#         et al treat several other bins as also having an excess from DM
#         annihilating, and thus exclude them from the background fit.
#
#     Returns
#     -------
#     n_sigma_max : float
#
#         The statistical significance for the DM e-+e+ flux for the bin with the
#         most significant excess from DM.
#     """
#     def dm_iflux(e_low, e_high):
#         return quad(lambda e: 2. * dm_flux(e), e_low, e_high, epsabs=0)[0]
#
#     idxs = list(set(range(len(dampe_bins))) - set(excluded_idxs))
#     n_sigma_max = 0.
#
#     # Track bin containing largest excess
#     e_low_max, e_high_max = np.nan, np.nan

#
#     for (e_low, e_high), flux, err in reversed(zip(dampe_bins[idxs],
#                                                    dampe_dflux[idxs],
#                                                    dampe_dflux_err[idxs])):
#         if e_low < e_high_excess:
#             # Residual flux
#             obs_iflux = (e_high - e_low) * flux
#             bg_iflux = quad(bg_flux, e_low, e_high, epsabs=0.)[0]
#             residual_iflux = obs_iflux - bg_iflux
#
#             # Error on integrated flux
#             obs_iflux_err = (e_high - e_low) * err
#
#             # Compare with flux from DM annihilations in the bin
#             n_sigma_bin = (dm_iflux(e_low, e_high) - residual_iflux) / \
#                 obs_iflux_err
#
#             if n_sigma_bin > n_sigma_max:
#                 n_sigma_max = n_sigma_bin
#                 e_low_max, e_high_max = e_low, e_high
#
#     if debug_msgs:
#         print("Bin with largest excess: ", e_low_max, e_high_max)
#
#     return n_sigma_max
#
#
# def constrain_ep_spec_pt(d, mx, lum, bg_flux, excluded_idxs=[]):
#     def dm_flux(e):
#         return phi_e(e, d, mx, lum)
#
#     return _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs)
