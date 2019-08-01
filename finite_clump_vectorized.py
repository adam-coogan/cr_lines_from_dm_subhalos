from numba import cfunc, jit
from numba.types import float64
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad, nquad
from scipy.optimize import minimize_scalar, brentq
from scipy.special import gamma as Gamma
from scipy.special import beta
from scipy.special import betainc

from dm_params import fx, sv
from background_models import Phi_e_bg, phi_e_bg_dampe, phi_e_bg_alt
from utilities import e_low_aniso_fermi, e_high_aniso_fermi, aniso_fermi
from utilities import kpc_to_cm, rho_critical, GeV_to_m_sun, fermi_psf
from utilities import speed_of_light, D, rho_max, lambert_w_series
from utilities import e_low_excess, e_high_excess
from utilities import Phi_excess, dn_de_g_ap
from utilities import bins_dampe, phis_dampe, phi_errs_dampe
from utilities import fermi_pt_src_sens_120_45 as fermi_pt_src_sens
from utilities import gamma_inc_upper

from nfw_clump import rho as rho_nfw
from nfw_clump import mass as mass_nfw
from nfw_clump import luminosity as luminosity_nfw
from nfw_clump import dphi_e_dr_llc as dphi_e_dr_nfw_llc
from nfw_clump import dJ_dr_llc as dJ_dr_nfw_llc

from exp_clump import rho as rho_exp
from exp_clump import mass as mass_exp
from exp_clump import luminosity as luminosity_exp
from exp_clump import dphi_e_dr_llc as dphi_e_dr_exp_llc
from exp_clump import dJ_dr_llc as dJ_dr_exp_llc
from exp_clump import ann_plateau_radius as ann_plateau_radius_exp


@np.vectorize
def get_points_near_far(dist, r_s):
    """Used to sample points closer and further than the clump's distance from
    Earth to make integrations more accurate.
    """
    n_near_pts = 20
    n_far_pts = 20
    if r_s / dist > 1:
        points_near = np.append(
            dist*(1 - np.logspace(-6, 0, n_near_pts)), dist)
    else:
        points_near = np.append(
            dist - np.logspace(-6, 0, n_near_pts) * r_s, dist)
    points_far = np.append(
            dist + np.logspace(np.log10(1e-6 * r_s), np.log10(9 * max(dist, r_s)), n_far_pts), dist)
    assert np.all(0 <= points_near)
    assert np.all(points_near <= dist)
    assert np.all(dist <= points_far)
    assert np.all(points_far <= 11 * max(dist, r_s))
    return points_near, points_far


def rho(dist, r_s, rho_s, gamma, halo):
    """Computes halo density.

    Parameters
    ----------
    dist : float
        Distance from halo center to point of interest (kpc). Must be positive.
    halo : string
        Halo name.

    Returns
    -------
    rho : float
        Halo density at specified point in GeV / cm^3.
    """
    if halo == "nfw":
        return rho_nfw(dist, r_s, rho_s, gamma)
    elif halo == "exp":
        return rho_exp(dist, r_s, rho_s, gamma)


def mass(r_s, rho_s, gamma, halo):
    """Computes total mass of halo.

    Parameters
    ----------
    halo : string
        Halo name.

    Returns
    -------
    M : float
        Total halo mass (for TT profile) or virial mass (for NFW profile)
        (number of solar masses)
    """
    if halo == "nfw":
        return mass_nfw(r_s, rho_s, gamma)
    elif halo == "exp":
        return mass_exp(r_s, rho_s, gamma)


def luminosity(r_s, rho_s, gamma, halo, mx=e_high_excess):
    """Computes the halo luminosity (Hz).
    """
    if halo == "nfw":
        return luminosity_nfw(r_s, rho_s, gamma, mx)
    elif halo == "exp":
        return luminosity_exp(r_s, rho_s, gamma, mx)


def lum_to_rho_norm(r_s, lum, gamma, halo):
    """Determines the halo's density normalization given its luminosity.

    Notes
    -----
    Assumes the luminosity is proportional to the density normalization
    squared.
    """
    return np.sqrt(lum / luminosity(r_s, 1., gamma, halo))


def phi_e(e, dist, r_s, rho_s, gamma, halo, mx=e_high_excess, epsrel=1e-5, limit=50):
    """Computes the differential flux phi|_{e-} for a DM clump.

    Parameters
    ----------
    e : float or float array
        Electron energies, GeV.
    d : float
        Distance to the clump's center, kpc.
    gamma : float
        NFW power index.
    """
    if halo == "nfw":
        dphi_e_dr = dphi_e_dr_nfw_llc
    elif halo == "exp":
        dphi_e_dr = dphi_e_dr_exp_llc

    @np.vectorize
    def _phi_e(e, dist, r_s, rho_s, gamma, epsrel=1e-5):
        if e > mx:
            return 0.
        else:  # perform integration over r
            args = (e, dist, r_s, rho_s, gamma, mx, sv, fx)
            # Split integral around center of clump
            points_near, points_far = get_points_near_far(dist, r_s)
            phi_e_near, err_near = quad(dphi_e_dr, 0, dist, args,
                                        points=points_near,
                                        epsabs=0, epsrel=epsrel, limit=limit)
            phi_e_far, err_far = quad(dphi_e_dr, dist, dist + 10*r_s, args,
                                      points=points_far,
                                      epsabs=0, epsrel=epsrel, limit=limit)
            return phi_e_near + phi_e_far

    return _phi_e(e, dist, r_s, rho_s, gamma)


def J_factor(dist, r_s, rho_s, gamma, halo, th_max):
    """Computes J factor for a target region.

    Notes
    -----
    Checked numerical stability for NFW

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
    if halo == "nfw":
        dJ_dr = dJ_dr_nfw_llc
    elif halo == "exp":
        dJ_dr = dJ_dr_exp_llc

    @np.vectorize
    def _J_factor(dist, r_s, rho_s, gamma):
        args = (th_max, dist, r_s, rho_s, gamma)
        # Split integral around center of clump
        points_near, points_far = get_points_near_far(dist, r_s)
        # Split integral around center of clump
        J_near, err_near = quad(dJ_dr, 0., dist, args, points=points_near, epsabs=0, epsrel=1e-5)
        J_far, err_far = quad(dJ_dr, dist, dist + 10*r_s, args, points=points_far, epsabs=0, epsrel=1e-5)
        return (J_near + J_far) * kpc_to_cm

    return _J_factor(dist, r_s, rho_s, gamma)


@np.vectorize
def phi_g(e, dist, r_s, rho_s, gamma, halo, th_max, mx=e_high_excess):
    """Computes the differential flux phi|_gamma for a DM clump.

    Parameters
    ----------
    e : list of floats
        Photon energies in GeV
    th_max : float
        Angular radius of observing region (rad)
    d : float
        Distance to clump center in kpc
    sv : float
        DM self-annihilation cross section in cm^3 / s
    fx : int
        1 if DM is self-conjugate, 2 if not.

    Returns
    -------
        Photon flux at earth from target region in (GeV cm^2 s sr)^{-1}
    """
    dOmega = 2*np.pi*(1.-np.cos(th_max))
    J = J_factor(dist, r_s, rho_s, gamma, halo, th_max)  # GeV^2 / cm^5
    dn_de_g = np.vectorize(dn_de_g_ap)(e, mx)
    return dOmega/(4*np.pi) * J * sv / (2.*fx*mx**2) * dn_de_g


def rho_s_dampe(dist, r_s, gamma, halo, mx=e_high_excess, bg_model="dampe"):
    """Get density normalization giving best fit to excess.

    Returns
    -------
    rho_s : float
        Density normalization in GeV / cm^3.
    """
    if halo == "nfw":
        dphi_e_dr = dphi_e_dr_nfw_llc
    elif halo == "exp":
        dphi_e_dr = dphi_e_dr_exp_llc

    # Residual integrated flux
    Phi_bg = Phi_e_bg(e_low_excess, e_high_excess, bg_model)
    Phi_e_residual = Phi_excess - Phi_bg

    @np.vectorize
    def _rho_s_dampe(dist, r_s, gamma):
        # Set rho_s to 1 and use that the flux is proportional to rho_s**2
        points_e = e_high_excess * (1 - np.logspace(-6, -1, 10))
        # Improve numerical stability by splitting the spatial integral
        args = (dist, r_s, 1., gamma, mx, sv, fx)
        points_near, points_far = get_points_near_far(dist, r_s)
        Phi_e_near, err_near = nquad(
            dphi_e_dr, args=args, ranges=[(0, dist), (e_low_excess, e_high_excess)],
            opts=[{"epsabs": 0, "epsrel": 1e-5, "points": points_near},
                  {"epsabs": 0, "epsrel": 1e-5, "points": points_e}])
        Phi_e_far, err_far = nquad(
            dphi_e_dr, args=args, ranges=[(dist, dist + 10*r_s), (e_low_excess, e_high_excess)],
            opts=[{"epsabs": 0, "epsrel": 1e-5, "points": points_far},
                  {"epsabs": 0, "epsrel": 1e-5, "points": points_e}])
        # Factor of 2 accounts for DAMPE measuring e+ and e-
        return np.sqrt(Phi_e_residual / (2*Phi_e_near + 2*Phi_e_far))

    return _rho_s_dampe(dist, r_s, gamma)


@np.vectorize
def gamma_ray_extent(dist, r_s, rho_s, gamma, halo, e, thresh=0.5):
    """Computes the angular extent of the subhalo at a specific gamma ray
    energy.

    Parameters
    ----------
    e : float
        Gamma ray energy (GeV).
    d : float
        Distance to subhalo (kpc).
    r_s : float
        Subhalo scale radius (kpc).
    rho_s : float
        Subhalo density normalization (GeV / cm^3).

    Returns
    -------
    th_ext : float
        Radius of observing region such that the flux in the region is equal to
        thresh times the total flux.
    """
    # Compute flux integrating over the whole sky
    phi_g_tot = phi_g(e, dist, r_s, rho_s, gamma, halo, np.pi)

    @np.vectorize
    def loss(log10_th):
        return (phi_g(e, dist, r_s, rho_s, gamma, halo, 10.**log10_th) / phi_g_tot - thresh)**2

    bracket_low = np.log10(1e-6 * fermi_psf)
    bracket_high = np.log10(np.pi)
    # Hacky but effective way of finding a bracketing interval
    log10_ths = np.linspace(bracket_low, bracket_high, 50)
    losses = loss(log10_ths)
    bracket_middle = log10_ths[np.nanargmin(losses)]
    # Make sure loss at bracket endpoints is defined
    bracket_low = log10_ths[np.where(~np.isnan(losses))[0]][0]
    bracket_high = log10_ths[np.where(~np.isnan(losses))[0]][-1]

    # Do not optimize if rho_s is nan
    if not np.isnan(rho_s):
        try:
            log10_th = minimize_scalar(
                loss, bracket=(bracket_low, bracket_middle, bracket_high),
                bounds=(fermi_psf, np.pi)).x
            return 10.**log10_th
        except:
            print("Not a bracketing interval")
            print(dist, r_s, loss(bracket_low), loss(bracket_high))
            return np.nan
    else:
        return np.nan


def line_width_constraint(dist, r_s, rho_s, gamma, halo, mx=e_high_excess,
                          n_sigma=3., bg_model="dampe", excluded_idxs=[]):
    """Returns significance of largest excess in a DAMPE bin aside from the one
    with the true excess.
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

    # Compute integrated flux in each bin
    Phi_residual = []
    Phi_errs = []
    for (e_low, e_high), phi, err in zip(bins, phis, phi_errs):
        Phi_dampe = (e_high - e_low) * phi
        Phi_bg = Phi_e_bg(e_low, e_high, bg_model)
        Phi_residual.append(Phi_dampe - Phi_bg)
        Phi_errs.append((e_high - e_low) * err)

    @np.vectorize
    def _line_width_constraint(dist, r_s, rho_s, gamma):
        args = (dist, r_s, rho_s, gamma, halo, mx)
        n_sigma_max = 0.
        for (e_low, e_high), Phi_res, Phi_err in zip(bins, Phi_residual, Phi_errs):
            # Factor of 2 is needed because DAMPE measures e+ and e-
            # The integrand is not sharply peaked outside the bin with the
            # excess, so we don't need to set `points`.
            Phi_clump = 2 * quad(phi_e, e_low, e_high, args, epsabs=0, epsrel=1e-5)[0]
            # Determine significance of DM contribution
            n_sigma_bin = (Phi_clump - Phi_res) / Phi_err
            n_sigma_max = max(n_sigma_bin, n_sigma_max)
            if n_sigma_max >= n_sigma:  # stop if threshold was exceeded
                return n_sigma_max

        return n_sigma_max

    return _line_width_constraint(dist, r_s, rho_s, gamma)


def line_width_constraint_chi2(dist, r_s, rho_s, gamma, halo, mx=e_high_excess,
                               bg_model="dampe", excluded_idxs=[]):
    """Returns significance of largest excess in a DAMPE bin aside from the one
    with the true excess.
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

    # Compute integrated flux in each bin
    Phi_residual = []
    Phi_errs = []
    for (e_low, e_high), phi, err in zip(bins, phis, phi_errs):
        Phi_dampe = (e_high - e_low) * phi
        Phi_bg = Phi_e_bg(e_low, e_high, bg_model)
        Phi_residual.append(Phi_dampe - Phi_bg)
        Phi_errs.append((e_high - e_low) * err)

    @np.vectorize
    def _line_width_constraint(dist, r_s, rho_s, gamma):
        args = (dist, r_s, rho_s, gamma, halo, mx)

        def bin_chi2(e_bin, Phi_res, Phi_err):
            # Factor of 2 is needed because DAMPE measures e+ and e-
            # The integrand is not sharply peaked outside the bin with the
            # excess, so we don't need to set `points`.
            e_low, e_high = e_bin
            Phi_clump = 2 * quad(phi_e, e_low, e_high, args, epsabs=0, epsrel=1e-5)[0]
            # Determine significance of DM contribution
            n_sigma = (Phi_clump - Phi_res) / Phi_err
            return n_sigma**2

        return sum([bin_chi2(e_bin, Phi_res, Phi_err)
                    for (e_bin, Phi_res, Phi_err) in zip(bins, Phi_residual, Phi_errs)])

    return _line_width_constraint(dist, r_s, rho_s, gamma)


def fermi_point_src_contraint(dist, r_s, gamma, halo, e_star=230.):
    """Computes the maximum halo density normalization consistent with Fermi's
    non-observation of point sources coming from DM clumps.

    Parameters
    ----------
    e_star : float
        This is the energy at which the photon spectrum from the clump will
        first touch Fermi's broadband point-source sensitivity curve. Since the
        mass and spectrum are fixed through our analysis, the default value
        applies for all clump parameters.

    Returns
    -------
    rho_s : float
        The maximum density normalization allowed by Fermi point source bounds.
    """
    phi_g_clump = phi_g(e_star, dist, r_s, 1, gamma, halo, fermi_psf)
    phi_g_sens = fermi_pt_src_sens(e_star)
    return np.sqrt(phi_g_sens / phi_g_clump)


@np.vectorize
def anisotropy_differential(e, dist, r_s, rho_s, gamma, halo, mx=e_high_excess,
                            delta_d_rel=0.001, bg_model="dampe"):
    # Compute derivative with respect to distance numerically
    if bg_model == "dampe":
        phi_e_bg = phi_e_bg_dampe(e)
    elif bg_model == "alt":
        phi_e_bg = phi_e_bg_alt(e)
    phi_e_d = 2 * phi_e(e, dist, r_s, rho_s, gamma, halo, mx, epsrel=1e-3*delta_d_rel)

    delta_d = delta_d_rel * dist
    phi_e_d_dd = 2 * phi_e(e, dist + delta_d, r_s, rho_s, gamma, halo, mx, epsrel=1e-3*delta_d_rel)

    dphi_e_dd = np.abs((phi_e_d_dd - phi_e_d) / delta_d)
    phi_e_tot = phi_e_d + phi_e_bg

    return 3 * D(e) / speed_of_light * dphi_e_dd / kpc_to_cm / phi_e_tot


@np.vectorize
def anisotropy_integrated(e_low, e_high, dist, r_s, rho_s, gamma, halo,
                          mx=e_high_excess, delta_d_rel=0.001, bg_model="dampe"):
    """Bin-averaged anisotropy."""
    points_e = np.clip(mx * (1 - np.logspace(-6, -1, 10)), e_low, e_high)
    args = (dist, r_s, rho_s, gamma, halo, delta_d_rel, bg_model)
    return quad(anisotropy_differential, e_low, e_high, points=points_e,
                args=args, epsabs=0, epsrel=1e-5)[0] / (e_high - e_low)
