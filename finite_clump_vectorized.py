from numba import cfunc, jit
from numba.types import float64
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar, brentq
from scipy.special import gamma as Gamma
from scipy.special import beta
from scipy.special import betainc

from background_models import bg_dampe
from constants import e_low_aniso_fermi, e_high_aniso_fermi, aniso_fermi
from constants import kpc_to_cm, rho_critical, GeV_to_m_sun, fermi_psf
from constants import speed_of_light, D, rho_max, lambert_w_series
from constants import dampe_excess_bin_low, dampe_excess_bin_high
from constants import dampe_excess_iflux, dn_de_gamma_AP
from constants import dampe_bins, dampe_dflux, dampe_dflux_err
from constants import fermi_pt_src_sens_120_45 as fermi_pt_src_sens
from constants import gamma_inc_upper
from nfw_clump import dphi2_de_dr as dphi2_de_dr_nfw
from nfw_clump import dJ_dr as dJ_dr_nfw
from nfw_clump import dphi3_de_dr_dd as dphi3_de_dr_dd_nfw
from tt_clump import dphi2_de_dr as dphi2_de_dr_tt
from tt_clump import dJ_dr as dJ_dr_tt
from tt_clump import dphi3_de_dr_dd as dphi3_de_dr_dd_exp
from tt_clump import ann_plateau_radius as ann_plateau_radius_exp


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
        def _rho(dist, r_s, rho_s, gamma):
            return rho_s * (r_s / dist)**gamma * \
                    (1. + dist / r_s)**(gamma - 3.)
    elif halo == "exp":
        def _rho(dist, r_s, rho_s, gamma):
            r_p = ann_plateau_radius_exp(r_s, rho_s, gamma)
            if dist < r_p:
                return rho_max
            else:
                return rho_s * (r_s / dist)**gamma * np.exp(-dist / r_s)

    return np.vectorize(_rho)(dist, r_s, rho_s, gamma)


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
        def _mass(r_s, rho_s, gamma):
            # Find virial mass numerically. The "magic number" of 10000 should
            # be fine for this project.
            try:
                def _r_vir_integrand(r):
                    return (rho(r, r_s, rho_s, gamma, halo) -
                            200. * rho_critical)

                r_vir = brentq(_r_vir_integrand, 0.01 * r_s, 100. * r_s,
                               xtol=1e-200)
            except RuntimeError:
                r_vir = np.nan

            # Integrate numerically: analytic result has an imaginary part
            factor = 4.*np.pi * GeV_to_m_sun * kpc_to_cm**3

            def _mass_integrand(r):
                return r**2 * rho(r, r_s, rho_s, gamma, halo)

            return factor * quad(_mass_integrand, 0, r_vir, epsabs=0,
                                 epsrel=1e-5)[0]
    elif halo == "exp":
        def _mass(r_s, rho_s, gamma):
            return (4. * np.pi * rho_s * (r_s * kpc_to_cm)**3 *
                    Gamma(3. - gamma)) * GeV_to_m_sun

    return np.vectorize(_mass)(r_s, rho_s, gamma)


def luminosity(r_s, rho_s, gamma, halo, mx=dampe_excess_bin_high, sv=3e-26,
               fx=1.):
    """Computes the halo luminosity (Hz).
    """
    factor = sv / (2*fx*mx**2)

    if halo == "nfw":
        def _luminosity(r_s, rho_s, gamma):
            rs_cm = kpc_to_cm * r_s
            L_clump = ((4*np.pi*rho_s**2*rs_cm**3) /
                       (30 - 47*gamma + 24*gamma**2 - 4*gamma**3))
            return factor * L_clump
    elif halo == "exp":
        def _luminosity(r_s, rho_s, gamma):
            # No annihilation plateau
            Rb_cm = kpc_to_cm * r_s
            L_clump = (2**(-1 + 2*gamma) * np.pi * Rb_cm**3 * rho_s**2 *
                       Gamma(3 - 2*gamma))
            return factor * L_clump
            # With annihilation plateau
            # Rb_cm = kpc_to_cm * r_s
            # r_p_cm = kpc_to_cm * ann_plateau_radius_exp(r_s, rho_s, gamma)
            # return np.pi/6. * factor * (3. * 4.**gamma * rho_s**2 * Rb_cm**3 *
            #                             gamma_inc_upper(3. - 2. * gamma,
            #                                             2. * r_p_cm / Rb_cm) +
            #                             8. * rho_max**2 * r_p_cm**3)

    return np.vectorize(_luminosity)(r_s, rho_s, gamma)


def lum_to_rho_norm(r_s, lum, gamma, halo, mx=dampe_excess_bin_high, sv=3e-26,
                    fx=1.):
    """Determines the halo's density normalization given its luminosity.

    Notes
    -----
    Assumes the luminosity is proportional to the density normalization
    squared.
    """
    return np.sqrt(lum / luminosity(r_s, 1., gamma, halo, mx, sv, fx))


def dphi_de_e(e, dist, r_s, rho_s, gamma, halo, mx=dampe_excess_bin_high,
              sv=3e-26, fx=1.):
    """Computes dphi/dE|_{e-} for a DM clump.

    Parameters
    ----------
    e : float or float array
        Electron energies, GeV.
    d : float
        Distance to the clump's center, kpc.
    mx : float
        DM mass, GeV.
    sv : float
        DM self-annihilation cross section, cm^3/s.
    gamma : float
        NFW power index.
    """
    if halo == "nfw":
        dphi2_de_dr = dphi2_de_dr_nfw
    elif halo == "exp":
        dphi2_de_dr = dphi2_de_dr_tt

    def _dphi_de_e(e, dist, r_s, rho_s, gamma):
        if e > mx:
            return 0.
        else:  # perform integration over r
            args = (e, dist, r_s, rho_s, gamma, mx, sv, fx)
            # Split integral around center of clump
            int_near, err_near = quad(dphi2_de_dr, 0, dist, args, points=[dist],
                                      epsabs=0, epsrel=1e-5)
            int_far, err_far = quad(dphi2_de_dr, dist, 10.*dist, args, points=[dist],
                                    epsabs=0, epsrel=1e-5)
            # print("dphi_de_e(): %.2e, %.2e" % (err_near / int_near,
            #                                    err_far / int_far))
            return int_near + int_far

    return np.vectorize(_dphi_de_e)(e, dist, r_s, rho_s, gamma)


def J_factor(dist, r_s, rho_s, gamma, halo, th_max):
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
    if halo == "nfw":
        dJ_dr = dJ_dr_nfw
    elif halo == "exp":
        dJ_dr = dJ_dr_tt

    def _J_factor(dist, r_s, rho_s, gamma):
        args = (th_max, dist, r_s, rho_s, gamma)
        # Split integral around center of clump
        int_near, err_near = quad(dJ_dr, 0., dist, args, points=[dist], epsabs=0,
                        epsrel=1e-5)
        int_far, err_far = quad(dJ_dr, dist, 10.*dist, args, points=[dist], epsabs=0,
                       epsrel=1e-5)
        # print("J_factor(): %.2e, %.2e" % (err_near / int_near, err_far / int_far))

        return (int_near + int_far) * kpc_to_cm

    return np.vectorize(_J_factor)(dist, r_s, rho_s, gamma)


def dphi_de_g(e, dist, r_s, rho_s, gamma, halo, th_max,
              mx=dampe_excess_bin_high, sv=3e-26, fx=1.):
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
    def _dphi_de_g(e, dist, r_s, rho_s, gamma):
        if e >= mx:
            return 0.
        else:
            dOmega = 2*np.pi*(1.-np.cos(th_max))
            J = J_factor(dist, r_s, rho_s, gamma, halo, th_max)  # GeV^2 / cm^5
            ret_val = (dOmega/(4*np.pi) * J * sv / (2.*fx*mx**2) *
                       dn_de_gamma_AP(e, mx))
            return ret_val

    return np.vectorize(_dphi_de_g)(e, dist, r_s, rho_s, gamma)


def rho_s_dampe(dist, r_s, gamma, halo, bg_flux=bg_dampe, sv=3e-26, fx=1.):
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
    rho_s : float
        Density normalization in GeV / cm^3.
    """
    mx = dampe_excess_bin_high

    # Residual integrated flux
    @cfunc(float64(float64))
    def bg_flux_cf(e):
        return bg_flux(e)
    bg_flux_LLC = LowLevelCallable(bg_flux_cf.ctypes)
    bg_iflux = quad(bg_flux_LLC, dampe_excess_bin_low, dampe_excess_bin_high,
                    epsabs=0, epsrel=1e-5)[0]
    residual_iflux = dampe_excess_iflux - bg_iflux

    if halo == "nfw":
        dphi2_de_dr = dphi2_de_dr_nfw
    elif halo == "exp":
        dphi2_de_dr = dphi2_de_dr_tt

    def _rho_s_dampe(dist, r_s, gamma):
        # Set rho_s to 1 and use the fact that the flux is proportional to
        # rho_s**2
        args = (dist, r_s, 1., gamma, mx, sv, fx)
        # Improve numerical stability by splitting the spatial integral. The
        # factor of 2 accounts for DAMPE measuring e+ and e-.
        dm_iflux_near, err_near = dblquad(
            dphi2_de_dr,
            dampe_excess_bin_low, dampe_excess_bin_high,
            lambda e: 0, lambda e: dist,
            args=args, epsabs=0, epsrel=1e-5)
        dm_iflux_near *= 2
        err_near *= 2
        dm_iflux_far, err_far = dblquad(
            dphi2_de_dr,
            dampe_excess_bin_low, dampe_excess_bin_high,
            lambda e: dist, lambda e: 10.*dist,
            args=args, epsabs=0, epsrel=1e-5)
        dm_iflux_far *= 2
        err_far *= 2
        # print("rho_s_dampe(): %.2e, %.2e" % (err_near / dm_iflux_near,
        #                                      err_far / dm_iflux_far))

        return np.sqrt(residual_iflux / (dm_iflux_near + dm_iflux_far))

    return np.vectorize(_rho_s_dampe)(dist, r_s, gamma)


def gamma_ray_extent(dist, r_s, rho_s, gamma, halo, e,
                     thresh=0.5, mx=dampe_excess_bin_high, sv=3e-26, fx=1.):
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
    r_s : float
        Subhalo scale radius (kpc).
    rho_s : float
        Subhalo density normalization (GeV / cm^3).

    Notes
    -----
    Throws many ZeroDivisionError exceptions. Lots of numerical issues at small
    r_s and large dist.

    Returns
    -------
    th_ext : float
        Radius of observing region such that the flux in the region is equal to
        thresh times the total flux.
    """
    def _gamma_ray_extent(dist, r_s, rho_s, gamma):
        # Compute flux integrating over the whole sky
        total_flux = dphi_de_g(e, dist, r_s, rho_s, gamma, halo, np.pi, mx, sv, fx)

        def loss(log10_th):
            return (dphi_de_g(e, dist, r_s, rho_s, gamma, halo, 10.**log10_th,
                              mx, sv, fx) / total_flux - thresh)**2

        bracket_low = np.log10(1e-5 * fermi_psf)
        bracket_high = np.log10(0.99 * np.pi)
        # Hacky but effective way of finding a bracketing interval
        log10_ths = np.linspace(bracket_low, bracket_high, 20)
        losses = np.vectorize(loss)(log10_ths)
        bracket_middle = log10_ths[np.nanargmin(losses)]
        # Make sure loss at bracket endpoints is defined
        bracket_low = log10_ths[np.where(~np.isnan(losses))[0]][0]
        bracket_high = log10_ths[np.where(~np.isnan(losses))[0]][-1]

        # Do not optimize if rho_s is nan
        if not np.isnan(rho_s):
            log10_th = minimize_scalar(
                loss,
                bracket=(bracket_low,
                         bracket_middle,
                         bracket_high),
                bounds=(fermi_psf, np.pi)).x
            return 10.**log10_th
        else:
            return np.nan

    return np.vectorize(_gamma_ray_extent)(dist, r_s, rho_s, gamma)


def line_width_constraint(dist, r_s, rho_s, gamma, halo, n_sigma=3.,
                          bg_flux=bg_dampe, mx=dampe_excess_bin_high, sv=3e-26,
                          fx=1., excluded_idxs=[]):
    """Returns significance of largest excess in a DAMPE bin aside from the one
    with the true excess.

    TODO: this is a mess. Also, should find a way to use cfunc to speed up the
    DM flux integrals.
    """
    # Get index of bin containing the DM mass
    mx_idx = np.digitize(dampe_excess_bin_high,
                         np.unique(dampe_bins.flatten()),
                         right=True)

    # Reverse the list since constraint is likely to be set by bin closest to
    # the excess.
    idxs = set(range(len(dampe_bins))) - set(excluded_idxs)
    idxs = idxs - set(range(mx_idx, dampe_bins.shape[0]))
    idxs = sorted(list(idxs))
    idxs.reverse()

    bins = dampe_bins[idxs]
    dfluxes = dampe_dflux[idxs]
    dflux_errs = dampe_dflux_err[idxs]

    # Compute residual integrated flux in each relevant bin
    @cfunc(float64(float64))
    def bg_flux_cf(e):
        return bg_flux(e)
    bg_flux_LLC = LowLevelCallable(bg_flux_cf.ctypes)

    residual_ifluxes = []
    iflux_errs = []
    for (e_low, e_high), flux, err in zip(bins, dfluxes, dflux_errs):
        # Integrated flux
        dampe_iflux = (e_high - e_low) * flux
        bg_iflux = quad(bg_flux_LLC, e_low, e_high, epsabs=0, epsrel=1e-5)[0]
        residual_ifluxes.append(dampe_iflux - bg_iflux)
        # Error on integrated flux
        iflux_errs.append((e_high - e_low) * err)

    def _line_width_constraint(dist, r_s, rho_s, gamma):
        args = (dist, r_s, rho_s, gamma, halo, mx, sv, fx)
        n_sigma_max = 0.
        for (e_low, e_high), res, err in zip(bins, residual_ifluxes,
                                               iflux_errs):
            # Factor of 2 is needed because DAMPE measures e+ and e-
            dm_iflux, err = quad(dphi_de_e, e_low, e_high, args, epsabs=0,
                                 epsrel=1e-5)
            dm_iflux *= 2.
            err *= 2.
            # print("lw_constraint(): %.2e" % (err / dm_iflux))
            # Determine significance of DM contribution
            n_sigma_bin = (dm_iflux - res) / err
            n_sigma_max = max(n_sigma_bin, n_sigma_max)

        return n_sigma_max

    return np.vectorize(_line_width_constraint)(dist, r_s, rho_s, gamma)


def fermi_point_src_contraint(dist, r_s, gamma, halo,
                              mx=dampe_excess_bin_high, sv=3e-26, fx=1.,
                              e_star=230.):
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
    dphi_de_g_dm = dphi_de_g(e_star, dist, r_s, 1, gamma, halo, fermi_psf, mx,
                             sv, fx)
    dphi_de_g_fermi = fermi_pt_src_sens(e_star)

    return np.sqrt(dphi_de_g_fermi / dphi_de_g_dm)


def integrated_aniso(dist, r_s, rho_s, gamma, halo, e_low=e_low_aniso_fermi[-1],
                     e_high=e_high_aniso_fermi[-1], bg_flux=bg_dampe,
                     mx=dampe_excess_bin_high, sv=3e-26, fx=1.):
    """Computes delta(e), the differential e+- anisotropy.

    TODO: fix this!!!
    """
    if halo == "exp":
        dphi3_de_dr_dd = dphi3_de_dr_dd_exp
    elif halo == "nfw":
        dphi3_de_dr_dd = dphi3_de_dr_dd_nfw

    def _helper(dist, r_s, rho_s):
            # Gradient of flux from clump
            # @jit(nopython=True)
            # def _int_num(r, e):
            #     coeff = 3*D(e) / speed_of_light / kpc_to_cm
            #     return coeff * dphi3_de_dr_dd(r, e, dist, r_s, rho_s, gamma, mx, sv, fx)
            num = 2*dblquad(
                dphi3_de_dr_dd, e_low, e_high, lambda e: 0., lambda e: 10*dist,
                epsabs=0, epsrel=1e-5)[0]

            denom = 2*quad(dphi_de_e, e_low, e_high, args=(dist, r_s, rho_s,
                                                           gamma, halo, mx, sv,
                                                           fx),
                           epsabs=0, epsrel=1e-5)[0]

            @jit(nopython=True)
            def _int_bg(e):
                return bg_flux(e)
            denom += quad(_int_bg, e_low, e_high, epsabs=0, epsrel=1e-5)[0]

            return num / denom

    return np.vectorize(_helper)(dist, r_s, rho_s)


def differential_aniso(e, dist, r_s, rho_s, gamma, halo, bg_flux=bg_dampe,
                       mx=dampe_excess_bin_high, sv=3e-26, fx=1.):
    """Computes delta(e), the differential e+- anisotropy.
    """
    if halo == "exp":
        dphi3_de_dr_dd = dphi3_de_dr_dd_exp
    elif halo == "nfw":
        dphi3_de_dr_dd = dphi3_de_dr_dd_nfw

    def _helper(e, dist, r_s, rho_s):
        if e > mx:
            return 0.
        else:
            dphi_de_tot = (bg_flux(e) +
                           dphi_de_e(e, dist, r_s, rho_s, gamma, halo, mx, sv, fx))
            args = (e, dist, r_s, rho_s, gamma, mx, sv, fx)
            # Gradient of flux from clump
            dphi2_de_dd_near = 2*quad(dphi3_de_dr_dd, 0, dist, args,
                                      points=[dist], epsabs=0, epsrel=1e-5)[0]
            dphi2_de_dd_far = 2*quad(dphi3_de_dr_dd, dist, 10.*dist, args,
                                     points=[dist], epsabs=0, epsrel=1e-5)[0]
            dphi2_de_dd = dphi2_de_dd_near + dphi2_de_dd_far

            return 3 * D(e) / speed_of_light * np.abs(dphi2_de_dd / dphi_de_tot) / kpc_to_cm

    return np.vectorize(_helper)(e, dist, r_s, rho_s)


#def anisotropy_constraint(dist, r_s, rho_s, gamma, halo, bg_flux=bg_dampe,
#                          mx=dampe_excess_bin_high, sv=3e-26, fx=1.,
#                          debug=False):
#    """Computes the ratio of the clump e-+e+ anisotropy to the Fermi anisotropy
#    bound in its highest-energy bin.
#    """
#    return (integrated_aniso(dist, r_s, rho_s, gamma, halo,
#                             e_low_aniso_fermi[-1], e_high_aniso_fermi[-1],
#                             bg_flux, mx, sv, fx, debug) /
#            aniso_fermi[-1])
