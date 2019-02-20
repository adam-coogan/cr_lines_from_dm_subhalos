from background_models import bg_dampe
from constants import rho_max, dampe_excess_bin_high, kpc_to_cm
from constants import dampe_bins, dampe_dflux, dampe_dflux_err, dn_de_gamma_AP
from constants import e0, b0, D0, delta, kpc_to_cm, speed_of_light, D
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from pointlike_clump import dphi_de_e_pt
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.optimize import root_scalar

def rho(dist, r_p, r_tr, gamma, rho_max=rho_max):
    def helper(dist, r_p, r_tr):
        if 0 < dist < r_p:
            return rho_max
        elif r_p < dist < r_tr:
            return rho_max * (r_p / dist)**gamma
        else:
            return 0.
    return np.vectorize(helper)(dist, r_p, r_tr)

def ann_plateau_radius(r_tr, lum, gamma, rho_max=rho_max, sv=3e-26, mx=dampe_excess_bin_high):
    def helper(r_tr, lum):
        def f(r_p):
            if gamma != 1.5:
                return (3*r_tr**3*(r_p/r_tr)**(2.*gamma) - 2*gamma*r_p**3 +
                        3*(2.*gamma-3)*mx**2*lum/(2*np.pi*sv*rho_max**2) / kpc_to_cm**3)
            else:
                return r_p**3 * (1. + 3.*np.log(r_tr/r_p)) - 3.*mx**2*lum/(2*np.pi*sv*rho_max**2) / kpc_to_cm**3
        def fprime(r_p):
            if gamma != 1.5:
                return 6*gamma*r_p*(r_tr*(r_p/r_tr)**(2.*gamma-1.) - r_p)
            else:
                return 9.*r_p**2*np.log(r_tr/r_p)

        sol = root_scalar(f, x0=1e-7*r_tr, fprime=fprime, method="newton",
                          xtol=1e-100, rtol=1e-5)
        root = np.real(sol.root)
        if root < 0:
            return np.nan
            # raise ValueError("r_p is negative.")
        elif root > r_tr:
            return np.nan
            # raise ValueError("r_p is larger than the truncation radius.")
        elif np.imag(sol.root) / root > 1e-5:
            return np.nan
            # raise ValueError("r_p is imaginary")
        else:
            return root

    return np.vectorize(helper)(r_tr, lum)

def lum(r_p, r_tr, gamma, rho_max=rho_max, mx=dampe_excess_bin_high, sv=3e-26, fx=1.):
    """Hz."""
    factor = kpc_to_cm**3 * sv / (2 * fx * mx**2)
    if gamma != 1.5:
        return (4*np.pi*rho_max**2 * (3*r_tr**3*(r_p/r_tr)**(2.*gamma) -
                                      2.*gamma*r_p**3)/(9.-6.*gamma)) * factor
    else:
        return 4./3.*np.pi*r_p**3*rho_max**2*(1. + 3*np.log(r_tr / r_p)) * factor

def line_width_constraint(dist, r_p, r_tr, gamma, bg_flux=bg_dampe,
                          mx=dampe_excess_bin_high, sv=3e-26, fx=1.,
                          excluded_idxs=[]):
    """Returns significance of largest excess in a DAMPE bin aside from the one
    with the true excess.
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

    def _line_width_constraint(dist, r_p, r_tr, gamma):
        args = (dist, mx, lum(r_p, r_tr, gamma, mx=mx, sv=sv, fx=fx))
        n_sigma_max = 0.
        for (e_low, e_high), res, err in zip(bins, residual_ifluxes,
                                               iflux_errs):
            # Factor of 2 is needed because DAMPE measures e+ and e-
            dm_iflux = 2.*quad(dphi_de_e_pt, e_low, e_high, args, epsabs=0,
                               epsrel=1e-5)[0]
            # Determine significance of DM contribution
            n_sigma_bin = (dm_iflux - res) / err
            n_sigma_max = max(n_sigma_bin, n_sigma_max)

        return n_sigma_max

    return np.vectorize(_line_width_constraint)(dist, r_p, r_tr, gamma)

@jit(nopython=True)
def K(r, th, d, r_p, r_tr, gamma):
    cth = np.cos(th)
    d_cl = np.sqrt(d**2 + r**2 - 2 * d * r * cth)

    if d_cl < r_p:
        return cth * rho_max**2
    elif r_p < d_cl < r_tr:
        return (d_cl**2*rho_max**2*(r_p/d_cl)**(2*gamma))/(2.*d*(-1 + gamma)*r)
    else:
        return 0.

@jit(nopython=True)
def K_full(r, d, r_p, r_tr, gamma):
    """K(th=0) - K(th=pi)
    """
    return K(r, 0., d, r_p, r_tr, gamma) - K(r, np.pi, d, r_p, r_tr, gamma)

@cfunc(float64(intc, CPointer(float64)))
def dphi2_de_dr_cf(n, xx):
    r = xx[0]
    e = xx[1]
    d = xx[2]
    r_p = xx[3]
    r_tr = xx[4]
    gamma = xx[5]
    mx = xx[6]
    sv = xx[7]
    fx = xx[8]

    if e >= mx:
        return 0.
    else:
        # Constant factor
        fact_const = np.pi*sv / (fx*mx**2) * speed_of_light/(4*np.pi)
        # Energy-dependent parts
        b = b0 * (e / e0)**2
        lam = D0 * e0 / (b0 * (1. - delta)) * \
            ((e0 / e)**(1. - delta) - (e0 / mx)**(1. - delta))
        fact_e = 1. / (b * (4.*np.pi*lam)**1.5)
        # Term with purely radial dependence
        r_term = r**2 * np.exp(-(r * kpc_to_cm)**2 / (4. * lam))
        # Term from performing theta integral
        K_term = K_full(r, d, r_p, r_tr, gamma)

        return fact_const * fact_e * K_term * (r_term * kpc_to_cm**3)

dphi2_de_dr = LowLevelCallable(dphi2_de_dr_cf.ctypes)

@cfunc(float64(intc, CPointer(float64)))
def dJ_dr_cf(n, xx):
    r = xx[0]
    th_max_arg = xx[1]
    d = xx[2]
    r_p = xx[3]
    r_tr = xx[4]
    gamma = xx[5]
    # Hacky way to deal with very steep profile
    th_max = th_max_arg
    # Solid angle subtended by target
    dOmega = 2*np.pi*(1. - np.cos(th_max))
    th_term = K(r, 0, d, r_p, r_tr, gamma) - K(r, th_max, d, r_p, r_tr, gamma)
    return 2*np.pi/dOmega * th_term

dJ_dr = LowLevelCallable(dJ_dr_cf.ctypes)

def J_factor(dist, r_p, r_tr, gamma, th_max):
    def _J_factor(dist, r_p, r_tr, gamma):
        args = (th_max, dist, r_p, r_tr, gamma)
        # Split integral around center of clump
        # Hacky way to deal with very steep profile
        int_near = quad(dJ_dr, np.max([0., dist-2*r_p]), dist, args,
                        points=[dist], epsabs=0, epsrel=1e-5)[0]
        int_far = quad(dJ_dr, dist, dist+2*r_p, args, points=[dist], epsabs=0,
                       epsrel=1e-5)[0]

        return (int_near + int_far) * kpc_to_cm

    return np.vectorize(_J_factor)(dist, r_p, r_tr, gamma)

def dphi_de_g(e, dist, r_p, r_tr, gamma, th_max, mx=dampe_excess_bin_high,
              sv=3e-26, fx=1.):
    def _dphi_de_g(e, dist, r_p, r_tr, gamma):
        if e >= mx:
            return 0.
        else:
            dOmega = 2*np.pi*(1.-np.cos(th_max))
            J = J_factor(dist, r_p, r_tr, gamma, th_max)  # GeV^2 / cm^5
            ret_val = (dOmega/(4*np.pi) * J * sv / (2.*fx*mx**2) *
                       dn_de_gamma_AP(e, mx))
            return ret_val

    return np.vectorize(_dphi_de_g)(e, dist, r_p, r_tr, gamma)