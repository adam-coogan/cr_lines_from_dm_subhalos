from collections import namedtuple
import ctypes
import numpy as np
from numba.extending import get_cython_function_address
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.optimize import brentq

from dm_params import mx, fx, sv
from utilities import e0, b0, D0, delta, kpc_to_cm, speed_of_light, D
from utilities import rho_critical, GeV_to_m_sun

# Obtained by finding the mangled function name in:
# >>> from scipy.special.cython_special import __pyx_capi__
# >>> __pyx_capi__
addr_hyp = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1hyp2f1")
functype_hyp = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double)
hyp2f1_cs = functype_hyp(addr_hyp)


@np.vectorize
def rho(dist, rs, rhos, gamma):
    return rhos * (rs / np.abs(dist))**gamma * (1. + np.abs(dist) / rs)**(gamma - 3.)


@np.vectorize
def mass(rs, rhos, gamma):
    # Find virial mass numerically. The "magic number" of 10000 should
    # be fine for this project.
    try:
        def _r_vir_integrand(r):
            return (rho(r, rs, rhos, gamma) - 200. * rho_critical)

        r_vir = brentq(_r_vir_integrand, 0.01 * rs, 100. * rs, xtol=1e-200)
    except RuntimeError:
        r_vir = np.nan

    # Integrate numerically: analytic result has an imaginary part
    factor = 4.*np.pi * GeV_to_m_sun * kpc_to_cm**3

    def _mass_integrand(r):
        return r**2 * rho(r, rs, rhos, gamma)

    return factor * quad(_mass_integrand, 0, r_vir, epsabs=0, epsrel=1e-5)[0]


@np.vectorize
def luminosity(rs, rhos, gamma):
    factor = sv / (2 * fx * mx**2)
    rs_cm = kpc_to_cm * rs
    L_clump = (4*np.pi*rhos**2*rs_cm**3) / (30 - 47*gamma + 24*gamma**2 - 4*gamma**3)
    return factor * L_clump


@jit(nopython=True)
def K(r, th, d, rs, rhos, gamma):
    cth = np.cos(th)

    if r == 0:
        return rhos**2 * rs**6 * (d + rs)**(-6 + 2 * gamma) * cth / d**(2 * gamma)
    else:
        # Numerically stable version of the law of cosines
        d_cl_2 = (d - r)**2 + 4 * d * r * np.sin(th / 2.)**2
        d_cl = np.sqrt(d_cl_2)
        if gamma == 1:
            return -(rhos**2*rs**2 *
                     (3*np.log(d_cl_2) -
                      6*np.log(rs + d_cl) +
                      (rs*(11*rs**2 + 6*d_cl_2 +
                           15*rs*d_cl)) /
                      (rs + d_cl)**3)) / (6.*d*r)
        else:
            return (rhos**2*rs**2*d_cl_2**(1 - gamma) *
                    (rs + d_cl)**(-5 + 2*gamma) *
                    ((-2 + gamma)*(-5 + 2*gamma)*rs**2 *
                     (-3*rs + 2*gamma*rs - 3*d_cl) -
                     3*d**2*((5 - 2*gamma)*rs + d_cl) +
                     6*d*r*cth*((5 - 2*gamma)*rs + d_cl) -
                     3*r**2*(5*rs - 2*gamma*rs + d_cl))) / \
                (2.*d*(-2 + gamma)*(-1 + gamma)*(-5 + 2*gamma)*(-3 + 2*gamma)*r)

@jit(nopython=True)
def K_full(r, d, rs, rhos, gamma):
    """K(th=0) - K(th=pi)
    """
    if gamma == 1:
        return (rhos**2*rs**2*((rs*(6*d**2 + 12*d*r + 6*r**2 + 15*(d + r)*rs +
                                    11*rs**2)) /
                               (d + r + rs)**3 - (rs*(6*d**2 - 12*d*r +
                                                      6*r**2 + 11*rs**2 +
                                                      15*rs*np.abs(d - r))) /
                               (rs + np.abs(d - r))**3 - 3*np.log((d - r)**2) +
                               6*np.log(d + r) - 6*np.log(d + r + rs) +
                               6*np.log(rs + np.abs(d - r)))) / (6.*d*r)
    else:
        return (rhos**2*rs**2*((d + r)**(2 - 2*gamma) *
                               (d + r + rs)**(-5 + 2*gamma) *
                               (3*d**3 + 3*r**3 + 3*(5 - 2*gamma)*r**2*rs +
                                3*(10 - 9*gamma + 2*gamma**2)*r*rs**2 +
                                (30 - 47*gamma + 24*gamma**2 -
                                 4*gamma**3)*rs**3 +
                                3*d**2*(3*r + (5 - 2*gamma)*rs) +
                                3*d*(3*r**2 + 2*(5 - 2*gamma)*r*rs +
                                     (10 - 9*gamma + 2*gamma**2)*rs**2)) +
                               np.abs(d - r)**(2 - 2*gamma) *
                               (rs + np.abs(d - r))**(-5 + 2*gamma) *
                               ((-5 + 2*gamma)*rs *
                                (3*d**2 - 6*d*r + 3*r**2 +
                                 (6 - 7*gamma + 2*gamma**2)*rs**2) -
                                3*(d**2 - 2*d*r + r**2 +
                                   (10 - 9*gamma + 2*gamma**2)*rs**2) *
                                np.abs(d - r)))) / \
            (2.*d*(-2 + gamma)*(-1 + gamma)*(-5 + 2*gamma)*(-3 + 2*gamma)*r)


# e+/- flux integrand
def dphi_e_dr(r, e, d, rs, rhos, gamma, mx, sv, fx):
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
        K_term = K_full(r, d, rs, rhos, gamma)

        return fact_const * fact_e * K_term * (r_term * kpc_to_cm**3)

dphi_e_dr_jit = jit(dphi_e_dr, nopython=True)
@cfunc(float64(intc, CPointer(float64)))
def dphi_e_dr_cfunc(n, xx):
    return dphi_e_dr_jit(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8])
dphi_e_dr_llc = LowLevelCallable(dphi_e_dr_cfunc.ctypes)


# J-factor integrand
def dJ_dr(r, th_max, d, rs, rhos, gamma):
    # Numerically stable expression for solid angle subtended by target
    dOmega = 4 * np.pi * np.sin(0.5 * th_max)**2
    th_term = K(r, 0, d, rs, rhos, gamma) - K(r, th_max, d, rs, rhos, gamma)
    return 2*np.pi/dOmega * th_term

dJ_dr_jit = jit(dJ_dr, nopython=True)
@cfunc(float64(intc, CPointer(float64)))
def dJ_dr_cfunc(n, xx):
    return dJ_dr_jit(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])
dJ_dr_llc = LowLevelCallable(dJ_dr_cfunc.ctypes)
