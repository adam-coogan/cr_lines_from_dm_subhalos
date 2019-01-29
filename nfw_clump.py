from collections import namedtuple
import ctypes
import numpy as np
from numba.extending import get_cython_function_address
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable

from constants import e0, b0, D0, delta, kpc_to_cm, speed_of_light

NFW_params = namedtuple("NFW_params", ["rs", "rhos", "gamma"])

# Obtained by finding the mangled function name in:
# >>> from scipy.special.cython_special import __pyx_capi__
# >>> __pyx_capi__
addr_hyp = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1hyp2f1")
functype_hyp = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double)
hyp2f1_cs = functype_hyp(addr_hyp)


@jit(nopython=True)
def K(r, th, d, rs, rhos, gamma):
    ct = np.cos(th)

    if gamma == 1:
        return -(rhos**2*rs**2 *
                 (3*np.log(d**2 + r**2 - 2*d*r*ct) -
                  6*np.log(rs + np.sqrt(d**2 + r**2 - 2*d*r*ct)) +
                  (rs*(11*rs**2 + 6*(d**2 + r**2 - 2*d*r*ct) +
                       15*rs*np.sqrt(d**2 + r**2 - 2*d*r*ct))) /
                  (rs + np.sqrt(d**2 + r**2 - 2*d*r*ct))**3)) / (6.*d*r)
    else:
        return (rhos**2*rs**2*(d**2 + r**2 - 2*d*r*ct)**(1 - gamma) *
                (rs + np.sqrt(d**2 + r**2 - 2*d*r*ct))**(-5 + 2*gamma) *
                ((-2 + gamma)*(-5 + 2*gamma)*rs**2 *
                 (-3*rs + 2*gamma*rs - 3*np.sqrt(d**2 + r**2 - 2*d*r*ct)) -
                 3*d**2*((5 - 2*gamma)*rs + np.sqrt(d**2 + r**2 - 2*d*r*ct)) +
                 6*d*r*ct*((5 - 2*gamma)*rs +
                           np.sqrt(d**2 + r**2 - 2*d*r*ct)) -
                 3*r**2*(5*rs - 2*gamma*rs +
                         np.sqrt(d**2 + r**2 - 2*d*r*ct)))) / \
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


@jit(nopython=True)
def P_aniso(r, d, rs, rhos, gamma):
    if gamma == 1:
        return (rhos**2*rs*
                ((12*(d**2 - r**2))/np.abs(d-r) -
                 (12*(d**2 - r**2))/(d + r) + (3*(d - r)*rs)/(d + r) -
                 (3*(d + r)*rs)/(d - r) + 3*rs*np.log((d - r)**2) -
                 3*rs*np.log((d + r)**2) - 6*rs*np.log(rs + np.abs(d-r)) +
                 6*rs*np.log(rs + (d + r)) -
                 (3*rs**3*(d**2 - r**2 + rs**2))/(rs + np.abs(d-r))**4 +
                 (2*rs**2*(-3*d**2 + 3*r**2 + rs**2))/(rs + np.abs(d-r))**3 +
                 (3*rs*(-3*d**2 + 3*r**2 + rs**2))/(rs + np.abs(d-r))**2 +
                 (6*(-2*d**2 + 2*r**2 + rs**2))/(rs + np.abs(d-r)) +
                 (3*rs**3*(d**2 - r**2 + rs**2))/(rs + (d + r))**4 -
                 (2*rs**2*(-3*d**2 + 3*r**2 + rs**2))/(rs + (d + r))**3 -
                 (3*rs*(-3*d**2 + 3*r**2 + rs**2))/(rs + (d + r))**2 -
                 (6*(-2*d**2 + 2*r**2 + rs**2))/(rs + (d + r))))/(6.*d**2*r)
    else:
        return (rhos**2*rs**(-1 + 2*gamma) *
                (-(((6*(d**2 - r**2) *
                     hyp2f1_cs(1 - 2*gamma,7 - 2*gamma,2 - 2*gamma, -(np.sqrt((d - r)**2)/rs)) *
                     np.sqrt((d - r)**2))/(-1 + 2*gamma) +
                    ((d - r) *
                     (gamma*(-3 + 2*gamma)*(d - r)*rs *
                      hyp2f1_cs(2 - 2*gamma,7 - 2*gamma,3 - 2*gamma, -(np.sqrt((d - r)**2)/rs)) +
                      (-1 + gamma) *
                      ((-3 + 2*gamma)*(d + r)*rs *
                       hyp2f1_cs(7 - 2*gamma,-2*gamma,1 - 2*gamma, -(np.sqrt((d - r)**2)/rs)) +
                       6*(d - r) *
                       hyp2f1_cs(3 - 2*gamma,7 - 2*gamma, 4 - 2*gamma,-(np.sqrt((d - r)**2)/rs)) *
                       np.sqrt((d - r)**2)))) /
                    ((-1 + gamma)*(-3 + 2*gamma))) /
                   ((d - r)**2)**gamma) +
                 ((6*(d**2 - r**2) *
                   hyp2f1_cs(1 - 2*gamma,7 - 2*gamma, 2 - 2*gamma,-(np.sqrt((d + r)**2)/rs)) *
                   np.sqrt((d + r)**2)) / (-1 + 2*gamma) +
                  ((d + r) * (gamma*(-3 + 2*gamma)*(d + r)*rs *
                              hyp2f1_cs(2 - 2*gamma,7 - 2*gamma,3 - 2*gamma, -(np.sqrt((d + r)**2)/rs)) +
                              (-1 + gamma) *
                              ((-3 + 2*gamma)*(d - r)*rs *
                               hyp2f1_cs(7 - 2*gamma,-2*gamma,1 - 2*gamma, -(np.sqrt((d + r)**2)/rs)) +
                               6*(d + r) *
                               hyp2f1_cs(3 - 2*gamma,7 - 2*gamma,4 - 2*gamma,-(np.sqrt((d + r)**2)/rs)) *
                               np.sqrt((d + r)**2)))) /
                  ((-1 + gamma)*(-3 + 2*gamma))) / ((d + r)**2)**gamma))/(2.*d**2*r)


@cfunc(float64(intc, CPointer(float64)))
def dphi3_de_dr_dd_cf(n, xx):
    r = xx[0]
    e = xx[1]
    d = xx[2]
    rs = xx[3]
    rhos = xx[4]
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
        P_term = P_aniso(r, d, rs, rhos, gamma)

        return fact_const * fact_e * P_term * (r_term * kpc_to_cm**3)


@cfunc(float64(intc, CPointer(float64)))
def dphi2_de_dr_cf(n, xx):
    r = xx[0]
    e = xx[1]
    d = xx[2]
    rs = xx[3]
    rhos = xx[4]
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
        K_term = K_full(r, d, rs, rhos, gamma)

        return fact_const * fact_e * K_term * (r_term * kpc_to_cm**3)


@cfunc(float64(intc, CPointer(float64)))
def dJ_dr_cf(n, xx):
    # Extract arguments
    r = xx[0]
    th_max = xx[1]
    d = xx[2]
    rs = xx[3]
    rhos = xx[4]
    gamma = xx[5]

    # Solid angle subtended by target
    dOmega = 2*np.pi*(1. - np.cos(th_max))

    th_term = K(r, 0, d, rs, rhos, gamma) - \
        K(r, th_max, d, rs, rhos, gamma)

    return 2*np.pi/dOmega * th_term


dphi2_de_dr = LowLevelCallable(dphi2_de_dr_cf.ctypes)
dphi3_de_dr_dd = LowLevelCallable(dphi3_de_dr_dd_cf.ctypes)
dJ_dr = LowLevelCallable(dJ_dr_cf.ctypes)
