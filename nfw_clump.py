from collections import namedtuple
import numpy as np
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable

from constants import e0, b0, D0, delta, kpc_to_cm, speed_of_light

NFW_params = namedtuple("NFW_params", ["rs", "rhos", "gamma"])


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
        fact_const = np.pi*sv / (fx*mx**2)

        # Energy-dependent parts
        b = b0 * (e / e0)**2
        lam = D0 * e0 / (b0 * (1. - delta)) * \
            ((e0 / e)**(1. - delta) - (e0 / mx)**(1. - delta))
        fact_e = 1. / (b * (4.*np.pi*lam)**1.5)

        # Term from performing theta integral
        K_diff = K(r, 0, d, rs, rhos, gamma) - \
            K(r, np.pi, d, rs, rhos, gamma)
        # Term with purely radial dependence
        r_term = r**2 * np.exp(-(r * kpc_to_cm)**2 / (4. * lam))

        # Put it all together
        return fact_const * fact_e * K_diff * r_term * \
            speed_of_light * kpc_to_cm**3


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
dJ_dr = LowLevelCallable(dJ_dr_cf.ctypes)
