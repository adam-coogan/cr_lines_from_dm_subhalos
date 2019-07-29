from collections import namedtuple
import numpy as np
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable
from scipy.special import gamma as Gamma

from dm_params import mx, fx, sv
from utilities import e0, b0, D0, delta, kpc_to_cm, speed_of_light, D
from utilities import GeV_to_m_sun
from utilities import rho_max, lambert_w_series, gamma_inc_upper

@np.vectorize
def rho(dist, Rb, rho0, gamma):
    r_p = ann_plateau_radius(Rb, rho0, gamma)
    if np.abs(dist) < r_p:
        return rho_max
    else:
        return rho0 * (Rb / np.abs(dist))**gamma * np.exp(-np.abs(dist) / Rb)


@np.vectorize
def mass(Rb, rho0, gamma):
    return (4. * np.pi * rho0 * (Rb * kpc_to_cm)**3 * Gamma(3. - gamma)) * GeV_to_m_sun


@np.vectorize
def luminosity(Rb, rho0, gamma):
    factor = sv / (2*fx*mx**2)
    # No annihilation plateau
    Rb_cm = kpc_to_cm * Rb
    L_clump = 2**(-1 + 2*gamma) * np.pi * Rb_cm**3 * rho0**2 * Gamma(3 - 2*gamma)
    return factor * L_clump
    # With annihilation plateau
    # Rb_cm = kpc_to_cm * Rb
    # r_p_cm = kpc_to_cm * ann_plateau_radius(Rb, rho0, gamma)
    # return np.pi/6. * factor * (3. * 4.**gamma * rho0**2 * Rb_cm**3 *
    #                             gamma_inc_upper(3. - 2. * gamma,
    #                                             2. * r_p_cm / Rb_cm) +
    #                             8. * rho_max**2 * r_p_cm**3)


@jit(nopython=True)
def ann_plateau_radius(Rb, rho0, gamma):
    """Annihilation plateau radius, kpc.
    """
    return gamma * Rb * lambert_w_series(rho0**(1. / gamma) *
                                         rho_max**(-1. / gamma) /
                                         gamma)


@jit(nopython=True)
def K(r, th, d, Rb, rho0, gamma):
    cth = np.cos(th)

    if r == 0:
        return (Rb / d)**(2 * gamma) * rho0**2 * cth / np.exp(2 * d / Rb)
    else:
        # r_p = ann_plateau_radius(Rb, rho0, gamma)
        # if d_cl < r_p:
        #     return cth * rho_max**2
        # else:
        # Numerically stable version of the law of cosines
        d_cl = np.sqrt((d - r)**2 + 4 * d * r * np.sin(th / 2.)**2)
        return (4**(-1. + gamma) * Rb**2 * rho0**2 *
                gamma_inc_upper(2. * (1. - gamma), 2 * d_cl / Rb)) / (d * r)


@jit(nopython=True)
def K_full(r, d, Rb, rho0, gamma):
    """K(th=0) - K(th=pi)
    """
    return K(r, 0., d, Rb, rho0, gamma) - K(r, np.pi, d, Rb, rho0, gamma)


# e+/- flux integrand
def dphi_e_dr(r, e, d, Rb, rho0, gamma, mx, sv, fx):
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
        K_term = K_full(r, d, Rb, rho0, gamma)

        return fact_const * fact_e * K_term * (r_term * kpc_to_cm**3)

dphi_e_dr_jit = jit(dphi_e_dr, nopython=True)
@cfunc(float64(intc, CPointer(float64)))
def dphi_e_dr_cfunc(n, xx):
    return dphi_e_dr_jit(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8])
dphi_e_dr_llc = LowLevelCallable(dphi_e_dr_cfunc.ctypes)


# J-factor integrand
def dJ_dr(r, th_max, d, Rb, rho0, gamma):
    # Numerically stable expression for solid angle subtended by target
    dOmega = 4 * np.pi * np.sin(0.5 * th_max)**2
    th_term = K(r, 0, d, Rb, rho0, gamma) - K(r, th_max, d, Rb, rho0, gamma)
    return 2*np.pi/dOmega * th_term

dJ_dr_jit = jit(dJ_dr, nopython=True)
@cfunc(float64(intc, CPointer(float64)))
def dJ_dr_cfunc(n, xx):
    return dJ_dr_jit(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])
dJ_dr_llc = LowLevelCallable(dJ_dr_cfunc.ctypes)
