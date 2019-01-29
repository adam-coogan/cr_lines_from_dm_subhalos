from collections import namedtuple
import numpy as np
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable

from constants import e0, b0, D0, delta, kpc_to_cm, speed_of_light, D
from constants import rho_max, lambert_w_series, gamma_inc_upper


# For holding halo parameters
TT_params = namedtuple("TT_params", ["Rb", "rho0", "gamma"])

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
    d_cl = np.sqrt(d**2 + r**2 - 2 * d * r * cth)
    # Annihilation plateau radius
    r_p = ann_plateau_radius(Rb, rho0, gamma)

    if d_cl < r_p:
        return cth * rho_max**2
    else:
        return (4**(-1. + gamma) * Rb**2 * rho0**2 *
                gamma_inc_upper(2. * (1. - gamma), 2 * d_cl / Rb)) / (d * r)


@jit(nopython=True)
def K_full(r, d, Rb, rho0, gamma):
    """K(th=0) - K(th=pi)
    """
    return K(r, 0., d, Rb, rho0, gamma) - K(r, np.pi, d, Rb, rho0, gamma)


@jit(nopython=True)
def P_aniso(r, d, Rb, rho0, gamma):
    if gamma != 1:
        return (2**(-3 + 2*gamma)*rho0**2 *
                (-4*(d**2 - r**2) * gamma_inc_upper(1 - 2*gamma, 2*np.abs(d-r)/Rb) +
                 4*(d**2 - r**2) * gamma_inc_upper(1 - 2*gamma, 2*(d+r)/Rb) -
                 2*gamma*Rb**2 * gamma_inc_upper(2*(1-gamma), 2*np.abs(d-r)/Rb) +
                 2*gamma*Rb**2 * gamma_inc_upper(2*(1-gamma), 2*(d+r)/Rb) -
                 Rb**2 * gamma_inc_upper(3 - 2*gamma, 2*np.abs(d-r)/Rb) +
                 Rb**2 * gamma_inc_upper(3 - 2*gamma, 2*(d+r)/Rb) -
                 8*d**2*gamma * gamma_inc_upper(-2*gamma, 2*np.abs(d - r)/Rb) +
                 8*gamma*r**2 * gamma_inc_upper(-2*gamma, 2*np.abs(d - r)/Rb) +
                 8*d**2*gamma * gamma_inc_upper(-2*gamma, 2*(d + r)/Rb) -
                 8*gamma*r**2 * gamma_inc_upper(-2*gamma, 2*(d + r)/Rb))) / (d**2*r)
    else:
        print("Cannot currently compute anisotropy for exp profile with gamma=1")


@cfunc(float64(intc, CPointer(float64)))
def dphi3_de_dr_dd_cf(n, xx):
    r = xx[0]
    e = xx[1]
    d = xx[2]
    Rb = xx[3]
    rho0 = xx[4]
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
        P_term = P_aniso(r, d, Rb, rho0, gamma)

        # TODO: check this and rename function!
        coeff = 3*D(e) / speed_of_light / kpc_to_cm

        return coeff * fact_const * fact_e * P_term * (r_term * kpc_to_cm**3)


@cfunc(float64(intc, CPointer(float64)))
def dphi2_de_dr_cf(n, xx):
    r = xx[0]
    e = xx[1]
    d = xx[2]
    Rb = xx[3]
    rho0 = xx[4]
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
        K_term = K_full(r, d, Rb, rho0, gamma)

        return fact_const * fact_e * K_term * (r_term * kpc_to_cm**3)


@cfunc(float64(intc, CPointer(float64)))
def dJ_dr_cf(n, xx):
    # Extract arguments
    r = xx[0]
    th_max = xx[1]
    d = xx[2]
    Rb = xx[3]
    rho0 = xx[4]
    gamma = xx[5]

    # Solid angle subtended by target
    dOmega = 2*np.pi*(1. - np.cos(th_max))

    th_term = K(r, 0, d, Rb, rho0, gamma) - K(r, th_max, d, Rb, rho0, gamma)

    return 2*np.pi/dOmega * th_term


dphi2_de_dr = LowLevelCallable(dphi2_de_dr_cf.ctypes)
dphi3_de_dr_dd = LowLevelCallable(dphi3_de_dr_dd_cf.ctypes)
dJ_dr = LowLevelCallable(dJ_dr_cf.ctypes)
