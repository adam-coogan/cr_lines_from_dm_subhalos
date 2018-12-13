from collections import namedtuple
import ctypes
import numpy as np
from numba import cfunc, jit, vectorize
from numba.extending import get_cython_function_address
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable

from constants import e0, b0, D0, delta, kpc_to_cm, speed_of_light

# Set up the gamma function. It needs to be defined this way since
# get_cython_function_address doesn't work with gamma, likely because gamma
# involves complex numbers.
addr_gs = get_cython_function_address("scipy.special.cython_special", "gammasgn")
functype_gs = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gammasgn_cs = functype_gs(addr_gs)

addr_gln = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype_gln = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gammaln_cs = functype_gln(addr_gln)

## Required to handle case where first argument to incomplete gamma function is 0.
# Unclear how to work around get_cython_function_address failing here.
#addr_expi = get_cython_function_address("scipy.special.cython_special", "expi")
#functype_expi = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
#expi_cs = functype_expi(addr_expi)

# Set up upper incomplete gamma function. scipy's version is normalized by
# 1/Gamma.
addr_ggi = get_cython_function_address("scipy.special.cython_special", "gammaincc")
functype_gi = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammaincc_cs = functype_gi(addr_ggi)


@vectorize('float64(float64, float64)', nopython=True)
def gamma_inc_upper(a, z):
    """Incomplete upper gamma function as defined in Mathematica and on
    wikipedia.

    Notes
    -----
    * Verified against Mathematica
    * Uses recursion relation 8.8.2 from https://dlmf.nist.gov/8.8 to handle
      negative values of a:
          gamma_inc_upper(a+1, z) == a * gamma_inc_upper(a, z) + z**a * exp(-z)

    Parameters
    ----------
    a : float
        First argument. Cannot be zero or a negative integer.
    z : float
    """
    assert a != 0
    assert not (int(a) == a and a < 0)

    if a > 0:
        gamma_val = np.exp(gammaln_cs(a)) * gammasgn_cs(a)
        return gamma_val * gammaincc_cs(a, z)
    else:
        a_c = int(np.ceil(np.abs(a)))
        a_f = int(np.floor(np.abs(a)))

        exp_terms = 0.
        denom = 1.
        for m in range(a_f, -1, -1):
            denom *= a + m
            exp_terms = (exp_terms + z**a) / (a + m)
        exp_terms *= np.exp(-z)

        gamma_val = np.exp(gammaln_cs(a + a_c)) * gammasgn_cs(a + a_c)
        return gamma_val * gammaincc_cs(a + a_c, z) / denom - exp_terms


# For holding halo parameters
TT_params = namedtuple("TT_params", ["Rb", "rho0", "gamma"])


@jit(nopython=True)
def K(r, th, d, Rb, rho0, gamma):
    return (4**(-1. + gamma) * Rb**2 * rho0**2 *
            gamma_inc_upper(2. - 2.*gamma,
                            (2*np.sqrt(d**2 + r**2 -
                                       2*d*r*np.cos(th))) / Rb)) / (d*r)


@jit(nopython=True)
def K_def(r, d, Rb, rho0, gamma):
    """K(th=0) - K(th=pi)
    """
    return (4**(gamma - 1)*Rb**2*rho0**2 *
            (-gamma_inc_upper(2*(1 - gamma), 2 * (d + r) / Rb) +
             gamma_inc_upper(2*(1 - gamma), 2 * np.abs(d - r) / Rb))) / (d*r)


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

        # Term from performing theta integral
        K_diff = K_def(r, d, Rb, rho0, gamma)
        # Term with purely radial dependence
        r_term = r**2 * np.exp(-(r * kpc_to_cm)**2 / (4. * lam))

        # Put it all together
        return fact_const * fact_e * K_diff * (r_term * kpc_to_cm**3)


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

    th_term = K(r, 0, d, Rb, rho0, gamma) - \
        K(r, th_max, d, Rb, rho0, gamma)

    return 2*np.pi/dOmega * th_term


dphi2_de_dr = LowLevelCallable(dphi2_de_dr_cf.ctypes)
dJ_dr = LowLevelCallable(dJ_dr_cf.ctypes)
