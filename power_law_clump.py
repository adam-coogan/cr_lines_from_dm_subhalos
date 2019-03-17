from background_models import Phi_e_bg
from utilities import rho_max, e_high_excess, kpc_to_cm
from utilities import bins_dampe, phis_dampe, phi_errs_dampe, dn_de_g_ap
from utilities import e0, b0, D0, delta, kpc_to_cm, speed_of_light, D
from dm_params import mx, fx, sv
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from pointlike_clump import phi_e as phi_e_pt
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad
from scipy.optimize import root_scalar


"""
Analyzes a clump with a truncated power-law profile.
"""


@np.vectorize
def rho(dist, r_p, r_tr, gamma, rho_max=rho_max):
    """Density profile.

    Parameters
    ----------
    dist : float
        Distance from clump's center, kpc.
    r_p : float
        Radius of annihilation plateau, kpc.
    r_tr : float
        Truncation radius, kpc.
    gamma : float
        Slope.
    rho_max : float
        Maximum DM density, which defines the annihilation plateau, GeV/cm^3.

    Returns
    -------
    Density, GeV/cm^3.
    """
    if 0 < dist < r_p:
        return rho_max
    elif r_p < dist < r_tr:
        return rho_max * (r_p / dist)**gamma
    else:
        return 0.


@np.vectorize
def ann_plateau_radius(r_tr, lum, gamma, rho_max=rho_max):
    """Solves for the annihilation plateau radius.

    Parameters
    ----------
    r_tr : float
        Truncation radius, kpc.
    lum : float
        Clump luminosity, Hz.
    gamma : float
        Slope.
    rho_max : float
        Maximum DM density, which defines the annihilation plateau, GeV/cm^3.

    Returns
    -------
    Annihilation plateau radius, kpc.
    """
    def f(r_p):  # rescaled luminosity
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


@np.vectorize
def lum(r_p, r_tr, gamma, rho_max=rho_max):
    """Computes the clump's luminosity.

    Parameters
    ----------
    r_p : float
        Radius of annihilation plateau, kpc.
    r_tr : float
        Truncation radius, kpc.
    gamma : float
        Slope.
    rho_max : float
        Maximum DM density, which defines the annihilation plateau, GeV/cm^3.

    Returns
    -------
    Clump luminosity, Hz.
    """
    factor = kpc_to_cm**3 * sv / (2 * fx * mx**2)
    if gamma != 1.5:
        return (4*np.pi*rho_max**2 * (3*r_tr**3*(r_p/r_tr)**(2.*gamma) - 2.*gamma*r_p**3)/(9.-6.*gamma)) * factor
    else:
        return 4./3.*np.pi*r_p**3*rho_max**2*(1. + 3*np.log(r_tr / r_p)) * factor
