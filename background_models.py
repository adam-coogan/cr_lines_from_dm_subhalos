from numba import cfunc, jit
from numba.types import float64, CPointer, intc
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad



def phi_e_bg_dampe(e):
    """Official DAMPE background model for the e+ e- flux

    Parameters
    ----------
    e : float
        e+ e- energie in GeV

    Returns
    -------
    dphi_de : float
        Fluxe in (cm^2 s sr GeV)^-1
    """
    phi0 = 1.62e-4 * 1e-4  # (cm^2 s sr GeV)^-1
    eb = 914  # GeV
    gamma1 = 3.09
    gamma2 = 3.92
    delta = 0.1

    return phi0 * (100. / e)**gamma1 * \
        (1. + (eb/e)**((gamma1 - gamma2) / delta))**(-delta)

phi_e_bg_dampe_jit = jit(phi_e_bg_dampe, nopython=True)
@cfunc(float64(float64))
def phi_e_bg_dampe_cfunc(e):
    return phi_e_bg_dampe_jit(e)
phi_e_bg_dampe_llc = LowLevelCallable(phi_e_bg_dampe_cfunc.ctypes)


def phi_e_bg_alt(e):
    """Background model for the e+ e- flux from Ge et al, arXiv:1712.02744

    Note
    -----
    The data in several bins is excluded from the background fit. See Fig. 1 in
    the reference above.

    Parameters
    ----------
    e : float
        e+ e- energie in GeV

    Returns
    -------
    dphi_de : float
        Fluxe in (cm^2 s sr GeV)^-1
    """
    phi0 = 246.0e-4   # (cm^2 s sr GeV)^-1
    gamma = 3.09
    delta = 10.
    delta_gamma1 = 0.095
    delta_gamma2 = -0.48
    ebr2 = 471.  # GeV
    ebr1 = 50.  # GeV

    return phi0 * e**(-gamma) * \
        (1 + (ebr1 / e)**delta)**(delta_gamma1/delta) * \
        (1 + (e/ebr2)**delta)**(delta_gamma2/delta)

phi_e_bg_alt_jit = jit(phi_e_bg_alt, nopython=True)
@cfunc(float64(float64))
def phi_e_bg_alt_cfunc(e):
    return phi_e_bg_alt_jit(e)
phi_e_bg_alt_llc = LowLevelCallable(phi_e_bg_alt_cfunc.ctypes)


def Phi_e_bg(e_low, e_high, model="dampe"):
    if model == "dampe":
        phi_e_bg_llc = phi_e_bg_dampe_llc
    elif model == "alt":
        phi_e_bg_llc = phi_e_bg_alt_llc
    else:
        raise ValueError("Unknown background model")

    return quad(phi_e_bg_llc, e_low, e_high, epsabs=0, epsrel=1e-5)[0]


I100 = 1.48e-7 * 1e3 # GeV^-1 cm^-2 s^-1 sr^-1
gamma_fermi = 2.31
e_cut = 362.  # GeV

@jit(nopython=True)
def phi_g_egb_fermi(e):
    """Fermi extragalactic gamma ray background flux.

    Returns
    -------
    float
        EGB in GeV^-1 cm^-2 s^-1 sr^-1.
    """
    return I100 * (e / 0.1)**(-gamma_fermi) * np.exp(-e / e_cut)
