import numpy as np
from numba import cfunc, jit
from numba.types import float64, CPointer, intc
from scipy import LowLevelCallable
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize_scalar

from constants import e0, b0, D0, delta, kpc_to_cm, speed_of_light
from constants import dampe_excess_bin_low, dampe_excess_bin_high
from constants import dampe_excess_iflux, dn_de_gamma_AP, _constrain_ep_spec


def rho_nfw(r, rs, rhos, gamma):
    return rhos * (rs / r)**gamma * (1. + r / rs)**(gamma - 3.)


def rho_tt(r, rho0, Rb, gamma):
    return rho0 / r**gamma * np.exp(-r / Rb)


@jit(nopython=True)
def K_factor(ct, r, d, rs, rhos, gamma):
    if gamma == 1:
        return -(rhos**2*rs**2 *
                 ((rs*(6*d**2 - 12*ct*d*r + 6*r**2 +
                       15*np.sqrt(d**2 - 2*ct*d*r + r**2)*rs + 11*rs**2)) /
                  (np.sqrt(d**2 - 2*ct*d*r + r**2) + rs)**3 +
                  3*np.log(d**2 - 2*ct*d*r + r**2) -
                  6*np.log(np.sqrt(d**2 - 2*ct*d*r + r**2) + rs)))/(6.*d*r)
    else:
        return -((d**2 - 2*ct*d*r + r**2)**(1 - gamma)*rhos**2*rs**2 *
                 (np.sqrt(d**2 - 2*ct*d*r + r**2) + rs)**(-5 + 2*gamma) *
                 (3*d**2*(np.sqrt(d**2 - 2*ct*d*r + r**2) + (5 - 2*gamma)*rs) -
                  6*ct*d*r*(np.sqrt(d**2 - 2*ct*d*r + r**2) +
                            (5 - 2*gamma)*rs) +
                  (10 - 9*gamma + 2*gamma**2)*rs**2 *
                  (3*np.sqrt(d**2 - 2*ct*d*r + r**2) + 3*rs - 2*gamma*rs) +
                  3*r**2*(np.sqrt(d**2 - 2*ct*d*r + r**2) +
                          5*rs - 2*gamma*rs))) / \
                (2.*d*(-2 + gamma)*(-1 + gamma) *
                 (-5 + 2*gamma)*(-3 + 2*gamma)*r)


# This is to be integrated over r
@cfunc(float64(intc, CPointer(float64)))
def dphi_de_integrand_cf(n, xx):
    r = xx[0]
    e = xx[1]
    d = xx[2]
    mx = xx[3]
    rs = xx[4]
    rhos = xx[5]
    sv = xx[6]
    gamma = xx[7]
    fx = xx[8]

    if e > mx:
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
        int_rho2_ct = K_factor(1., r, d, rs, rhos, gamma) - \
            K_factor(-1., r, d, rs, rhos, gamma)
        # Term with purely radial dependence
        r_term = r**2 * np.exp(-(r * kpc_to_cm)**2 / (4. * lam))

        # Put it all together
        return fact_const * fact_e * int_rho2_ct * r_term * \
            speed_of_light * kpc_to_cm**3


dphi_de_integrand = LowLevelCallable(dphi_de_integrand_cf.ctypes)


def dphi_de_e(e, d, mx, rs, rhos, sv=3e-26, gamma=0.5, fx=2):
    """Computes dphi/dE|_{e-} for a DM clump.

    Parameters
    ----------
    e : float or float array
        Electron energies, GeV.
    d : float
        Distance to the clump's center, kpc.
    mx : float
        DM mass, GeV.
    rs : float
        Clump scale radius, kpc.
    rhos : float
        Clump density normalization, GeV/cm^3.
    sv : float
        DM self-annihilation cross section, cm^3/s.
    gamma : float
        NFW power index.
    """
    if e > mx:
        return 0.
    else:
        return quad(dphi_de_integrand, 0, 100.*d,
                    args=(e, d, mx, rs, rhos, sv, gamma, fx),
                    points=[d],
                    epsabs=0,
                    epsrel=1e-5)[0]


@cfunc(float64(intc, CPointer(float64)))
def J_factor_integrand_cf(n, xx):
    """
    Parameters
    ----------
    r, ctmax, d, rs, rhos, gamma
    """
    # Extract arguments
    r = xx[0]
    ctmax = xx[1]
    d = xx[2]
    rs = xx[3]
    rhos = xx[4]
    gamma = xx[5]

    # Solid angle subtended by target
    dOmega = 2*np.pi*(1.-ctmax)

    th_term = K_factor(1, r, d, rs, rhos, gamma) - \
        K_factor(ctmax, r, d, rs, rhos, gamma)

    return 2*np.pi/dOmega * th_term


J_factor_integrand = LowLevelCallable(J_factor_integrand_cf.ctypes)


def J_factor(ctmax, d, rs, rhos, gamma=0.5):
    """Computes J factor for a target region.

    Parameters
    ----------
    ctmax : float
        cos(theta_max), where theta_max is the angular diameter of the target.
    d : float
        Distance to center of DM clump in kpc.
    rs : float
        Clump scale radius in kpc
    rhos : float
        Clump density normalization in GeV / cm^3.
    gamma : float
        NFW power index.

    Returns
    -------
    J : float
        J factor in GeV^2/cm^5
    """
    i1 = quad(J_factor_integrand, 0., d,
              args=(ctmax, d, rs, rhos, gamma),
              points=[d],
              epsabs=0,
              epsrel=1e-5)[0]
    i2 = quad(J_factor_integrand, d, 100.*d,
              args=(ctmax, d, rs, rhos, gamma),
              points=[d],
              epsabs=0,
              epsrel=1e-5)[0]

    return (i1 + i2) * kpc_to_cm


def lum(mx, rs, rhos, sv=3e-26, gamma=0.5, fx=2):
    # cm^3 / s / GeV^2
    factor = sv / (2*fx*mx**2)
    # GeV^2 / cm^3
    rs_cm = kpc_to_cm * rs
    rho2_int = (4 * np.pi * rhos**2 * rs_cm**(3 - 2*gamma)) / \
        ((30 - 47*gamma + 24*gamma**2 - 4*gamma**3) * (1/rs_cm)**(2*gamma))
    # 1 / s

    return factor * rho2_int


def lum_to_rhos(lum, mx, rs, sv=3e-26, gamma=0.5, fx=2):
    coeff_1 = sv / (2*fx*mx**2)
    rs_cm = kpc_to_cm * rs
    coeff_2 = (4*np.pi*rs_cm**(3.-2.*gamma)) / \
        ((30. - 47.*gamma + 24.*gamma**2 - 4.*gamma**3) *
         (1/rs_cm)**(2.*gamma))

    return np.sqrt(lum / (coeff_1 * coeff_2))


def lum_dampe(d, rs, bg_flux, sv=3e-26, gamma=0.5, fx=2):
    mx = dampe_excess_bin_high
    rhos = rhos_dampe(d, rs, bg_flux, sv, gamma, fx)

    return lum(mx, rs, rhos, sv, gamma, fx)


def dphi_de_gamma(e, ctmax, d, mx, rs, rhos, sv=3e-26, gamma=0.5, fx=2):
    """Computes dphi/dE|_gamma for a DM clump.

    Parameters
    ----------
    e : list of floats
        Photon energies in GeV
    ctmax : float
    d : float
        Distance to clump center in kpc
    mx : float
        DM mass in GeV
    rs : float
        Clump scale radius in kpc
    rhos : float
        Clump density normalization in GeV / cm^3
    sv : float
        DM self-annihilation cross section in cm^3 / s
    gamma : float
    fx : int
        1 if DM is self-conjugate, 2 if not.

    Returns
    -------
        Photon flux at earth from target region in (GeV cm^2 s sr)^{-1}
    """
    if e >= mx:
        return 0.
    else:
        dOmega = 2*np.pi*(1.-ctmax)
        J = J_factor(ctmax, d, rs, rhos, gamma)  # GeV^2 / cm^5

        ret_val = dOmega/(4*np.pi) * J * sv / (2.*fx*mx**2) * \
            dn_de_gamma_AP(e, mx)

        return ret_val


def rhos_dampe(d, rs, bg_flux, sv=3e-26, gamma=0.5, fx=2):
    """Get density normalization giving best fit to excess.

    Parameters
    ----------
    d : float
        Distance to the clump's center, kpc.
    rs : float
        Clump scale radius, kpc.
    bg_flux : float -> float
        Background flux in (GeV cm^2 s sr)^-1
    sv : float
        DM self-annihilation cross section, cm^3/s.
    gamma : float
        NFW power index.

    Returns
    -------
    rhos : float
        Density normalization in GeV / cm^3.
    """
    mx = dampe_excess_bin_high

    # Integrated residual flux
    @cfunc(float64(float64))
    def bg_flux_cf(e):
        return bg_flux(e)

    bg_flux_LLC = LowLevelCallable(bg_flux_cf.ctypes)
    residual_iflux = dampe_excess_iflux - quad(bg_flux_LLC,
                                               dampe_excess_bin_low,
                                               dampe_excess_bin_high,
                                               epsabs=0,
                                               epsrel=1e-5)[0]

    # Integrated DM flux
    dm_iflux = 2.*dblquad(dphi_de_integrand,
                          dampe_excess_bin_low, dampe_excess_bin_high,
                          lambda e: 0, lambda e: 100.*d,
                          args=(d, mx, rs, 1., sv, gamma, fx),
                          epsabs=0)[0]

    return np.sqrt(residual_iflux / dm_iflux)


def dphi_de_e_dampe(e, d, rs, bg_flux, sv=3e-26, gamma=0.5, fx=2):
    """Gets differential electron flux by fitting rho_s to the DAMPE excess.
    """
    mx = dampe_excess_bin_high
    rhos = rhos_dampe(d, rs, bg_flux, sv, gamma)

    return np.vectorize(dphi_de_e)(e, d, mx, rs, rhos, sv, gamma, fx)


def dphi_de_gamma_dampe(e, ctmax, d, rs, bg_flux, sv=3e-26, gamma=0.5, fx=2):
    """Gets differential photon flux by fitting rho_s to the DAMPE excess.
    """
    mx = dampe_excess_bin_high
    rhos = rhos_dampe(d, rs, bg_flux, sv, gamma)

    return np.vectorize(dphi_de_gamma)(e, ctmax, d, mx, rs, rhos, sv, gamma,
                                       fx)


def gamma_ray_extent(e, d, mx, rs, rhos, thresh=0.5, sv=3e-26, gamma=0.5,
                     fx=2):
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
    rs : float
        Subhalo scale radius (kpc).
    rhos : float
        Subhalo density normalization (GeV / cm^3).

    Returns
    -------
    th_ext : float
        Radius of observing region such that the flux in the region is equal to
        thresh times the total flux.
    """
    total_flux = dphi_de_gamma(e, np.cos(np.pi), d, mx, rs, rhos, sv, gamma,
                               fx)

    def fn(th):
        return (dphi_de_gamma(e, np.cos(th), d, mx, rs, rhos, sv, gamma, fx) /
                total_flux - thresh)**2

    if not np.isnan(rhos):
        return minimize_scalar(fn, bounds=(0, np.pi)).x
    else:
        return np.nan


def constrain_ep_spec(d, mx, rs, rhos, bg_flux, sv=3e-26, gamma=0.5, fx=2,
                      excluded_idxs=[]):
    def dm_flux(e):
        return dphi_de_e(e, d, mx, rs, rhos, sv, gamma, fx)

    return _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs)
