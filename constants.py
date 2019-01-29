import matplotlib.pyplot as plt
import numpy as np
import ctypes
from numba import jit, cfunc, vectorize
from numba.extending import get_cython_function_address
from scipy.integrate import quad
from scipy.interpolate import interp1d

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
# Can fix by finding the mangled function name in:
# >>> from scipy.special.cython_special import __pyx_capi__
# >>> __pyx_capi__
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


@vectorize('float64(float64)', nopython=True)
def lambert_w_series(x):
    return x - x**2 + 3./2.* x**3 - 8./3.*x**4 + 125./24.*x**5


speed_of_light = 3.0e10  # cm / s
kpc_to_cm = 3.086e21  # 3.086 x 10^21 cm / kpc
hbar = 6.582e-16*1/1e9  # GeV s
Hz_to_GeV2_cm3 = 1 / (hbar**2 * speed_of_light**3)
GeV_to_m_sun = 8.96515e-58  # m_sun / GeV
t_universe = 13.772e9 * (365 * 24 * 60 * 60)  # age of universe, s

rho_earth = 0.3  # local DM density, GeV / cm^3
rho_critical = 5.61e-6  # critical density, GeV / cm^3

b0 = 1.0e-16  # GeV/s
e0 = 1.  # GeV

D0 = 1.0e28  # cm^2 / s
delta = 0.7

alpha_em = 1. / 137.
me = 0.501 * 1e-3  # m_e (GeV)

fermi_psf = 0.15 * np.pi / 180.  # arxiv:0902.1089
fermi_psf_solid_angle = 2.*np.pi*(1. - np.cos(fermi_psf))

# Load AMS positron data
pos_frac_es, pos_frac = \
        np.loadtxt("data/fluxes/positron_fraction_ams.dat").T[[0, 3]]

# Load AMS fluxes and bins
ams_es, ams_bin_low, ams_bin_high, ams_dflux, ams_stat, ams_syst = \
        np.loadtxt("data/fluxes/electron_positron_flux_ams.dat").T
ams_dflux = ams_dflux * 1e-4
ams_dflux_err = np.sqrt(ams_stat**2 + ams_syst**2) * 1e-4

# Load DAMPE fluxes and bins
dampe_bin_low, dampe_bin_high, dampe_es, dampe_dflux, dampe_stat, dampe_syst =\
        np.loadtxt("data/fluxes/electron_positron_flux_dampe.dat").T
dampe_dflux = dampe_dflux * 1e-4
dampe_bins = np.transpose([dampe_bin_low, dampe_bin_high])
dampe_dflux_err = np.sqrt(dampe_stat**2 + dampe_syst**2) * 1e-4

# Anisotropy bound on e- only from AMS (arxiv:1612.08957)
e_low_aniso_ams, e_high_aniso_ams, aniso_ams = 16., 350., 0.006
# Anisotropy bound on e-+e+ from Fermi (arxiv:1703.01073)
e_low_aniso_fermi, e_high_aniso_fermi, aniso_fermi = np.loadtxt("data/fermi_epm_aniso.csv").T

# Lower and upper limits on bin with the excess
dampe_excess_bin_low, dampe_excess_bin_high = 1318.3, 1513.6
# Integrated flux in "excess" cm^-2 s^-1 sr^-1
dampe_excess_iflux = (dampe_excess_bin_high - dampe_excess_bin_low) * \
        dampe_dflux[np.abs(dampe_es-1400.).argmin()]

def max_dm_density(mx, sv):
    """Maximum possible DM density.
    """
    return mx / sv / t_universe

# Max DM density with fiducial parameters
rho_max = max_dm_density(mx=dampe_excess_bin_high, sv=3e-26)

# Fermi broadband flux sensitivity: max flux of a power law source at the
# detection threshold for any power law. From here.
# (b, l) = (120, 45)
e_g_f_120_45, dphi_de_g_f_120_45 = np.loadtxt("data/fermi/broadband_flux_"
                                              "sensitivity_p8r2_source_v6_all"
                                              "_10yr_zmax100_n10.0_e1.50_ts25"
                                              "_120_045.csv").T
e_g_f_120_45 = e_g_f_120_45 / 1e3 # MeV -> GeV
dphi_de_g_f_120_45 = 624.15091*dphi_de_g_f_120_45 / e_g_f_120_45**2  # erg -> GeV, divide by E^2
fermi_pt_src_sens_120_45 = interp1d(e_g_f_120_45, dphi_de_g_f_120_45, bounds_error=False)
# (b, l) = (0, 0)
e_g_f_0_0, dphi_de_g_f_0_0 = np.loadtxt("data/fermi/broadband_flux_sensitivity"
                                        "_p8r2_source_v6_all_10yr_zmax100_n10."
                                        "0_e1.50_ts25_000_000.txt").T
e_g_f_0_0 = e_g_f_0_0 / 1e3 # MeV -> GeV
dphi_de_g_f_0_0 = 624.15091*dphi_de_g_f_0_0 / e_g_f_0_0**2  # erg -> GeV, divide by E^2
fermi_pt_src_sens_0_0 = interp1d(e_g_f_0_0, dphi_de_g_f_0_0, bounds_error=False)

def plot_obs_helper(bin_ls, bin_rs, vals, errs, ax, label=None, color="r",
                    alpha=0.75, lw=0.75):
    """Plots observations.

    Parameters
    ----------
    bin_ls : numpy array
        Left edges of bins (GeV)
    bin_rs : numpy array
        Right edges of bins (GeV)
    vals : numpy array
        Bin fluxes in (GeV cm^2 s sr)^-1
    errs : numpy array
        Bin error bars
    """
    for i, (bl, br, l, u) in enumerate(zip(bin_ls, bin_rs, vals-errs,
                                           vals+errs)):
        if i != len(bin_ls) - 1:
            ax.fill_between([bl, br], [l, l], [u, u],
                            color=color, alpha=alpha, lw=lw)
        else:
            ax.fill_between([bl, br], [l, l], [u, u],
                            color=color, alpha=alpha, lw=lw, label=label)


def plot_obs(power, ax, highlight_excess_bins=True):
    """Plots DAMPE observations.

    Parameters
    ----------
    power : float
        The y values plotted are E^power dN/dE
    highlight_excess_bins : bool
        True highlights Ge et al's excess bins in green
    """
    obs_color = "goldenrod"
    excess_color = "steelblue"

    # Observations
    plot_obs_helper(dampe_bin_low, dampe_bin_high,
                    dampe_es**power * dampe_dflux,
                    dampe_es**power * dampe_dflux_err,
                    ax, label="DAMPE", alpha=0.3, lw=0.5, color=obs_color)

    if highlight_excess_bins:
        #for excess_idxs, color in zip([range(23, 28), [29]], ['r', 'b']):
        excess_idxs = [29]
        plot_obs_helper(dampe_bin_low[excess_idxs],
                        dampe_bin_high[excess_idxs],
                        dampe_es[excess_idxs]**power *
                        dampe_dflux[excess_idxs],
                        dampe_es[excess_idxs]**power *
                        dampe_dflux_err[excess_idxs],
                        ax, alpha=0.5, lw=0.5, color=excess_color)


@jit(nopython=True)
def b(e):
    """Energy loss coefficient for e+ e- propagation.

    Parameters
    ----------
    e : numpy array
        e+ e- energies in GeV

    Returns
    -------
        Coefficient in GeV / s
    """
    return b0 * (e / e0)**2


@jit(nopython=True)
def D(e):
    """Diffusion coefficient for e+ e- propagation.

    Parameters
    ----------
    e : numpy array
        e+ e- energies in GeV

    Returns
    -------
        Coefficient in cm^2 / s
    """
    return D0 * (e / e0)**delta


@jit(nopython=True)
def t_diff(e, d):
    """Diffusion timescale (s)
    """
    return (d*kpc_to_cm)**2 / D(e)


@jit(nopython=True)
def t_loss(e):
    """Energy loss timescale (s)
    """
    return e / b(e)


# lambda_prop must be less than this quantity
lambda_prop_max = D0 * e0 / (b0 * (1. - delta))  # cm^2


@jit(nopython=True)
def lambda_prop(e, mx):
    """Special combination of diffusion and energy loss coefficients

    Parameters
    ----------
    e : numpy array
        e+ e- energies in GeV
    mx : float
        DM mass in GeV

    Returns
    -------
    lambda : numpy array
        lambda in cm^2
    """
    if e > mx:
        return 0.
    else:
        return D0*((e/e0)**delta*e0**2*mx - e*e0**2*(mx/e0)**delta) / \
            (b0*e*mx - b0*e*mx*delta)


@jit(nopython=True)
def dn_de_gamma_AP(e, mx):
    """FSR spectrum for xbar x -> e+ e- g using the Altarelli-Parisi
    approximation.

    Parameters
    ----------
    e : float
        Photon energy in GeV.
    mx : float
        DM mass in GeV.

    Returns
    -------
    dnde : float
        Spectrum in GeV^-1.
    """
    Q = 2.*mx
    mu_e = me / Q
    x = 2.*e / Q

    if e > mx:
        return 0.
    else:
        coeff = 2.*alpha_em / (np.pi*Q)
        x_term = (1. + (1. - x)**2) / x * (np.log((1. - x) / mu_e**2) - 1.)
        dnde = coeff * x_term

        if dnde > 0:
            return dnde
        else:
            return 0.


def _constrain_ep_spec(dm_flux, bg_flux, excluded_idxs=[], debug_msgs=False):
    """Determines the significance of the e-+e+ flux from DM annihilation in
    other bins.

    Parameters
    ----------
    dm_flux : float -> float
        A function returning the e-+e+ flux from DM annihilation as a function
        of the lepton's energy in GeV.
    bg_flux : float -> float
        A function giving the background flux.
    excluded_idxs : list of ints
        A list specifying indices of bins to ignore. This is useful because Ge
        et al treat several other bins as also having an excess from DM
        annihilating, and thus exclude them from the background fit.

    Returns
    -------
    n_sigma_max : float

        The statistical significance for the DM e-+e+ flux for the bin with the
        most significant excess from DM.
    """
    def dm_iflux(e_low, e_high):
        return quad(lambda e: 2. * dm_flux(e), e_low, e_high, epsabs=0)[0]

    idxs = list(set(range(len(dampe_bins))) - set(excluded_idxs))
    n_sigma_max = 0.

    # Track bin containing largest excess
    e_low_max, e_high_max = np.nan, np.nan

    for (e_low, e_high), flux, err in reversed(zip(dampe_bins[idxs],
                                                   dampe_dflux[idxs],
                                                   dampe_dflux_err[idxs])):
        if e_low < dampe_excess_bin_high:
            # Residual flux
            obs_iflux = (e_high - e_low) * flux
            bg_iflux = quad(bg_flux, e_low, e_high, epsabs=0.)[0]
            residual_iflux = obs_iflux - bg_iflux

            # Error on integrated flux
            obs_iflux_err = (e_high - e_low) * err

            # Compare with flux from DM annihilations in the bin
            n_sigma_bin = (dm_iflux(e_low, e_high) - residual_iflux) / \
                obs_iflux_err

            if n_sigma_bin > n_sigma_max:
                n_sigma_max = n_sigma_bin
                e_low_max, e_high_max = e_low, e_high

    if debug_msgs:
        print("Bin with largest excess: ", e_low_max, e_high_max)

    return n_sigma_max
