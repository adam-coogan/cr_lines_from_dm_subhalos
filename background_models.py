from numba import jit


@jit(nopython=True)
def bg_dampe(e):
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


@jit(nopython=True)
def bg_alt(e):
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
