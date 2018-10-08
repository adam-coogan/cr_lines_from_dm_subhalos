import numpy as np
from numba import jitclass
from numba import float64

from constants import kpc_to_cm

spec = [
    ("rs", float64),
    ("rhos", float64),
    ("gamma", float64)
]


@jitclass(spec)
class NFWClump(object):
    def __init__(self, rs, rhos, gamma):
        self.rs = rs
        self.rhos = rhos
        self.gamma = gamma

    def rho(self, r):
        return self.rhos * (self.rs / r)**self.gamma * \
                (1. + r / self.rs)**(self.gamma - 3.)

    def luminosity(self, mx, sv=3e-26, fx=2.):
        factor = sv / (2*fx*mx**2)

        rs_cm = kpc_to_cm * self.rs
        L_clump = (4*np.pi*self.rhos**2*rs_cm**3) / \
            (30 - 47*self.gamma + 24*self.gamma**2 -
             4*self.gamma**3)

        return factor * L_clump
