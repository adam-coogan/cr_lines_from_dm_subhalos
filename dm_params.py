from utilities import e_high_excess

"""
Use this file to set DM parameters for analysis:

    mx: the DM mass
    sv: thermally-averaged self-annihilation cross section
    fx: 1 if DM is self-conjugate, 2 if not.

These values should be imported into the other analysis files.
"""
mx = e_high_excess  # ~1.5 TeV
sv = 3e-26  # cm^3 / s
fx = 1.
