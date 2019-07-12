import numpy as np


def wrap(chi_1, chi_2):
    while chi_1 - chi_2 > np.pi:
        chi_1 = chi_1 - 2.0 * np.pi
    while chi_1 - chi_2 < -np.pi:
        chi_1 = chi_1 + 2.0 * np.pi
    return chi_1


def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2 * np.pi
    while x > 2 * np.pi:
        x -= 2 * np.pi
    return x


def wrap_vector(x):
    xwrap = np.array(np.mod(x, 2 * np.pi))
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    if np.size(xwrap) == 1:
        return float(xwrap)
    else:
        return xwrap
