import numpy as np


def compute_gravitational_force(cphi, sphi, ctheta, stheta, mass, g):
    """ Gravitational force expressed in body frame
    """

    fx = -mass * g * stheta
    fy = mass * g * ctheta * sphi
    fz = mass * g * ctheta * cphi
    return fx, fy, fz


def compute_gamma(attrs):
    Jx = attrs.Jx
    Jy = attrs.Jy
    Jz = attrs.Jz
    Jxz = attrs.Jxz
    gamma_0 = Jx * Jz - Jxz ** 2
    gamma_1 = (Jxz * (Jx - Jy + Jz)) / gamma_0
    gamma_2 = (Jz * (Jz - Jy) + Jxz ** 2) / gamma_0
    gamma_3 = Jz / gamma_0
    gamma_4 = Jxz / gamma_0
    gamma_5 = (Jz - Jx) / Jy
    gamma_6 = Jxz / Jy
    gamma_7 = ((Jx - Jy) * Jx + Jxz ** 2) / gamma_0
    gamma_8 = Jx / gamma_0

    gamma = np.array([gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6, gamma_7,
                      gamma_8])

    return gamma


def simple_propeller_forces(params, Va, delta_t):
    rho = params.rho
    S_prop = params.S_prop
    C_prop = params.C_prop
    k_motor = params.k_motor
    fx = 0.5 * rho * S_prop * C_prop * \
         (k_motor ** 2 * delta_t ** 2 - Va ** 2)
    fy = 0.
    fz = 0.
    return fx, fy, fz


def simple_propeller_torques(params, delta_t):
    kT_p = params.kT_p
    kOmega = params.kOmega
    l = -kT_p * kOmega ** 2 * delta_t ** 2
    m = 0.
    n = 0.
    return l, m, n


def propeller_thrust_torque(dt, Va, mav_p):
    # propeller thrust and torque
    rho = mav_p.rho
    D = mav_p.D_prop

    V_in = mav_p.V_max * dt
    a = rho * D ** 5 * mav_p.C_Q0 / (2 * np.pi) ** 2
    b = rho * D ** 4 * mav_p.C_Q1 * Va / (2 * np.pi) + mav_p.KQ ** 2 / mav_p.R_motor
    c = rho * D ** 3 * mav_p.C_Q2 * Va ** 2 - \
        (mav_p.KQ * V_in) / mav_p.R_motor + mav_p.KQ * mav_p.i0
    radicand = b ** 2 - 4 * a * c
    if radicand < 0:
        radicand = 0
    Omega_op = (-b + np.sqrt(radicand)) / (2 * a)

    J_op = 2 * np.pi * Va / (Omega_op * D)
    C_T = mav_p.C_T2 * J_op ** 2 + mav_p.C_T1 * J_op + mav_p.C_T0
    C_Q = mav_p.C_Q2 * J_op ** 2 + mav_p.C_Q1 * J_op + mav_p.C_Q0
    n = Omega_op / (2 * np.pi)
    fp_x = rho * n ** 2 * D ** 4 * C_T
    Mp_x = rho * n ** 2 * D ** 5 * C_Q

    return fp_x, Mp_x