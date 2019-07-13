import numpy as np
from hummingbird.tools.rotations import Quaternion2Euler
from control import TransferFunction as TF
from hummingbird.parameters import aerosonde_parameters as mav_p
import pickle as pkl


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    rho = mav_p.rho
    S = mav_p.S_wing
    Va = mav._Va
    b = mav_p.b
    beta = mav._beta
    alpha = mav._alpha
    c = mav_p.c
    Jy = mav_p.Jy
    g = mav_p.gravity
    mass = mav_p.mass

    a_phi_1 = 0.5 * rho * (Va ** 2) * S * b * mav_p.C_p_p * b / (2 * Va)
    a_phi_2 = 0.5 * rho * (Va ** 2) * S * b * mav_p.C_p_delta_a
    T_phi_delta_a = TF([a_phi_2], [1, -a_phi_1, 0])
    T_chi_phi = TF(np.array([g / Va]), [1, 0])

    beta_dr = (rho * Va * S) / (2 * mav_p.mass * np.cos(beta))
    a_beta1 = -beta_dr * mav_p.C_Y_beta
    a_beta2 = beta_dr * mav_p.C_Y_delta_r
    T_beta_delta_r = TF([a_beta2], [1, a_beta1])

    theta_de = rho * Va ** 2 * c * S / (2 * Jy)
    a_theta1 = -theta_de * mav_p.C_m_q * c / (2 * Va)
    a_theta2 = -theta_de * mav_p.C_m_alpha
    a_theta3 = theta_de * mav_p.C_m_delta_e
    T_theta_delta_e = TF([a_theta3], [1, a_theta1, a_theta2])

    T_h_theta = TF([Va], [1, 0])
    _, theta, _ = Quaternion2Euler(trim_state[6:10])
    T_h_Va = TF([theta], [1, 0])

    C_Ds = mav_p.C_D_0 + mav_p.C_D_alpha * alpha + mav_p.C_D_delta_e * trim_input[0]
    a_V1 = ((rho * Va * S * C_Ds) - dT_dVa(mav, Va, trim_input[3])) / mass
    a_V2 = dT_ddelta_t(mav, Va, trim_input[3]) / mass
    a_V3 = g * np.cos(theta - alpha)
    T_Va_delta_t = TF([a_V2], [1, a_V1])
    T_Va_theta = TF([-a_V3], [1, a_V1])

    with open("trim.pkl", 'wb') as f:
        vals = [trim_state, trim_input, a_phi_1, a_phi_2,
                a_beta1, a_beta2, a_theta1, a_theta2, a_theta3]
        pkl.dump(vals, f)

    data = []
    with open("trim.pkl", 'rb') as f:
        data = pkl.load(f)

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    epsilon = 0.01
    fp1, _ = mav._prop_thrust_torque(delta_t, Va - epsilon)
    fp2, _ = mav._prop_thrust_torque(delta_t, Va + epsilon)

    dThrust = (fp2 - fp1) / (2 * epsilon)
    return dThrust


def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    epsilon = 0.001
    fp1, _ = mav._prop_thrust_torque(delta_t - epsilon, Va)
    fp2, _ = mav._prop_thrust_torque(delta_t + epsilon, Va)

    dThrust = (fp2 - fp1) / (2 * epsilon)
    return dThrust

# def compute_ss_model(mav, trim_state, trim_input):
# return A_lon, B_lon, A_lat, B_lat

# def euler_state(x_quat):
#     # convert state x with attitude represented by quaternion
#     # to x_euler with attitude represented by Euler angles
#      return x_euler

# def quaternion_state(x_euler):
#     # convert state x_euler with attitude represented by Euler angles
#     # to x_quat with attitude represented by quaternions
#     return x_quat

# def f_euler(mav, x_euler, input):
#     # return 12x1 dynamics (as if state were Euler state)
#     # compute f at euler_state
#     return f_euler_

# def df_dx(mav, x_euler, input):
#     # take partial of f_euler with respect to x_euler
#     return A

# def df_du(mav, x_euler, delta):
#     # take partial of f_euler with respect to delta
#     return B
