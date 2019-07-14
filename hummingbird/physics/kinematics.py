from warnings import warn

import numpy as np

from hummingbird.parameters.constants import StateEnum, BodyFrameEnum
from hummingbird.physics.generics import compute_gamma
from hummingbird.tools.rotations import Quaternion2Rotation


def compute_kinematics(forces, moments, x, attrs):
    """ Integrates dynamics and computes the state derivative.
    """
    gamma = compute_gamma(attrs)

    # TODO: Check is a function deserializing the state vector would be better

    u = x[StateEnum.u]
    v = x[StateEnum.v]
    w = x[StateEnum.w]
    velocity = np.array([u, v, w])

    phi = x[StateEnum.phi]
    theta = x[StateEnum.theta]
    psi = x[StateEnum.psi]
    attitude = np.array([phi, theta, psi])

    p = x[StateEnum.p]
    q = x[StateEnum.q]
    r = x[StateEnum.r]
    rates = np.array([p, q, r])

    fx = forces[BodyFrameEnum.x]
    fy = forces[BodyFrameEnum.y]
    fz = forces[BodyFrameEnum.z]
    forces = np.array([fx, fy, fz])

    l = moments[BodyFrameEnum.x]
    m = moments[BodyFrameEnum.y]
    n = moments[BodyFrameEnum.z]

    trigonometric_attitude = compute_trigonometric_attitude_vector(attitude)
    Rv_b = compute_rotation_matrix_body_to_vehicle_frame(trigonometric_attitude)
    Sv_b = compute_derivate_rotation_matrix_body_to_vehicle_frame(trigonometric_attitude)

    mass = attrs.mass
    inv_mass = 1. / mass
    Jy = attrs.Jy
    inv_Jy = 1. / Jy

    d_pos = np.dot(Rv_b, velocity)
    d_vel = np.cross(velocity, rates) + inv_mass * forces
    d_att = np.dot(Sv_b, rates)

    dp = gamma[1] * p * q - gamma[2] * q * r + gamma[3] * l + gamma[4] * n
    dq = gamma[5] * p * r - gamma[6] * (p * p - r * r) + m * inv_Jy
    dr = gamma[7] * p * q - gamma[1] * q * r + gamma[4] * l + gamma[8] * n
    d_rates = np.array([dp, dq, dr])

    dx = np.concatenate((d_pos, d_vel, d_att, d_rates))

    return dx


def compute_kinematics_from_quat(forces_moments, state, params):
    """
    for the dynamics xdot = f(x, u), returns f(x, u)
    """
    # extract the states
    pn = state[0]
    pe = state[1]
    pd = state[2]
    u = state[3]
    v = state[4]
    w = state[5]
    e0 = state[6]
    e1 = state[7]
    e2 = state[8]
    e3 = state[9]
    p = state[10]
    q = state[11]
    r = state[12]
    #   extract forces/moments
    fx = forces_moments[0]
    fy = forces_moments[1]
    fz = forces_moments[2]
    l = forces_moments[3]
    m = forces_moments[4]
    n = forces_moments[5]

    R_vb = Quaternion2Rotation(np.array([e0, e1, e2, e3]))  # body->vehicle

    # position kinematics
    pn_dot, pe_dot, pd_dot = R_vb @ np.array([u, v, w])

    # position dynamics
    vec_pos = np.array([r * v - q * w, p * w - r * u, q * u - p * v])
    u_dot, v_dot, w_dot = vec_pos + 1 / params.mass * np.array([fx, fy, fz])

    # rotational kinematics
    mat_rot = np.array([[0, -p, -q, -r],
                        [p, 0, r, -q],
                        [q, -r, 0, p],
                        [r, q, -p, 0]])
    e0_dot, e1_dot, e2_dot, e3_dot = 0.5 * mat_rot @ np.array([e0, e1, e2, e3])

    # rotational dynamics
    g1 = params.gamma1
    g2 = params.gamma2
    g3 = params.gamma3
    g4 = params.gamma4
    g5 = params.gamma5
    g6 = params.gamma6
    g7 = params.gamma7
    g8 = params.gamma8

    vec_rot = np.array([g1 * p * q - g2 * q * r, g5 * p * r - g6 * (p ** 2 - r ** 2), g7 * p * q - g1 * q * r])
    vec_rot2 = np.array([g3 * l + g4 * n, m / params.Jy, g4 * l + g8 * n])

    p_dot, q_dot, r_dot = vec_rot + vec_rot2

    # collect the derivative of the states
    x_dot = np.array([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                      e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot])

    return x_dot


def compute_trigonometric_attitude_vector(attitude):
    phi = attitude[0]
    theta = attitude[1]
    psi = attitude[2]

    cr = np.cos(phi)
    sr = np.sin(phi)

    cp = np.cos(theta)
    if np.isclose(cp, 0., 1e-5):
        warn("[Dynamics] Euler angles are not defined if theta is close to +/- 90°.")
        cp = 1e-5

    sp = np.sin(theta)
    tp = np.tan(theta)

    cy = np.cos(psi)
    sy = np.sin(psi)

    return [cr, sr, cp, sp, tp, cy, sy]


def compute_rotation_matrix_body_to_vehicle_frame(trigonometric_attitude):
    cr = trigonometric_attitude[0]
    sr = trigonometric_attitude[1]

    cp = trigonometric_attitude[2]
    sp = trigonometric_attitude[3]

    cy = trigonometric_attitude[5]
    sy = trigonometric_attitude[6]

    Rv_b = np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])

    return Rv_b


def compute_derivate_rotation_matrix_body_to_vehicle_frame(trigonometric_attitude):
    cr = trigonometric_attitude[0]
    sr = trigonometric_attitude[1]

    cp = trigonometric_attitude[2]
    tp = trigonometric_attitude[4]

    Sv_b = np.array([[1., sr * tp, cr * tp],
                     [0., cr, -sr],
                     [0., sr / cp, cr / cp]])

    return Sv_b


def R_vb(attitude):
    trigonometric_attitude = compute_trigonometric_attitude_vector(attitude)
    return compute_rotation_matrix_body_to_vehicle_frame(trigonometric_attitude)


def R_bv(attitude):
    return R_vb(attitude).T
