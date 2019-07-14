from warnings import warn

import numpy as np

from hummingbird.parameters.constants import StateEnum, BodyFrameEnum
from hummingbird.physics.generics import compute_gamma


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
