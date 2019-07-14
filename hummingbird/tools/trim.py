import numpy as np
from scipy.optimize import minimize
from hummingbird.tools.rotations import Euler2Quaternion, rotz
from hummingbird.parameters.constants import StateQuatEnum as SQE, StateEnum as SE, PhysicalConstants as pc
from hummingbird.maths.gradient_descent import gradient_descent
from hummingbird.physics.kinematics import compute_kinematics_from_quat


def compute_trim(mav, Va, gamma, turn_radius=np.inf):
    # define initial state and input
    e = Euler2Quaternion([0, gamma, 0])
    state0 = np.array([0,  # (0)
                       0,  # (1)
                       -100,  # (2)
                       Va,  # (3)
                       0,  # (4)
                       0,  # (5)
                       e[0],  # (6)
                       e[1],  # (7)
                       e[2],  # (8)
                       e[3],  # (9)
                       0,  # (10)
                       0,  # (11)
                       0])  # (12)
    delta_e = 0
    delta_a = 0
    delta_r = 0
    delta_t = 0.5
    delta0 = np.array([delta_e, delta_a, delta_r, delta_t])

    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints

    if turn_radius == np.inf:
        cons = ({'type': 'eq',
                 'fun': lambda x: np.array([
                     x[3] ** 2 + x[4] ** 2 + x[5] ** 2 - Va ** 2,  # magnitude of velocity vector is Va
                     x[4],  # v=0, force side velocity to be zero
                     x[6] ** 2 + x[7] ** 2 + x[8] ** 2 + x[9] ** 2 - 1.,  # force quaternion to be unit length
                     x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                     x[9],  # e3=0
                     x[10],  # p=0  - angular rates should all be zero
                     x[11],  # q=0
                     x[12],  # r=0
                 ]),
                 'jac': lambda x: np.array([
                     [0., 0., 0., 2 * x[3], 2 * x[4], 2 * x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 2 * x[6], 2 * x[7], 2 * x[8], 2 * x[9], 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                 ])
                 })
    else:
        cons = ({'type': 'eq',
                 'fun': lambda x: np.array([
                     x[3] ** 2 + x[4] ** 2 + x[5] ** 2 - Va ** 2,  # magnitude of velocity vector is Va
                     x[4],  # v=0, force side velocity to be zero
                     x[6] ** 2 + x[7] ** 2 + x[8] ** 2 + x[9] ** 2 - 1.,  # force quaternion to be unit length
                     # x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                     # x[9],  # e3=0
                     x[10],  # p=0  - angular rates should all be zero
                     x[11],  # q=0
                     # x[12],  # r=0
                 ]),
                 'jac': lambda x: np.array([
                     [0., 0., 0., 2 * x[3], 2 * x[4], 2 * x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 2 * x[6], 2 * x[7], 2 * x[8], 2 * x[9], 0., 0., 0., 0., 0., 0., 0.],
                     # [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                 ])
                 })

    # # solve the minimization problem to find the trim states and inputs
    # res = minimize(trim_objective, x0, method='SLSQP', args=(mav, Va, gamma, turn_radius),
    #                constraints=cons, options={'ftol': 1e-15, 'disp': True})

    res = minimize(trim_objective, x0, method='L-BFGS-B', args=(mav, Va, gamma, turn_radius),
                   constraints = cons, options={'ftol': 1e-15, 'disp': True, 'iprint': 1, 'maxls': 50, 'gtol': 1e-010})

    # extract trim state and input and return
    print(res)
    trim_state = np.array(res.x[0:13])
    trim_input = np.array(res.x[13:17])

    return trim_state, trim_input


# objective function to be minimized
def trim_objective(x, mav, Va, gamma, turn_radius):
    state = x[:13]
    x[6:10] = x[6:10] / np.linalg.norm(x[6:10])
    mav._state = state
    wrench = mav.forces_moments(state, x[13:], mav.mav_p)
    xd = compute_kinematics_from_quat(wrench, x, mav.mav_p)
    if turn_radius == np.inf:
        xd_star = np.array([0., 0., -Va * np.sin(gamma),
                            0., 0., 0.,
                            0., 0., 0., 0.,
                            0., 0., 0.])
    else:
        quatd_star = Euler2Quaternion([0., 0., Va * np.cos(gamma) / turn_radius])
        xd_star = np.array([0., 0., -Va * np.sin(gamma),
                            0., 0., 0.,
                            quatd_star[0], quatd_star[1], quatd_star[2], quatd_star[3],
                            0., 0., 0.])
    error = (xd_star - xd)[2:]
    J = np.linalg.norm(error) ** 2 + np.linalg.norm(wrench) ** 2
    return J


def compute_trimmed_states_inputs(params, Va, gamma, turn_radius, alpha, beta, phi):
    P = params

    # TODO: Is it clearer to use R var name instead of turn_radius?
    R = turn_radius
    g = pc.g
    mass = params.mass
    Jx = params.Jx
    Jy = params.Jy
    Jz = params.Jz
    Jxz = params.Jxz

    # TODO: Gamma computation should be moved to parameters class, as it is constant during
    # flight.
    gamma_0 = Jx * Jz - Jxz**2
    gamma_1 = (Jxz * (Jx - Jy + Jz)) / gamma_0
    gamma_2 = (Jz * (Jz - Jy) + Jxz**2) / gamma_0
    gamma_3 = Jz / gamma_0
    gamma_4 = Jxz / gamma_0
    #gamma_5 = (Jz - Jx)/Jy
    #gamma_6 = Jxz/Jy
    gamma_7 = ((Jx - Jy) * Jx + Jxz**2) / gamma_0
    gamma_8 = Jx / gamma_0

    S = params.S
    b = params.b
    c = params.c
    rho = params.rho
    e = params.e
    S_prop = params.S_prop
    k_motor = params.k_motor

    x = np.zeros((SE.size,), dtype=np.double)
    x[SE.u] = Va * np.cos(alpha) * np.cos(beta)
    x[SE.v] = Va * np.sin(beta)
    x[SE.w] = Va * np.sin(alpha) * np.cos(beta)
    theta = alpha + gamma
    x[SE.phi] = phi
    x[SE.theta] = theta
    x[SE.p] = -Va / R * np.sin(theta)
    x[SE.q] = Va / R * np.sin(phi) * np.cos(theta)
    x[SE.r] = Va / R * np.cos(phi) * np.cos(theta)
    #u = x[3]
    v = x[SE.v]
    w = x[SE.w]
    p = x[SE.p]
    q = x[SE.q]
    r = x[SE.r]

    C0 = 0.5 * rho * Va**2 * S

    def delta_e():
        C1 = (Jxz * (p**2 - r**2) + (Jx - Jz) * p * r) / (C0 * c)
        Cm0 = params.Cm0
        Cm_alpha = params.Cm_alpha
        Cm_q = params.Cm_q
        Cm_delta_e = params.Cm_delta_e
        return (C1 - Cm0 - Cm_alpha * alpha - Cm_q * c * q * 0.5 / Va) / Cm_delta_e
    delta_e = delta_e()

    def delta_t():
        CL0 = params.CL0
        CL_alpha = params.CL_alpha
        M = params.M
        alpha_0 = params.alpha_0
        CD_alpha = params.CD_alpha
        CD_p = params.CD_p
        CD_q = params.CD_q
        CL_q = params.CL_q
        CL_delta_e = params.CL_delta_e
        CD_delta_e = params.CD_delta_e
        C_prop = params.C_prop
        c1 = np.exp(-M * (alpha - alpha_0))
        c2 = np.exp(M * (alpha + alpha_0))
        sigmoid_alpha = (1 + c1 + c2) / ((1 + c1) * (1 + c2))
        CL_alpha_NL = (1. - sigmoid_alpha) * (CL0 + CL_alpha * alpha) + sigmoid_alpha * \
                      2. * np.sign(alpha) * np.sin(alpha) * \
                      np.sin(alpha) * np.cos(alpha)
        AR = b**2 / S
        CD_alpha_NL = CD_p + (CL0 + CL_alpha * alpha)**2 / (np.pi * e * AR)
        CX = -CD_alpha_NL * np.cos(alpha) + CL_alpha_NL * np.sin(alpha)
        CX_delta_e = -CD_delta_e * \
                     np.cos(alpha) + CL_delta_e * np.sin(alpha)
        CX_q = -CD_q * np.cos(alpha) + CL_q * np.sin(alpha)
        C2 = 2 * mass * (-r * v + q * w + g * np.sin(theta))
        C3 = -2 * C0 * (CX + CX_q * c * q * 0.5 /
                        Va + CX_delta_e * delta_e)
        C4 = rho * C_prop * S_prop * k_motor**2
        return np.sqrt((C2 + C3) / C4 + Va**2 / k_motor**2)
    delta_t = delta_t()

    def delta_a_delta_r():
        Cl_delta_a = params.Cl_delta_a
        Cn_delta_a = params.Cn_delta_a
        Cl_delta_r = params.Cl_delta_r
        Cn_delta_r = params.Cn_delta_r
        Cl0 = params.Cl0
        Cn0 = params.Cn0
        Cl_p = params.Cl_p
        Cn_p = params.Cn_p
        Cl_beta = params.Cl_beta
        Cn_beta = params.Cn_beta
        Cl_r = params.Cl_r
        Cn_r = params.Cn_r

        # TODO: Create a specific function to compute aerodynamic coeffs (and test it)
        Cp_delta_a = gamma_3 * Cl_delta_a + gamma_4 * Cn_delta_a
        Cp_delta_r = gamma_3 * Cl_delta_r + gamma_4 * Cn_delta_r
        Cr_delta_a = gamma_4 * Cl_delta_a + gamma_8 * Cn_delta_a
        Cr_delta_r = gamma_4 * Cl_delta_r + gamma_8 * Cn_delta_r
        Cp_0 = gamma_3 * Cl0 + gamma_4 * Cn0
        Cp_beta = gamma_3 * Cl_beta + gamma_4 * Cn_beta
        Cp_p = gamma_3 * Cl_p + gamma_4 * Cn_p
        Cp_r = gamma_3 * Cl_r + gamma_4 * Cn_r
        Cr_0 = gamma_4 * Cl0 + gamma_8 * Cn0
        Cr_beta = gamma_4 * Cl_beta + gamma_8 * Cn_beta
        Cr_p = gamma_4 * Cl_p + gamma_8 * Cn_p
        Cr_r = gamma_4 * Cl_r + gamma_8 * Cn_r

        C5 = (-gamma_1 * p * q + gamma_2 * q * r) / (C0 * b)
        C6 = (-gamma_7 * p * q + gamma_1 * q * r) / (C0 * b)
        v0 = C5 - Cp_0 - Cp_beta * beta - Cp_p * \
             b * p * 0.5 / Va - Cp_r * b * r * 0.5 / Va
        v1 = C6 - Cr_0 - Cr_beta * beta - Cr_p * \
             b * p * 0.5 / Va - Cr_r * b * r * 0.5 / Va
        v = [v0, v1]
        B = np.array([[Cp_delta_a, Cp_delta_r], [
            Cr_delta_a, Cr_delta_r]], dtype=np.double)
        if Cp_delta_r == 0. and Cr_delta_r == 0.:
            return [v0 / B[0][0], 0.]
        elif Cp_delta_a == 0. and Cr_delta_a == 0.:
            return [0.0, v1 / B[1][1]]
        else:
            _delta_a_delta_r = np.dot(np.linalg.inv(B), v)
            return _delta_a_delta_r[0], _delta_a_delta_r[1]

    delta_a, delta_r = delta_a_delta_r()

    control_inputs = [delta_e, delta_a, delta_r, delta_t]

    return x, control_inputs


def compute_gradient_descent_trim(params, f, Va, gamma, turn_radius, max_iters=5000, epsilon=1e-8, kappa=1e-6):

    def J(f, alpha, beta, phi):
        """ Cost function used for gradient descent.

        .. TODO:: Externalize gradient descent cost function, and test it.

        :param alpha:
        :param beta:
        :param phi:
        :return:
        """
        trimmed_state, trimmed_control = compute_trimmed_states_inputs(params, Va, gamma, turn_radius,
                                                                       alpha, beta, phi)
        xd = f(trimmed_state, trimmed_control)
        xd[0] = 0.
        xd[1] = 0.

        xd_star = np.zeros((13,), dtype=np.double)
        xd_star[2] = -Va * np.sin(gamma)
        chi_dot = Va / turn_radius * np.cos(gamma)
        xd_star[6:10] = Euler2Quaternion([0., 0., chi_dot])
        J = np.linalg.norm(xd_star[2:] - xd[2:])**2
        return J

    alpha_0 = -0.0
    beta_0 = 0.
    phi_0 = 0.

    alpha, beta, phi = gradient_descent(J, f, alpha_0, beta_0, phi_0, max_iters, epsilon, kappa)
    trimmed_state, trimmed_control = compute_trimmed_states_inputs(params, Va, gamma, turn_radius, alpha, beta, phi)

    return trimmed_state, trimmed_control
