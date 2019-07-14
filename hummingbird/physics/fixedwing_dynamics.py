from functools import partial

from hummingbird.parameters.constants import ActuatorEnum, PhysicalConstants as pc, StateEnum
from hummingbird.physics.dynamics_base import *
from hummingbird.maths.generics import compute_sigmoid
from hummingbird.physics.kinematics import compute_kinematics, compute_kinematics_from_quat
from hummingbird.physics.generics import *
from hummingbird.maths.gradient_descent import gradient_descent


import numpy as np
from hummingbird.message_types.msg_state import MsgState
from hummingbird.parameters.aerosonde_parameters import MavParameters
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.physics.generics import propeller_thrust_torque
from hummingbird.tools.rotations import Quaternion2Rotation, Quaternion2Euler
from math import asin, exp, acos
from hummingbird.parameters.constants import StateQuatEnum as SQE
from hummingbird.maths.generics import normalize_vector

np.set_printoptions(suppress=True, precision=4)


class FixedwingDynamics(DynamicsBase):
    """ Fixed Wing UAV Dynamics

        :param x0: Initial state vector
        :type x0: vector [12x1]
        :param t0: Initial time of the simulation
        :type t0: double [s]
        :param dt_integration: Sampling time of the integrator
        :type dt_integration: double [s]
    """

    def __init__(self,
                 x0=MavParameters().initial_state,
                 t0=sim_p.start_time,
                 dt_integration=sim_p.dt_simulation,
                 mav_p=MavParameters):
        self.mav_p = mav_p
        super(FixedwingDynamics, self).__init__(x0, t0, dt_integration)
        self.t0 = t0
        self.set_integrator(FixedwingDynamics.dynamics,
                            'dop853', jac=None, rtol=1e-8)
        self.partial_forces_moments = partial(
                FixedwingDynamics.forces_moments, params=self.mav_p)
        self._control_inputs = [0., 0., 0., 0.]

        self._dt_simulation = dt_integration

        self.R_vb = Quaternion2Rotation(self._state[6:10])  # Rotation body->vehicle
        self.R_bv = np.copy(self.R_vb).T  # vehicle->body

        self._forces = np.zeros(3)
        self._wind = np.zeros(3)  # wind in NED frame in meters/

        self._Va = self.mav_p.u0
        self._alpha = 0
        self._beta = 0
        self.true_state = MsgState()

    def reset_state(self):
        self._state = np.array([self.mav_p.pn0,  # (0)
                                self.mav_p.pe0,  # (1)
                                self.mav_p.pd0,  # (2)
                                self.mav_p.u0,  # (3)
                                self.mav_p.v0,  # (4)
                                self.mav_p.w0,  # (5)
                                self.mav_p.e0,  # (6)
                                self.mav_p.e1,  # (7)
                                self.mav_p.e2,  # (8)
                                self.mav_p.e3,  # (9)
                                self.mav_p.p0,  # (10)
                                self.mav_p.q0,  # (11)
                                self.mav_p.r0])  # (12)

    def update(self, delta, wind=np.zeros(6)):
        self.control_inputs = delta
        self._wind = wind
        if self.control_inputs is not None:
            t = self._dt_simulation + self.integrator.t
            self.integrate(t)
        else:
            raise Exception('set control inputs first')

        self._forces = self.forces_moments(self._state, self.control_inputs, self.mav_p)[0:3]
        self.R_vb = Quaternion2Rotation(self._state[6:10])  # Rotation body->vehicle
        self.R_bv = np.copy(self.R_vb).T  # vehicle->body
        self._update_true_state()

    @staticmethod
    def forces_moments(x, delta, params):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_e, delta_a, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        de = delta[0]
        da = delta[1]
        dr = delta[2]
        dt = delta[3]
        P = params

        # Gravity
        R_bv = Quaternion2Rotation(x[SQE.e0:SQE.e3+1]).T
        fg = R_bv @ np.array([0, 0, P.mass * P.gravity])

        # Air data
        Va = np.linalg.norm(x[SQE.u:SQE.w+1])
        alpha, beta = FixedwingDynamics.compute_alpha_beta(Va, x[SQE.u], x[SQE.v], x[SQE.w])
        # Dynamic pressure
        p_dyn = compute_dynamic_pressure(P.rho, Va)

        # Propeller
        fp, Mp = propeller_thrust_torque(dt, Va, P)

        # Aerodynamic forces/moments

        # Longitudinal
        M = P.M
        alpha = alpha
        alpha0 = P.alpha0
        Va = Va
        q_S = p_dyn * P.S_wing
        q = x[SQE.q]
        c = P.c

        sigma_alpha = (1 + exp(-M * (alpha - alpha0)) + exp(M * (alpha + alpha0))) / \
                      ((1 + exp(-M * (alpha - alpha0))) * (1 + exp(M * (alpha + alpha0))))
        CL_alpha = (1 - sigma_alpha) * (P.C_L_0 + P.C_L_alpha * alpha) + \
                   sigma_alpha * (2 * np.sign(alpha) * (np.sin(alpha) ** 2) * np.cos(alpha))
        F_lift = q_S * (
                    CL_alpha + P.C_L_q * (c / (2 * Va)) * q + P.C_L_delta_e * de)
        CD_alpha = P.C_D_p + ((P.C_L_0 + P.C_L_alpha * alpha) ** 2) / \
                   (np.pi * P.e * P.AR)
        F_drag = q_S * (
                    CD_alpha + P.C_D_q * (c / (2 * Va)) * q + P.C_D_delta_e * de)
        m = q_S * c * (P.C_m_0 + P.C_m_alpha * alpha +
                                             P.C_m_q * (c / (2. * Va)) * q + P.C_m_delta_e * de)

        # Lateral
        b = P.b
        p = x[SQE.p]
        r = x[SQE.r]
        rho = P.rho
        S = P.S_wing

        # Calculating fy
        fa_y = q_S * (P.C_Y_0 + P.C_Y_beta * beta +
               P.C_Y_p * (b / (2 * Va)) * p + P.C_Y_r *
               (b / (2 * Va)) * r + P.C_Y_delta_a * da +
               P.C_Y_delta_r * dr)

        # Calculating l
        l = q_S * b * (P.C_ell_0 + P.C_ell_beta * beta +
            P.C_ell_p * (b / (2 * Va)) * p + P.C_ell_r * (
                        b / (2 * Va)) *
            r + P.C_ell_delta_a * da + P.C_ell_delta_r * dr)

        # Calculating n
        n = q_S * b * (P.C_n_0 + P.C_n_beta * beta +
            P.C_n_p * (b / (2 * Va)) * p + P.C_n_r * (
                        b / (2 * Va)) * r +
            P.C_n_delta_a * da + P.C_n_delta_r * dr)

        # Combining into force/moment arrays
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        [fa_x, fa_z] = np.array([[ca, -sa], [sa, ca]]) @ np.array([-F_drag, -F_lift])
        fa = np.array([fa_x, fa_y, fa_z])
        Ma = np.array([l, m, n])

        # Summing forces and moments
        [fx, fy, fz] = fg + fa + fp
        [Mx, My, Mz] = Ma + Mp
        return np.array([fx, fy, fz, Mx, My, Mz])

    def update_true_state_from_forces_moments(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._dt_simulation

        dx = compute_kinematics_from_quat(forces_moments, self._state, self.mav_p)
        k1 = compute_kinematics_from_quat(forces_moments, self._state, self.mav_p)
        k2 = compute_kinematics_from_quat(forces_moments, self._state + time_step / 2. * k1, self.mav_p)
        k3 = compute_kinematics_from_quat(forces_moments, self._state + time_step / 2. * k2, self.mav_p)
        k4 = compute_kinematics_from_quat(forces_moments, self._state + time_step * k3, self.mav_p)
        self._state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # normalize the quaternion
        self._state[SQE.e0:SQE.e3+1] = normalize_vector(self._state[SQE.e0:SQE.e3+1])
        self.R_vb = Quaternion2Rotation(self._state[6:10])  # body->vehicle
        self.R_bv = np.copy(self.R_vb).T  # vehicle->body
        self._update_true_state()

    @staticmethod
    def compute_alpha_beta(Va, u, v, w):
        if u > 0:
            alpha = np.arctan2(w, u)
        elif Va == 0:
            alpha = 0
        else:
            alpha = np.pi / 2
        if Va > 0:
            beta = np.arcsin(v / Va)
        else:
            beta = 0

        return alpha, beta

    @staticmethod
    def dynamics(t, x, params, control_inputs, forces_moments):
        forces_moments = forces_moments(x, control_inputs)

        dx = compute_kinematics_from_quat(forces_moments, x, params)
        return dx

    def compute_trimmed_states_inputs(self, Va, gamma, turn_radius, alpha, beta, phi):
        params = self.mav_p

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

        x = np.zeros((StateEnum.size,), dtype=np.double)
        # TODO: refactor matrix product to make it look like it
        x[StateEnum.u] = Va * np.cos(alpha) * np.cos(beta)
        x[StateEnum.v] = Va * np.sin(beta)
        x[StateEnum.w] = Va * np.sin(alpha) * np.cos(beta)
        theta = alpha + gamma
        x[StateEnum.phi] = phi
        x[StateEnum.theta] = theta
        x[StateEnum.p] = -Va / R * np.sin(theta)
        x[StateEnum.q] = Va / R * np.sin(phi) * np.cos(theta)
        x[StateEnum.r] = Va / R * np.cos(phi) * np.cos(theta)
        #u = x[3]
        v = x[StateEnum.v]
        w = x[StateEnum.w]
        p = x[StateEnum.p]
        q = x[StateEnum.q]
        r = x[StateEnum.r]

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

    def trim(self, Va, gamma, turn_radius, max_iters=5000, epsilon=1e-8, kappa=1e-6):
        R = turn_radius

        def J(alpha, beta, phi):
            """ Cost function used for gradient descent.

            .. TODO:: Externalize gradient descent cost function, and test it.

            :param alpha:
            :param beta:
            :param phi:
            :return:
            """
            trimmed_state, trimmed_control = self.compute_trimmed_states_inputs(
                    Va, gamma, turn_radius, alpha, beta, phi)
            f = self.f(trimmed_state, trimmed_control)
            f[0] = 0.
            f[1] = 0.

            xdot = np.zeros((12,), dtype=np.double)
            xdot[2] = -Va * np.sin(gamma)
            xdot[8] = Va / turn_radius * np.cos(gamma)
            J = np.linalg.norm(xdot[2:] - f[2:])**2
            return J

        alpha_0 = -0.0
        beta_0 = 0.
        phi_0 = 0.

        alpha, beta, phi = gradient_descent(J, alpha_0, beta_0, phi_0, max_iters, epsilon, kappa)
        trimmed_state, trimmed_control = self.compute_trimmed_states_inputs(
                Va, gamma, R, alpha, beta, phi)

        return trimmed_state, trimmed_control

    def f(self, x, control_inputs):
        forces, moments = self.partial_forces_moments(x, control_inputs)

        dx = compute_kinematics(forces, moments, x, self.mav_p)
        return dx

    @property
    def control_inputs(self):
        return self._control_inputs

    @control_inputs.setter
    def control_inputs(self, inputs):
        self._control_inputs = inputs
        self.integrator.set_f_params(
                self.mav_p, self._control_inputs, self.partial_forces_moments)

    def calc_gamma_chi(self):
        Vg = self.R_vb @ self._state[3:6]
        gamma = asin(-Vg[2] / np.linalg.norm(Vg))  # h_dot = Vg*sin(gamma)

        Vg_h = Vg * np.cos(gamma)
        chi = acos(Vg_h[0] / np.linalg.norm(Vg_h))
        if Vg_h[1] < 0:
            chi *= -1

        return gamma, chi

    def _update_true_state(self):
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.true_state.pn = self._state[0]
        self.true_state.pe = self._state[1]
        self.true_state.h = -self._state[2]
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(self._state[3:6])
        self.true_state.gamma, self.true_state.chi = self.calc_gamma_chi()
        self.true_state.p = self._state[10]
        self.true_state.q = self._state[11]
        self.true_state.r = self._state[12]
        self.true_state.wn = self._wind[0]
        self.true_state.we = self._wind[1]


def compute_dynamic_pressure(rho, Va):
    p_dyn = 0.5 * rho * Va ** 2
    return p_dyn


def compute_nonlinear_CL_alpha(alpha, CL0, CL_alpha, sigmoid_alpha):
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    nonlinear_CL_alpha = (1. - sigmoid_alpha) * (CL0 + CL_alpha * alpha) + sigmoid_alpha * \
                         2. * np.sign(alpha) * sa * sa * ca
    return nonlinear_CL_alpha


def compute_aspect_ratio(b, S):
    aspect_ratio = b ** 2 / S
    return aspect_ratio
