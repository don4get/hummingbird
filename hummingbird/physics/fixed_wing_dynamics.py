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


class FixedWingDynamics(DynamicsBase):
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
        super(FixedWingDynamics, self).__init__(x0, t0, dt_integration)
        self.t0 = t0
        self.set_integrator(FixedWingDynamics.dynamics,
                            'dop853', jac=None, rtol=1e-8)
        self.partial_forces_moments = partial(
                FixedWingDynamics.forces_moments, params=self.mav_p)
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
        self._update_velocity_data(wind)
        if self.control_inputs is not None:
            t = self._dt_simulation + self.integrator.t
            self.integrate(t)
        else:
            raise Exception('set control inputs first')

        self._forces = self.forces_moments(self._state, self.control_inputs, self.mav_p)[0:3]
        self._update_true_state()

        # TODO: Add a param to select which integrator we want to use
        # # get forces and moments acting on rigid bod
        # # get forces and moments acting on rigid bod
        # forces_moments = self.forces_moments(self._state, delta, self.mav_p)
        # # update the airspeed, angle of attack, and side slip angles
        # self._update_velocity_data(wind)
        #
        # self.update_true_state_from_forces_moments(forces_moments)

    def _update_velocity_data(self, wind=np.zeros(6)):
        R_bv = Quaternion2Rotation(self._state[6:10]).T
        # compute airspeed
        V_wb = R_bv @ wind[:3] + wind[3:]
        V_ab = self._state[3:6] - V_wb
        self._Va = np.linalg.norm(V_ab)

        # compute angle of attack
        self._alpha = np.arctan2(V_ab[2], V_ab[0])

        # compute sideslip angle
        self._beta = asin(V_ab[1] / self._Va)

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
        alpha, beta = FixedWingDynamics.compute_alpha_beta(Va, x[SQE.u], x[SQE.v], x[SQE.w])
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
        self.R_vb = Quaternion2Rotation(self._state[6:10])
        self.R_bv = self.R_vb.T
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
