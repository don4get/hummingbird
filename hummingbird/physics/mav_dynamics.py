import numpy as np
from hummingbird.message_types.msg_state import MsgState
from hummingbird.parameters.aerosonde_parameters import MavParameters
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.physics.generics import propeller_thrust_torque
from hummingbird.tools.rotations import Quaternion2Rotation, Quaternion2Euler
from math import asin, exp, acos
from hummingbird.parameters.constants import StateQuatEnum

np.set_printoptions(suppress=True, precision=4)


class MavDynamics:
    def __init__(self, mav_p=MavParameters(), dt=sim_p.dt_simulation):
        self.mav_p = mav_p
        self._dt_simulation = dt
        self._state = np.zeros(StateQuatEnum.size)
        self.reset_state()

        self.R_vb = Quaternion2Rotation(self._state[6:10])  # Rotation body->vehicle
        self.R_bv = np.copy(self.R_vb).T  # vehicle->body

        self._forces = np.zeros(3)
        self._wind = np.zeros(3)  # wind in NED frame in meters/sec
        self._update_velocity_data()

        self._Va = self.mav_p.u0
        self._alpha = 0
        self._beta = 0
        self.true_state = MsgState()
        self._update_true_state()

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

    ###################################
    # public functions
    def update(self, delta, wind=np.zeros(6)):
        """
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        """

        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        # update the airspeed, angle of attack, and side slip angles
        self._update_velocity_data(wind)

        self.update_true_state_from_forces_moments(forces_moments)

    def update_true_state_from_forces_moments(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._dt_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2. * k1, forces_moments)
        k3 = self._derivatives(self._state + time_step / 2. * k2, forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        # normalize the quaternion
        e0 = self._state[6]
        e1 = self._state[7]
        e2 = self._state[8]
        e3 = self._state[9]
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6] = self._state[6] / normE
        self._state[7] = self._state[7] / normE
        self._state[8] = self._state[8] / normE
        self._state[9] = self._state[9] / normE
        self.R_vb = Quaternion2Rotation(self._state[6:10])  # body->vehicle
        self.R_bv = np.copy(self.R_vb).T  # vehicle->body
        self._update_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
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

        self.R_vb = Quaternion2Rotation(np.array([e0, e1, e2, e3]))  # body->vehicle

        # position kinematics
        pn_dot, pe_dot, pd_dot = self.R_vb @ np.array([u, v, w])

        # position dynamics
        vec_pos = np.array([r * v - q * w, p * w - r * u, q * u - p * v])
        u_dot, v_dot, w_dot = vec_pos + 1 / self.mav_p.mass * np.array([fx, fy, fz])

        # rotational kinematics
        mat_rot = np.array([[0, -p, -q, -r],
                            [p, 0, r, -q],
                            [q, -r, 0, p],
                            [r, q, -p, 0]])
        e0_dot, e1_dot, e2_dot, e3_dot = 0.5 * mat_rot @ np.array([e0, e1, e2, e3])

        # rotational dynamics
        g1 = self.mav_p.gamma1
        g2 = self.mav_p.gamma2
        g3 = self.mav_p.gamma3
        g4 = self.mav_p.gamma4
        g5 = self.mav_p.gamma5
        g6 = self.mav_p.gamma6
        g7 = self.mav_p.gamma7
        g8 = self.mav_p.gamma8

        vec_rot = np.array([g1 * p * q - g2 * q * r, g5 * p * r - g6 * (p ** 2 - r ** 2), g7 * p * q - g1 * q * r])
        vec_rot2 = np.array([g3 * l + g4 * n, m / self.mav_p.Jy, g4 * l + g8 * n])

        p_dot, q_dot, r_dot = vec_rot + vec_rot2

        # collect the derivative of the states
        x_dot = np.array([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                          e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot])

        return x_dot

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

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        de = delta[0]
        da = delta[1]
        dr = delta[2]
        dt = delta[3]

        # gravity
        self.R_bv = Quaternion2Rotation(self._state[6:10]).T
        fg = self.R_bv @ np.array([0, 0, self.mav_p.mass * self.mav_p.gravity])

        fp, Mp = propeller_thrust_torque(dt, self._Va, self.mav_p)

        # Aerodynamic forces/moments

        # Longitudinal
        M = self.mav_p.M
        alpha = self._alpha
        alpha0 = self.mav_p.alpha0
        rho = self.mav_p.rho
        Va = self._Va
        S = self.mav_p.S_wing
        q = self._state[11]
        c = self.mav_p.c

        sigma_alpha = (1 + exp(-M * (alpha - alpha0)) + exp(M * (alpha + alpha0))) / \
                      ((1 + exp(-M * (alpha - alpha0))) * (1 + exp(M * (alpha + alpha0))))
        CL_alpha = (1 - sigma_alpha) * (self.mav_p.C_L_0 + self.mav_p.C_L_alpha * alpha) + \
                    sigma_alpha * (2 * np.sign(alpha) * (np.sin(alpha) ** 2) * np.cos(alpha))
        F_lift = 0.5 * rho * (Va ** 2) * S * (CL_alpha + self.mav_p.C_L_q * (c / (2 * Va)) * q + self.mav_p.C_L_delta_e * de)
        CD_alpha = self.mav_p.C_D_p + ((self.mav_p.C_L_0 + self.mav_p.C_L_alpha * alpha) ** 2) / \
                   (np.pi * self.mav_p.e * self.mav_p.AR)
        F_drag = 0.5 * rho * (Va ** 2) * S * (CD_alpha + self.mav_p.C_D_q * (c / (2 * Va)) * q + self.mav_p.C_D_delta_e * de)
        m = 0.5 * rho * (Va ** 2) * S * c * (self.mav_p.C_m_0 + self.mav_p.C_m_alpha * alpha +
                                             self.mav_p.C_m_q * (c / (2. * Va)) * q + self.mav_p.C_m_delta_e * de)

        # Lateral
        b = self.mav_p.b
        Va = self._Va
        beta = self.true_state.beta
        p = self.true_state.p
        r = self.true_state.r
        rho = self.mav_p.rho
        S = self.mav_p.S_wing

        # Calculating fy
        fa_y = 1 / 2.0 * rho * (Va ** 2) * S * (self.mav_p.C_Y_0 + self.mav_p.C_Y_beta * beta +
                                                self.mav_p.C_Y_p * (b / (2 * Va)) * p + self.mav_p.C_Y_r *
                                                (b / (2 * Va)) * r + self.mav_p.C_Y_delta_a * da +
                                                self.mav_p.C_Y_delta_r * dr)

        # Calculating l
        l = 1 / 2.0 * rho * (Va ** 2) * S * b * (self.mav_p.C_ell_0 + self.mav_p.C_ell_beta * beta +
                                                 self.mav_p.C_ell_p * (b / (2 * Va)) * p + self.mav_p.C_ell_r * (b / (2 * Va)) *
                                                 r + self.mav_p.C_ell_delta_a * da + self.mav_p.C_ell_delta_r * dr)

        # Calculating n
        n = 1 / 2.0 * rho * (Va ** 2) * S * b * (self.mav_p.C_n_0 + self.mav_p.C_n_beta * beta +
                                                 self.mav_p.C_n_p * (b / (2 * Va)) * p + self.mav_p.C_n_r * (b / (2 * Va)) * r +
                                                 self.mav_p.C_n_delta_a * da + self.mav_p.C_n_delta_r * dr)

        # Combining into force/moment arrays
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        [fa_x, fa_z] = np.array([[ca, -sa], [sa, ca]]) @ np.array([-F_drag, -F_lift])
        fa = np.array([fa_x, fa_y, fa_z])
        Ma = np.array([l, m, n])

        # Summing forces and moments
        [fx, fy, fz] = fg + fa + fp
        [Mx, My, Mz] = Ma + Mp
        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        return np.array([fx, fy, fz, Mx, My, Mz])

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
