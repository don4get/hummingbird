import numpy as np
from hummingbird.message_types.msg_state import MsgState
from hummingbird.message_types.msg_sensors import MsgSensors
from hummingbird.parameters import aerosonde_parameters as mav_p
import hummingbird.parameters.sensor_parameters as sensor_p
from hummingbird.physics.generics import propeller_thrust_torque
from hummingbird.tools.rotations import Quaternion2Rotation, Quaternion2Euler
from math import asin, exp, acos

np.set_printoptions(suppress=True, precision=4)


class MavDynamics:
    def __init__(self, ts):
        self._ts_simulation = ts
        self.reset_state()

        self.R_vb = Quaternion2Rotation(self._state[6:10])  # Rotation body->vehicle
        self.R_bv = np.copy(self.R_vb).T  # vehicle->body

        self._forces = np.zeros(3)
        self._wind = np.zeros(3)  # wind in NED frame in meters/sec
        self._update_velocity_data()

        self._Va = mav_p.u0
        self._alpha = 0
        self._beta = 0
        self.true_state = MsgState()
        self.sensors = MsgSensors()
        self._update_true_state()

        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

        self.update_sensors()

    def reset_state(self):
        self._state = np.array([mav_p.pn0,  # (0)
                                mav_p.pe0,  # (1)
                                mav_p.pd0,  # (2)
                                mav_p.u0,  # (3)
                                mav_p.v0,  # (4)
                                mav_p.w0,  # (5)
                                mav_p.e0,  # (6)
                                mav_p.e1,  # (7)
                                mav_p.e2,  # (8)
                                mav_p.e3,  # (9)
                                mav_p.p0,  # (10)
                                mav_p.q0,  # (11)
                                mav_p.r0])  # (12)

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
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step / 2. * k1, forces_moments)
        k3 = self._derivatives(self._state + time_step / 2. * k2, forces_moments)
        k4 = self._derivatives(self._state + time_step * k3, forces_moments)
        self._state += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
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

    def update_sensors(self):
        """
            Return value of sensors on MAV: gyros, accels, static_pressure, 
            dynamic_pressure, GPS
        """
        theta = self.true_state.theta
        phi = self.true_state.phi
        g = mav_p.gravity
        m = mav_p.mass
        rho = mav_p.rho

        gyro_eta = np.random.randn(3) * sensor_p.gyro_sigma
        accl_eta = np.random.randn(3) * sensor_p.accel_sigma
        static_pres_eta = np.random.randn() * sensor_p.static_pres_sigma
        diff_pres_eta = np.random.randn() * sensor_p.diff_pres_sigma

        self.sensors.gyro_x = self.true_state.p + sensor_p.gyro_x_bias + gyro_eta[0]
        self.sensors.gyro_y = self.true_state.q + sensor_p.gyro_y_bias + gyro_eta[1]
        self.sensors.gyro_z = self.true_state.r + sensor_p.gyro_z_bias + gyro_eta[2]
        self.sensors.accel_x = self._forces[0] / m + g * np.sin(theta) + accl_eta[0]
        self.sensors.accel_y = self._forces[1] / m - g * np.cos(theta) * np.sin(phi) + accl_eta[1]
        self.sensors.accel_z = self._forces[2] / m - g * np.cos(theta) * np.cos(phi) + accl_eta[2]
        self.sensors.static_pressure = rho * g * self.true_state.h + static_pres_eta
        self.sensors.diff_pressure = (rho * self.true_state.Va ** 2) / 2 + diff_pres_eta

        if self._t_gps >= sensor_p.ts_gps:
            gps_error = np.exp(-sensor_p.gps_beta * sensor_p.ts_gps)
            gps_eta = np.random.randn(3) * sensor_p.gps_neh_sigmas  # n, e, h sigmas
            gps_eta_Vg = np.random.randn() * sensor_p.gps_Vg_sigma
            gps_eta_course = np.random.randn() * sensor_p.gps_course_sigma

            self._gps_eta_n = gps_error * self._gps_eta_n + gps_eta[0]
            self._gps_eta_e = gps_error * self._gps_eta_e + gps_eta[1]
            self._gps_eta_h = gps_error * self._gps_eta_h + gps_eta[2]
            self.sensors.gps_n = self.true_state.pn + self._gps_eta_n
            self.sensors.gps_e = self.true_state.pe + self._gps_eta_e
            self.sensors.gps_h = self.true_state.h + self._gps_eta_h
            self.sensors.gps_Vg = self.true_state.Vg + gps_eta_Vg
            self.sensors.gps_course = self.true_state.chi + gps_eta_course
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation

        return self.sensors

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
        u_dot, v_dot, w_dot = vec_pos + 1 / mav_p.mass * np.array([fx, fy, fz])

        # rotational kinematics
        mat_rot = np.array([[0, -p, -q, -r],
                            [p, 0, r, -q],
                            [q, -r, 0, p],
                            [r, q, -p, 0]])
        e0_dot, e1_dot, e2_dot, e3_dot = 0.5 * mat_rot @ np.array([e0, e1, e2, e3])

        # rotational dynamics
        g1 = mav_p.gamma1
        g2 = mav_p.gamma2
        g3 = mav_p.gamma3
        g4 = mav_p.gamma4
        g5 = mav_p.gamma5
        g6 = mav_p.gamma6
        g7 = mav_p.gamma7
        g8 = mav_p.gamma8

        vec_rot = np.array([g1 * p * q - g2 * q * r, g5 * p * r - g6 * (p ** 2 - r ** 2), g7 * p * q - g1 * q * r])
        vec_rot2 = np.array([g3 * l + g4 * n, m / mav_p.Jy, g4 * l + g8 * n])

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
        fg = self.R_bv @ np.array([0, 0, mav_p.mass * mav_p.gravity])

        thrust, torque = propeller_thrust_torque(dt, self._Va, mav_p)
        fp = np.array([thrust, 0, 0])
        Mp = np.array([torque, 0, 0])

        # Aerodynamic forces/moments

        # Longitudinal
        M = mav_p.M
        alpha = self._alpha
        alpha0 = mav_p.alpha0
        rho = mav_p.rho
        Va = self._Va
        S = mav_p.S_wing
        q = self._state[11]
        c = mav_p.c

        sigma_alpha = (1 + exp(-M * (alpha - alpha0)) + exp(M * (alpha + alpha0))) / \
                      ((1 + exp(-M * (alpha - alpha0))) * (1 + exp(M * (alpha + alpha0))))
        CL_alpha = (1 - sigma_alpha) * (mav_p.C_L_0 + mav_p.C_L_alpha * alpha) + \
                    sigma_alpha * (2 * np.sign(alpha) * (np.sin(alpha) ** 2) * np.cos(alpha))
        F_lift = 0.5 * rho * (Va ** 2) * S * (CL_alpha + mav_p.C_L_q * (c / (2 * Va)) * q + mav_p.C_L_delta_e * de)
        CD_alpha = mav_p.C_D_p + ((mav_p.C_L_0 + mav_p.C_L_alpha * alpha) ** 2) / \
                   (np.pi * mav_p.e * mav_p.AR)
        F_drag = 0.5 * rho * (Va ** 2) * S * (CD_alpha + mav_p.C_D_q * (c / (2 * Va)) * q + mav_p.C_D_delta_e * de)
        m = 0.5 * rho * (Va ** 2) * S * c * (mav_p.C_m_0 + mav_p.C_m_alpha * alpha +
                                             mav_p.C_m_q * (c / (2. * Va)) * q + mav_p.C_m_delta_e * de)

        # Lateral
        b = mav_p.b
        Va = self._Va
        beta = self.true_state.beta
        p = self.true_state.p
        r = self.true_state.r
        rho = mav_p.rho
        S = mav_p.S_wing

        # Calculating fy
        fa_y = 1 / 2.0 * rho * (Va ** 2) * S * (mav_p.C_Y_0 + mav_p.C_Y_beta * beta +
                                                mav_p.C_Y_p * (b / (2 * Va)) * p + mav_p.C_Y_r *
                                                (b / (2 * Va)) * r + mav_p.C_Y_delta_a * da +
                                                mav_p.C_Y_delta_r * dr)

        # Calculating l
        l = 1 / 2.0 * rho * (Va ** 2) * S * b * (mav_p.C_ell_0 + mav_p.C_ell_beta * beta +
                                                 mav_p.C_ell_p * (b / (2 * Va)) * p + mav_p.C_ell_r * (b / (2 * Va)) *
                                                 r + mav_p.C_ell_delta_a * da + mav_p.C_ell_delta_r * dr)

        # Calculating n
        n = 1 / 2.0 * rho * (Va ** 2) * S * b * (mav_p.C_n_0 + mav_p.C_n_beta * beta +
                                                 mav_p.C_n_p * (b / (2 * Va)) * p + mav_p.C_n_r * (b / (2 * Va)) * r +
                                                 mav_p.C_n_delta_a * da + mav_p.C_n_delta_r * dr)

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
