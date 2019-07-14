from functools import partial

from hummingbird.parameters.constants import ActuatorEnum, PhysicalConstants as pc, StateEnum
from hummingbird.physics.dynamics_base import *
from hummingbird.maths.generics import compute_sigmoid
from hummingbird.physics.kinematics import compute_kinematics
from hummingbird.maths.gradient_descent import gradient_descent
import numpy as np


class FixedwingDynamics(DynamicsBase):
    """ Fixed Wing UAV Dynamics

        :param x0: Initial state vector
        :type x0: vector [12x1]
        :param t0: Initial time of the simulation
        :type t0: double [s]
        :param dt_integration: Sampling time of the integrator
        :type dt_integration: double [s]
    """

    def __init__(self, x0, t0, dt_integration, params):
        self.params = params
        super(FixedwingDynamics, self).__init__(x0, t0, dt_integration)
        self.t0 = t0
        self.set_integrator(FixedwingDynamics.dynamics,
                            'dop853', jac=None, rtol=1e-8)
        self.partial_forces_and_moments = partial(
                FixedwingDynamics.forces_and_moments, params=self.params)
        self._control_inputs = [0., 0., 0., 0.]

    @staticmethod
    def forces_and_moments(y, control_inputs, params):
        """ Compute the second law of newton, given the state and the
        control inputs

        :param y: State vector
        :type y: vector [12x1]
        :param control_inputs: actuators
        :type control_inputs: vector [4x1]
        :param params: [description]
        :type params: [type]
        :returns: [description]
        :rtype: {[type]}
        """
        g = pc.g
        mass = params.mass # ['params']['mass']
        S = params.S
        b = params.b
        c = params.c
        rho = params.rho
        e = params.e
        kT_p = params.kT_p
        kOmega = params.kOmega

        u = y[StateEnum.u]
        v = y[StateEnum.v]
        w = y[StateEnum.w]
        phi = y[StateEnum.phi]
        theta = y[StateEnum.theta]
        p = y[StateEnum.p]
        q = y[StateEnum.q]
        r = y[StateEnum.r]

        Va = np.sqrt(u**2 + v**2 + w**2)

        alpha, beta = FixedwingDynamics.compute_alpha_beta(Va, u, v, w)

        delta_e = control_inputs[ActuatorEnum.elevator]
        delta_a = control_inputs[ActuatorEnum.aileron]
        delta_r = control_inputs[ActuatorEnum.rudder]
        delta_t = control_inputs[ActuatorEnum.thrust]

        # Dynamic pressure
        p_dyn = compute_dynamic_pressure(rho, Va)

        def longitudinal_aerodynamic_forces_moments():
            CL0 = params.CL0
            CL_alpha = params.CL_alpha
            CL_q = params.CL_q
            CL_delta_e = params.CL_delta_e
            M = params.M
            alpha_0 = params.alpha_0

            sigmoid_alpha = compute_sigmoid(alpha_0, alpha, M)
            nonlinear_CL_alpha = compute_nonlinear_CL_alpha(alpha, CL0, CL_alpha, sigmoid_alpha)

            if np.isclose(Va, 0., 1e-5):
                lift = 0.
            else:
                lift = p_dyn * S * (nonlinear_CL_alpha  + CL_q * c * q * 0.5 / Va + CL_delta_e * delta_e)

            CD0 = params.CD0
            CD_alpha = params.CD_alpha
            CD_q = params.CD_q
            CD_delta_e = params.CD_delta_e
            CD_p = params.CD_p
            aspect_ratio = compute_aspect_ratio(b, S)
            # TODO: Figure out why CD_p is used here
            # CD_alpha = CD_p + (CL0 + CL_alpha * alpha)**2 / (np.pi * e * AR)

            if np.isclose(Va, 0., 1e-5) :
                drag = 0.
            else:
                drag = p_dyn * S * (CD0 + CD_alpha * alpha +
                                    CD_q * c * q * 0.5 / Va +
                                    CD_delta_e * delta_e)

            Cm0 = params.Cm0
            Cm_alpha = params.Cm_alpha
            Cm_q = params.Cm_q
            Cm_delta_e = params.Cm_delta_e
            Cm_alpha = Cm0 + Cm_alpha * alpha
            m = 0.5 * rho * S * c * (Cm_alpha * Va**2 +
                                     Cm_q * c * q * 0.5 * Va +
                                     Cm_delta_e * delta_e * Va**2)

            fx = -drag * np.cos(alpha) + lift * np.sin(alpha)
            fz = -drag * np.sin(alpha) - lift * np.cos(alpha)
            return fx, fz, m

        def lateral_forces_moments():
            const = 0.5 * rho * S
            CY0 = params.CY0
            CY_beta = params.CY_beta
            CY_p = params.CY_p
            CY_r = params.CY_r
            CY_delta_a = params.CY_delta_a
            CY_delta_r = params.CY_delta_r
            # TODO: factorize some operations in the following expressions (fy, l, n)
            fy = const * (CY0 * Va**2 +
                          CY_beta * beta * Va**2 +
                          CY_p * b * p * 0.5 * Va +
                          CY_r * r * b * 0.5 * Va +
                          CY_delta_a * delta_a * Va**2 +
                          CY_delta_r * delta_r * Va**2)

            Cl0 = params.Cl0
            Cl_beta = params.Cl_beta
            Cl_p = params.Cl_p
            Cl_r = params.Cl_r
            Cl_delta_a = params.Cl_delta_a
            Cl_delta_r = params.Cl_delta_r
            l = b * const * (Cl0 * Va**2 +
                             Cl_beta * beta * Va**2 +
                             Cl_p * b * p * 0.5 * Va +
                             Cl_r * r * b * 0.5 * Va +
                             Cl_delta_a * delta_a * Va**2 +
                             Cl_delta_r * delta_r * Va**2)

            Cn0 = params.Cn0
            Cn_beta = params.Cn_beta
            Cn_p = params.Cn_p
            Cn_r = params.Cn_r
            Cn_delta_a = params.Cn_delta_a
            Cn_delta_r = params.Cn_delta_r
            n = b * const * (Cn0 * Va**2 +
                             Cn_beta * beta * Va**2 +
                             Cn_p * b * p * 0.5 * Va +
                             Cn_r * r * b * 0.5 * Va +
                             Cn_delta_a * delta_a * Va**2 +
                             Cn_delta_r * delta_r * Va**2)
            return fy, l, n


        sphi = np.sin(phi)
        cphi = np.cos(phi)

        stheta = np.sin(theta)
        ctheta = np.cos(theta)

        f_aero_x, f_aero_z, m_aero = longitudinal_aerodynamic_forces_moments()
        f_aero_y, l_aero, n_aero = lateral_forces_moments()
        g_x, g_y, g_z = compute_gravitational_force(cphi, sphi, ctheta, stheta, mass, g)
        f_prop_x, f_prop_y, f_prop_z = propeller_forces(params, Va, delta_t)
        l_prop, m_prop, n_prop = propeller_torques(params, delta_t)
        fx = f_aero_x + g_x + f_prop_x
        fy = f_aero_y + g_y + f_prop_y
        fz = f_aero_z + g_z + f_prop_z
        l = l_aero + l_prop
        m = m_aero + m_prop
        n = n_aero + n_prop
        return [fx, fy, fz], [l, m, n]

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
    def dynamics(t, x, params, control_inputs, forces_and_moments):
        forces, moments = forces_and_moments(x, control_inputs)

        dx = compute_kinematics(forces, moments, x, params)
        return dx

    def compute_trimmed_states_inputs(self, Va, gamma, turn_radius, alpha, beta, phi):
        params = self.params

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
        forces, moments = self.partial_forces_and_moments(x, control_inputs)

        dx = compute_kinematics(forces, moments, x, self.params)
        return dx

    @property
    def control_inputs(self):
        return self._control_inputs

    @control_inputs.setter
    def control_inputs(self, inputs):
        self._control_inputs = inputs
        self.integrator.set_f_params(
                self.params, self._control_inputs, self.partial_forces_and_moments)


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
