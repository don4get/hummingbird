import numpy as np
from hummingbird.parameters.control_parameters import ControlParameters
from hummingbird.parameters.aerosonde_parameters import MavParameters
from hummingbird.maths.pid import Pid
from hummingbird.message_types.msg_state import MsgState
from hummingbird.tools.transfer_function import TransferFunction
from hummingbird.parameters.constants import StateEnum as SE, ActuatorEnum as AE, PhysicalConstants as PC


class Autopilot:
    def __init__(self, ts_control, mav_p=MavParameters()):
        self.auto_p = ControlParameters()
        self.mav_p = mav_p
        # instantiate lateral controllers
        self.roll_to_aileron = Pid(
            kp=self.auto_p.roll_kp,
            kd=self.auto_p.roll_kd,
            limit=np.radians(self.auto_p.delta_a_max_deg))
        self.course_to_roll = Pid(
            kp=self.auto_p.course_kp,
            ki=self.auto_p.course_ki,
            dt=ts_control,
            limit=np.radians(self.auto_p.roll_max_deg))
        self.sideslip_to_rudder = Pid(
            kp=self.auto_p.sideslip_kp,
            ki=self.auto_p.sideslip_ki,
            dt=ts_control,
            limit=np.radians(self.auto_p.delta_r_max_deg))
        # TODO: yaw_damper tf should inherit from tf in python-control
        self.yaw_damper = TransferFunction(
            num=np.array([self.auto_p.yaw_damper_kp, 0]),
            den=np.array([1, 1 / self.auto_p.yaw_damper_tau_r]),
            Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_to_elevator = Pid(
            kp=self.auto_p.pitch_kp,
            kd=self.auto_p.pitch_kd,
            limit=np.radians(self.auto_p.delta_e_max_deg))
        self.altitude_to_pitch = Pid(
            kp=self.auto_p.altitude_kp,
            ki=self.auto_p.altitude_ki,
            dt=ts_control,
            limit=np.radians(self.auto_p.pitch_max_deg))
        self.airspeed_to_throttle = Pid(
            kp=self.auto_p.airspeed_throttle_kp,
            ki=self.auto_p.airspeed_throttle_ki,
            dt=ts_control,
            limit=self.auto_p.delta_t_max,
            lower_limit=self.auto_p.delta_t_min)
        # TODO: tune airspeed to pitch pid
        self.airspeed_to_pitch = Pid(
            kp=1.,
            ki=0.,
            dt=ts_control,
            limit=self.auto_p.delta_e_max_deg)
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        # lateral autopilot
        phi_c = self.course_to_roll(cmd.course_command, state.chi)
        delta_a = self.roll_to_aileron(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        theta_c = self.altitude_to_pitch(cmd.altitude_command, state.h)
        delta_e = self.pitch_to_elevator(theta_c, state.theta, state.q)
        delta_t = self.airspeed_to_throttle(cmd.airspeed_command, state.Va)

        # construct output and commanded states
        delta = np.array([delta_e, delta_a, delta_r, delta_t])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def set_gains_from_trim(self, trimmed_state, trimmed_control):
        Va_trim = np.linalg.norm(trimmed_state[3:6])

        self.set_course_pid_gains(Va_trim)
        self.set_roll_pid_gains(Va_trim)
        self.set_altitude_pid_gains(Va_trim)
        self.set_pitch_pid_gains(Va_trim)
        alpha_trim = np.arctan(trimmed_state[SE.w] / trimmed_state[SE.u])
        self.set_airspeed_pitch_pid_gains(Va_trim, alpha_trim, trimmed_control[AE.elevator])
        self.set_airspeed_thrust_pid_gains(Va_trim, alpha_trim, trimmed_control[
            AE.elevator], trimmed_control[AE.thrust])

    def set_course_pid_gains(self, Va):
        # Consider ground speed is equal to airspeed
        # TODO: Improve vg != va
        Vg = Va
        self.course_to_roll.kp = 2 * self.auto_p.course_ksi * self.auto_p.course_omega * Vg / PC.g
        self.course_to_roll.ki = self.auto_p.course_omega ** 2 * Vg / PC.g

    def set_roll_pid_gains(self, Va):
        if np.isclose(Va, 0):
            return

        params = self.mav_p
        S = params.S_wing
        b = params.b
        rho = params.rho
        Jx = params.Jx
        Jz = params.Jz
        Jxz = params.Jxz
        gamma_3 = self.mav_p.gamma3
        gamma_4 = self.mav_p.gamma4
        Cl_p = params.C_ell_p
        Cl_delta_a = params.C_ell_delta_a
        Cn_delta_a = params.C_n_delta_a
        Cn_p = params.C_n_p
        Cp_delta_a = gamma_3 * Cl_delta_a + gamma_4 * Cn_delta_a
        Cp_p = gamma_3 * Cl_p + gamma_4 * Cn_p
        aphi_1 = -0.5 * rho * Va * S * b * Cp_p * 0.5 * b
        aphi_2 = 0.5 * rho * Va ** 2 * S * b * Cp_delta_a

        self.roll_to_aileron.kp = self.auto_p.delta_a_max_deg / \
                                     self.auto_p.error_roll_max_deg * np.sign(aphi_2)
        self.roll_to_aileron.ki = self.auto_p.roll_ki
        omega_phi = np.sqrt(np.abs(aphi_2) * self.auto_p.delta_a_max_deg / self.auto_p.error_roll_max_deg)
        zeta = self.auto_p.roll_zeta
        self.roll_to_aileron.kd = (2 * zeta * omega_phi - aphi_1) / aphi_2

    def set_altitude_pid_gains(self, Va):
        if np.isclose(Va, 0):
            return

        S = self.mav_p.S_wing
        c = self.mav_p.c
        rho = self.mav_p.rho
        Jy = self.mav_p.Jy
        Cm_alpha = self.mav_p.C_m_q
        Cm_delta_e = self.mav_p.C_m_delta_e
        atheta_2 = -rho * Va**2 * c * S * Cm_alpha * 0.5 / Jy
        atheta_3 = rho * Va**2 * c * S * Cm_delta_e * 0.5 / Jy
        omega_h = self.auto_p.altitude_omega
        kp_theta = self.auto_p.delta_e_max_deg / self.auto_p.error_pitch_max_deg * np.sign(atheta_3)
        K_theta_dc = (kp_theta * atheta_3) / (atheta_2 + kp_theta * atheta_3)
        zeta = self.auto_p.altitude_pitch_zeta
        self.altitude_to_pitch.kp = 2.0 * zeta * omega_h / (K_theta_dc * Va)
        self.altitude_to_pitch.ki = omega_h ** 2 / (K_theta_dc * Va)

    def set_pitch_pid_gains(self, Va):
        if np.isclose(Va, 0):
            return

        S = self.mav_p.S_wing
        c = self.mav_p.c
        rho = self.mav_p.rho
        Jy = self.mav_p.Jy
        Cm_q = self.mav_p.C_m_q
        # TODO: Check if Cm_alpha works
        Cm_alpha = self.mav_p.C_m_q
        Cm_delta_e = self.mav_p.C_m_delta_e
        atheta_1 = -rho * Va * c * S * Cm_q * 0.5 * c
        atheta_2 = -rho * Va**2 * c * S * Cm_alpha * 0.5 / Jy
        atheta_3 = rho * Va**2 * c * S * Cm_delta_e * 0.5 / Jy
        self.pitch_to_elevator.kp = self.auto_p.delta_e_max_deg / self.auto_p.error_pitch_max_deg * np.sign(atheta_3)
        self.pitch_to_elevator.ki = self.auto_p.pitch_ki
        omega_theta = np.sqrt(
            atheta_2 + self.auto_p.delta_e_max_deg / self.auto_p.error_pitch_max_deg * np.abs(atheta_3))
        zeta = self.auto_p.pitch_zeta
        self.pitch_to_elevator.kd = (
            2 * zeta * omega_theta - atheta_1) / atheta_3

    def set_airspeed_pitch_pid_gains(self, Va, alpha, delta_e):
        if np.isclose(Va, 0):
            return

        S = self.mav_p.S_wing
        c = self.mav_p.c
        rho = self.mav_p.rho
        Jy = self.mav_p.Jy
        CD0 = self.mav_p.C_D_0
        CD_alpha = self.mav_p.C_D_alpha
        CD_delta_e = self.mav_p.C_D_delta_e
        C_prop = self.mav_p.C_prop
        S_prop = self.mav_p.S_prop
        mass = self.mav_p.mass
        aV_1 = rho * Va * S * (CD0 + CD_alpha * alpha + CD_delta_e * delta_e) / mass + rho * S_prop * C_prop * Va / mass
        Cm_alpha = self.mav_p.C_m_q
        Cm_delta_e = self.mav_p.C_m_delta_e
        atheta_2 = -rho * Va ** 2 * c * S * Cm_alpha * 0.5 / Jy
        atheta_3 = rho * Va ** 2 * c * S * Cm_delta_e * 0.5 / Jy
        omega_v2 = 0.5
        kp_theta = self.auto_p.delta_e_max_deg / self.auto_p.error_pitch_max_deg * np.sign(atheta_3)

        K_theta_dc = (kp_theta * atheta_3) / (atheta_2 + kp_theta * atheta_3)
        zeta = self.auto_p.airspeed_pitch_zeta
        self.airspeed_to_pitch.kp = (aV_1 - 2.0 * zeta * omega_v2) / (K_theta_dc * PC.g)
        self.airspeed_to_pitch.ki = - omega_v2 ** 2 / (K_theta_dc * PC.g)

    def set_airspeed_thrust_pid_gains(self, Va, alpha, delta_e, delta_t):
        S = self.mav_p.S_wing
        rho = self.mav_p.rho
        CD0 = self.mav_p.C_D_0
        CD_alpha = self.mav_p.C_D_alpha
        CD_delta_e = self.mav_p.C_D_delta_e
        C_prop = self.mav_p.C_prop
        S_prop = self.mav_p.S_prop
        mass = self.mav_p.mass
        k_motor = self.mav_p.K_V
        av_1 = rho * Va * S * (CD0 + CD_alpha * alpha + CD_delta_e * delta_e) / mass + rho * S_prop * C_prop * Va / mass
        av_2 = rho * S_prop * C_prop * k_motor ** 2 * delta_t / mass
        omega_v = 1.
        zeta = self.auto_p.airspeed_throttle_zeta
        kp_v = (2.0 * zeta * omega_v - av_1) / av_2
        ki_v = omega_v ** 2 / av_2
        self.airspeed_to_throttle.kp = kp_v
        self.airspeed_to_throttle.ki = ki_v
