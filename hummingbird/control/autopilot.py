"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np

sys.path.append('..')
from hummingbird.parameters import control_parameters as AP
from hummingbird.control.pid_control import PiControl, PdControlWithRate
from hummingbird.message_types.msg_state import MsgState
from hummingbird.tools.transfer_function import TransferFunction


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_to_aileron = PdControlWithRate(
            kp=AP.roll_kp,
            kd=AP.roll_kd,
            limit=np.radians(45))
        self.course_to_roll = PiControl(
            kp=AP.course_kp,
            ki=AP.course_ki,
            Ts=ts_control,
            limit=np.radians(30))
        self.sideslip_to_rudder = PiControl(
            kp=AP.sideslip_kp,
            ki=AP.sideslip_ki,
            Ts=ts_control,
            limit=np.radians(45))
        self.yaw_damper = TransferFunction(
            num=np.array([AP.yaw_damper_kp, 0]),
            den=np.array([1, 1 / AP.yaw_damper_tau_r]),
            Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_to_elevator = PdControlWithRate(
            kp=AP.pitch_kp,
            kd=AP.pitch_kd,
            limit=np.radians(45))
        self.altitude_to_pitch = PiControl(
            kp=AP.altitude_kp,
            ki=AP.altitude_ki,
            Ts=ts_control,
            limit=np.radians(30))
        self.airspeed_to_throttle = PiControl(
            kp=AP.airspeed_throttle_kp,
            ki=AP.airspeed_throttle_ki,
            Ts=ts_control,
            limit=1.0,
            lower_lim=0)
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        # lateral autopilot
        phi_c = self.course_to_roll.update(cmd.course_command, state.chi,
                                           reset_flag=True)
        delta_a = self.roll_to_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        theta_c = self.altitude_to_pitch.update(cmd.altitude_command, state.h)
        delta_e = self.pitch_to_elevator.update(theta_c, state.theta, state.q)
        delta_t = self.airspeed_to_throttle.update(cmd.airspeed_command, state.Va)

        # construct output and commanded states
        delta = np.array([delta_e, delta_a, delta_r, delta_t])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state
