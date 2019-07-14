import numpy as np
from hummingbird.parameters import aerosonde_parameters as mav_p
from hummingbird.message_types.msg_state import MsgState
from hummingbird.estimation.ekf_attitude import EkfAttitude
from hummingbird.estimation.ekf_position import EkfPosition
from hummingbird.maths.filters.alpha_filter import AlphaFilter
from hummingbird.parameters.constants import PhysicalConstants as pc


class Observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = MsgState()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = AlphaFilter(alpha=0.5)
        self.lpf_gyro_y = AlphaFilter(alpha=0.5)
        self.lpf_gyro_z = AlphaFilter(alpha=0.5)
        self.lpf_accel_x = AlphaFilter(alpha=0.5)
        self.lpf_accel_y = AlphaFilter(alpha=0.5)
        self.lpf_accel_z = AlphaFilter(alpha=0.5)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = AlphaFilter(alpha=0.9, y0=1350)
        self.lpf_diff = AlphaFilter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = EkfAttitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()

    def update(self, measurements):
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        gyro_x = self.lpf_gyro_x.update(measurements.gyro_x)
        gyro_y = self.lpf_gyro_y.update(measurements.gyro_y)
        gyro_z = self.lpf_gyro_z.update(measurements.gyro_z)

        self.estimated_state.p = gyro_x - self.estimated_state.bx
        self.estimated_state.q = gyro_y - self.estimated_state.by
        self.estimated_state.r = gyro_z - self.estimated_state.bz

        measurements.accel_x = self.lpf_accel_x.update(measurements.accel_x)
        measurements.accel_y = self.lpf_accel_y.update(measurements.accel_y)
        measurements.accel_z = self.lpf_accel_z.update(measurements.accel_z)

        # invert sensor model to get altitude and airspeed
        static_p = self.lpf_static.update(measurements.static_pressure)
        diff_p = self.lpf_diff.update(measurements.diff_pressure)
        if diff_p < 0:
            diff_p = 0
        self.estimated_state.h = static_p / (pc.rho0 * pc.g)
        self.estimated_state.Va = np.sqrt(2 * diff_p / pc.rho0)

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state
