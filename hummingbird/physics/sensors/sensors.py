import numpy as np
from hummingbird.message_types.msg_sensors import MsgSensors
from hummingbird.parameters.sensor_parameters import SensorParameters
from hummingbird.parameters.aerosonde_parameters import MavParameters
from hummingbird.parameters.simulation_parameters import SimulationParameters
from hummingbird.message_types.msg_state import MsgState


class Sensors:
    sim_p = SimulationParameters()

    def __init__(self,
                 mav_p=MavParameters,
                 sensor_p=SensorParameters,
                 dt_simulation=sim_p.dt_simulation,
                 initial_state=MsgState(),
                 initial_forces=np.zeros(3)):
        self.dt_simulation = dt_simulation
        self.mav_p = mav_p
        self.sensor_p = sensor_p
        self.sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

        self.update_sensors(initial_state, initial_forces)

    def update_sensors(self, true_state, forces):
        """
            Return value of sensors on MAV: gyros, accels, static_pressure,
            dynamic_pressure, GPS
        """
        theta = true_state.theta
        phi = true_state.phi
        g = self.mav_p.gravity
        m = self.mav_p.mass
        rho = self.mav_p.rho
        sensor_p = self.sensor_p

        gyro_eta = np.random.randn(3) * sensor_p.gyro_sigma
        accl_eta = np.random.randn(3) * sensor_p.accel_sigma
        static_pres_eta = np.random.randn() * sensor_p.static_pres_sigma
        diff_pres_eta = np.random.randn() * sensor_p.diff_pres_sigma

        self.sensors.gyro_x = true_state.p + sensor_p.gyro_x_bias + gyro_eta[0]
        self.sensors.gyro_y = true_state.q + sensor_p.gyro_y_bias + gyro_eta[1]
        self.sensors.gyro_z = true_state.r + sensor_p.gyro_z_bias + gyro_eta[2]
        self.sensors.accel_x = forces[0] / m + g * np.sin(theta) + accl_eta[0]
        self.sensors.accel_y = forces[1] / m - g * np.cos(theta) * np.sin(phi) + accl_eta[1]
        self.sensors.accel_z = forces[2] / m - g * np.cos(theta) * np.cos(phi) + accl_eta[2]
        self.sensors.static_pressure = rho * g * true_state.h + static_pres_eta
        self.sensors.diff_pressure = (rho * true_state.Va ** 2) / 2 + diff_pres_eta

        if self._t_gps >= sensor_p.ts_gps:
            gps_error = np.exp(-sensor_p.gps_beta * sensor_p.ts_gps)
            gps_eta = np.random.randn(3) * sensor_p.gps_neh_sigmas  # n, e, h sigmas
            gps_eta_Vg = np.random.randn() * sensor_p.gps_Vg_sigma
            gps_eta_course = np.random.randn() * sensor_p.gps_course_sigma

            self._gps_eta_n = gps_error * self._gps_eta_n + gps_eta[0]
            self._gps_eta_e = gps_error * self._gps_eta_e + gps_eta[1]
            self._gps_eta_h = gps_error * self._gps_eta_h + gps_eta[2]
            self.sensors.gps_n = true_state.pn + self._gps_eta_n
            self.sensors.gps_e = true_state.pe + self._gps_eta_e
            self.sensors.gps_h = true_state.h + self._gps_eta_h
            self.sensors.gps_Vg = true_state.Vg + gps_eta_Vg
            self.sensors.gps_course = true_state.chi + gps_eta_course
            self._t_gps = 0.
        else:
            self._t_gps += self.dt_simulation

        return self.sensors
