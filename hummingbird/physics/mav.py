from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.physics.fixedwing_dynamics import FixedwingDynamics
from hummingbird.physics.sensors.sensors import Sensors
from hummingbird.parameters.aerosonde_parameters import MavParameters
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.parameters.sensor_parameters import SensorParameters


class Mav:
    def __init__(self, mav_p=MavParameters,
                 sensor_p=SensorParameters,
                 dt_dynamics=sim_p.dt_simulation,
                 dt_simu=sim_p.dt_simulation,
                 dt_sensors=sim_p):
        self.dynamics = MavDynamics()
        self.sensors = Sensors()


class Fixedwing:
    def __init__(self, mav_p=MavParameters,
                 sensor_p=SensorParameters,
                 dt_dynamics=sim_p.dt_simulation,
                 dt_simu=sim_p.dt_simulation,
                 dt_sensors=sim_p):

        self.dynamics = FixedwingDynamics()
        self.sensors = Sensors()
    def __init__(self, x0, t0, dynamics_config_file):

        self.params = FixedwingParameters(dynamics_config_file)

        self.dynamics = FixedwingDynamics(x0, t0, 0.001, self.params)

    def update_state(self, dt):
        if self.dynamics.control_inputs is not None:
            t = dt + self.dynamics.integrator.t
            self.dynamics.integrate(t)
        else:
            raise Exception('set control inputs first')

    def set_state(self, x, t):
        self.dynamics.integrator.set_initial_value(x, t)

    def set_control_inputs(self, control_inputs):
        self.dynamics.control_inputs = control_inputs

    def get_control_inputs(self):
        return self.dynamics.control_inputs
