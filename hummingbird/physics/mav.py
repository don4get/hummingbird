from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.physics.sensors.sensors import Sensors
from hummingbird.parameters import aerosonde_parameters as mav_p
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.parameters import sensor_parameters as sensor_p


class Mav:
    def __init__(self, mav_p=mav_p, sensor_p=sensor_p, dt_dynamics=sim_p
                 .dt_simulation, dt_sensors=sim_p
                 .dt_simulation):
        self.dynamics = MavDynamics()
        self.sensors = Sensors()
