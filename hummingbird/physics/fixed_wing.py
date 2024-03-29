from hummingbird.physics.fixed_wing_dynamics import FixedWingDynamics
# from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.physics.sensors.sensors import Sensors
from hummingbird.parameters.aerosonde_parameters import MavParameters
from hummingbird.parameters.sensor_parameters import SensorParameters
from hummingbird.parameters.simulation_parameters import SimulationParameters


class FixedWing:
    sim_p = SimulationParameters()

    def __init__(self,
                 mav_p=MavParameters(),
                 sensor_p=SensorParameters(),
                 obs_p=None,
                 pfollow_p=None,
                 pmanager_p=None,
                 pplanner_p=None,
                 dt_dynamics=sim_p.dt_simulation,
                 dt_simu=sim_p.dt_simulation,
                 dt_sensors=sim_p):
        # self.dynamics = MavDynamics(x0=mav_p.initial_state,
        #                                   t0=sim_p.start_time,
        #                                   dt_integration=sim_p.dt_simulation,
        #                                   mav_p=mav_p)
        self.dynamics = FixedWingDynamics()
        # self.dynamics = MavDynamics()
        self.sensors = Sensors(mav_p=mav_p,
                               sensor_p=sensor_p,
                               dt_simulation=self.sim_p.dt_simulation,
                               initial_state=self.dynamics.true_state,
                               initial_forces=self.dynamics._forces)

