import sys
import numpy as np
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter

from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.fixed_wing import FixedWing
from hummingbird.control.autopilot import Autopilot
from hummingbird.tools.signals import Signals
from hummingbird.message_types.msg_autopilot import MsgAutopilot
from hummingbird.estimation.observer import Observer


class ObserverSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="sensors.avi",
                                     bounding_box=(0, 0, 1000, 1000),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.mav_view = MavViewer()
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWing()
        self.wind = WindSimulation()
        self.ctrl = Autopilot(sim_p.dt_controller)
        self.obsv = Observer(sim_p.dt_controller)
        self.measurements = self.mav.sensors.sensors

        # autopilot commands
        self.commands = MsgAutopilot()
        self.Va_command = Signals(dc_offset=25.0,
                                  amplitude=3.0,
                                  start_time=2.0,
                                  frequency=0.01)
        self.h_command = Signals(dc_offset=100.0,
                                 amplitude=10.0,
                                 start_time=0.0,
                                 frequency=0.02)
        self.chi_command = Signals(dc_offset=np.radians(0),
                                   amplitude=np.radians(45),
                                   start_time=5.0,
                                   frequency=0.02)

    def simulate(self):
        while self.sim_time < sim_p.end_time:

            # -------autopilot commands-------------
            self.commands.airspeed_command = self.Va_command.square(self.sim_time)
            self.commands.course_command = self.chi_command.square(self.sim_time)
            self.commands.altitude_command = self.h_command.square(self.sim_time)

            # -------controller-------------
            estimated_state = self.obsv.update(self.measurements)  # estimate states from measurements
            delta, commanded_state = self.ctrl.update(self.commands, estimated_state)

            # -------physical system-------------
            current_wind = self.wind.update()  # get the new wind vector
            self.mav.dynamics.update(delta, current_wind)  # propagate the MAV dynamics
            self.measurements = self.mav.sensors.update_sensors(self.mav.dynamics.true_state,
                                                                self.mav.dynamics._forces)  # update the sensors

            # -------update viewer-------------
            self.mav_view.update(self.mav.dynamics.true_state)  # plot body of MAV
            if self.display_data:
                self.data_view.update(self.mav.dynamics.true_state,  # true states
                                      estimated_state,  # estimated states
                                      commanded_state,  # commanded states
                                      sim_p.dt_simulation)

            # -------increment time-------------
            self.sim_time += sim_p.dt_simulation

        sys.exit(self.mav_view.app.exec_())

