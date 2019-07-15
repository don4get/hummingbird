import sys
import numpy as np
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.fixed_wing_dynamics import FixedWingDynamics
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.control.autopilot import Autopilot
from hummingbird.tools.signals import Signals
from hummingbird.message_types.msg_autopilot import MsgAutopilot


class AutopilotSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="autopilot.avi",
                                     bounding_box=(0, 0, 800, 600),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.mav_view = MavViewer()
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWingDynamics()
        self.wind = WindSimulation()
        self.ctrl = Autopilot(self.sim_p.dt_controller)

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
                                   frequency=0.2)

    def simulate(self):
        while self.sim_time < self.sim_p.end_time:

            # -------controller-------------
            estimated_state = self.mav.true_state  # uses true states in the control
            self.commands.airspeed_command = self.Va_command.square(self.sim_time)
            self.commands.course_command = self.chi_command.sinusoid(self.sim_time)
            self.commands.altitude_command = self.h_command.square(self.sim_time)
            delta, commanded_state = self.ctrl.update(self.commands, estimated_state)

            # -------physical system-------------
            current_wind = self.wind.update()  # get the new wind vector
            self.mav.update(delta, current_wind)  # propagate the MAV dynamics

            # -------update viewer-------------
            self.mav_view.update(self.mav.true_state)  # plot body of MAV
            if self.display_data:
                self.data_view.update(self.mav.true_state,  # true states
                                      self.mav.true_state,  # estimated states
                                      commanded_state,  # commanded states
                                      self.sim_p.dt_simulation)

            if self.record_video:
                self.video.update(self.sim_time)

            # -------increment time-------------
            self.sim_time += self.sim_p.dt_simulation

        if self.record_video:
            self.video.close()

        sys.exit(self.mav_view.app.exec_())


if __name__ == "__main__":
    simulator = AutopilotSimulator()
    simulator.simulate()
