import sys
import numpy as np
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter

from hummingbird.graphics.path_viewer import PathViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.fixed_wing import FixedWing
from hummingbird.control.autopilot import Autopilot
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.path_follower import PathFollower
from hummingbird.message_types.msg_path import MsgPath


class PathFollowerSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True, config="line"):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="path_follower_"+config+".avi",
                                     bounding_box=(0, 0, 800, 600),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.path_view = PathViewer()  # initialize the viewer
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWing()
        self.wind = WindSimulation()
        self.ctrl = Autopilot(self.sim_p.dt_controller)
        self.obsv = Observer(self.sim_p.dt_controller)
        self.measurements = self.mav.sensors.sensors

        self.path_follow = PathFollower()

        self.path = MsgPath()
        self.path.type = config
        if self.path.type == 'line':
            self.path.line_origin = np.array([0.0, 0.0, -100.0])
            self.path.line_direction = np.array([0.5, 1.0, 0.0])
            self.path.line_direction = self.path.line_direction / np.linalg.norm(self.path.line_direction)
        else:  # path.type == 'orbit'
            self.path.orbit_center = np.array([0.0, 0.0, -100.0])  # center of the orbit
            self.path.orbit_radius = 300.0  # radius of the orbit
            self.path.orbit_direction = 'CW'  # orbit direction: 'CW'==clockwise, 'CCW'==counter clockwise

    def simulate(self):
        while self.sim_time < self.sim_p.end_time:

            # -------observer-------------
            self.measurements = self.mav.sensors.update_sensors(self.mav.dynamics.true_state,
                                                                self.mav.dynamics._forces)  # get sensor measurements
            estimated_state = self.obsv.update(self.measurements)  # estimate states from measurements

            # -------path follower-------------
            autopilot_commands = self.path_follow.update(self.path, estimated_state)
            # autopilot_commands = self.path_follow.update(self.path, self.mav.dynamics.true_state)  # for debugging

            # -------controller-------------
            delta, commanded_state = self.ctrl.update(autopilot_commands, estimated_state)

            # -------physical system-------------
            current_wind = self.wind.update()  # get the new wind vector
            self.mav.dynamics.update(delta, current_wind)  # propagate the MAV dynamics

            # -------update viewer-------------
            self.path_view.update(self.path, self.mav.dynamics.true_state)  # plot path and MAV
            if self.display_data:
                self.data_view.update(self.mav.dynamics.true_state,  # true states
                                      estimated_state,  # estimated states
                                      commanded_state,  # commanded states
                                      self.sim_p.dt_simulation)

            if self.record_video:
                self.video.update(self.sim_time)

            # -------increment time-------------
            self.sim_time += self.sim_p.dt_simulation

        if self.record_video:
            self.video.close()

        sys.exit(self.path_view.app.exec_())


if __name__ == "__main__":
    simulator = PathFollowerSimulator()
    simulator.simulate()
