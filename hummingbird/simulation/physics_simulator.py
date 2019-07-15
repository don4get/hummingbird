from hummingbird.simulation.simulator import Simulator, simulate
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.video_writer import VideoWriter
from hummingbird.physics.fixed_wing_dynamics import FixedWingDynamics
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
import numpy as np
import sys


class PhysicsSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True, config="still_air"):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="physics.avi",
                                     bounding_box=(0, 0, 1000, 1000),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.mav_view = MavViewer()
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWingDynamics()
        self.config = config
        if self.config == "windy":
            self.wind = WindSimulation(self.sim_p.ts_simulation)

    def simulate(self):
        while self.sim_time < self.sim_p.end_time:
            # -------set control surfaces-------------
            delta_e = -0.5
            delta_a = 0.
            delta_r = 0.
            delta_t = 1.0

            delta = np.array([delta_e, delta_a, delta_r, delta_t])

            # -------physical system-------------
            Va = self.mav._Va  # grab updated Va from MAV dynamics
            # current_wind = self.wind.update(Va)  # get the new wind vector
            self.mav.update(delta, np.zeros(6))  # propagate the MAV dynamics

            # -------update viewer-------------
            self.mav_view.update(self.mav.true_state)  # plot body of MAV
            if self.display_data:
                self.data_view.update(self.mav.true_state,  # true states
                                      self.mav.true_state,  # estimated states
                                      self.mav.true_state,  # commanded states
                                      self.sim_p.dt_simulation)
            if self.record_video:
                self.video.update(self.sim_time)

            # -------increment time-------------
            self.sim_time += self.sim_p.dt_simulation
        if self.record_video:
            self.video.close()
        sys.exit(self.mav_view.app.exec_())
