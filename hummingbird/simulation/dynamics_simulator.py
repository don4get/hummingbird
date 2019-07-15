from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.video_writer import VideoWriter
from hummingbird.physics.fixed_wing_dynamics import FixedWingDynamics
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
import numpy as np
import sys


class DynamicsSimulator(Simulator):
    def __init__(self, record_video=False, config="still_air"):
        Simulator.__init__(self, record_video)
        if self.record_video:
            self.video = VideoWriter(video_name="dynamics.avi",
                                     bounding_box=(0, 0, 1000, 1000),
                                     output_rate=self.sim_p.dt_video)

        self.sim_p.end_time = 50.
        self.mav_view = MavViewer()
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWingDynamics()
        self.config = config
        if self.config == "windy":
            self.wind = WindSimulation(self.sim_p.ts_simulation)

    def simulate(self):
        fm_list = ['fx', 'fy', 'fz', 'l', 'm', 'n']
        # main simulation loop
        for i in range(6):
            # initialize the simulation time
            sim_time = self.sim_p.start_time
            forces_moments = np.zeros(6)  # fx, fy, fz, l, m, n
            if i < 3:
                val = 100  # [N]
                print('***** APPLYING FORCE {} OF {} N *****'.format(fm_list[i], val))
            else:
                val = 0.05  # [Nm]
                print('***** APPLYING MOMENT {} OF {} Nm *****'.format(fm_list[i], val))
            forces_moments[i] = val
            self.mav.reset_state()
            while sim_time < self.sim_p.end_time:
                # -------vary states to check viewer-------------
                Va = self.mav._Va
                self.mav.update_true_state_from_forces_moments(forces_moments)
                # current_wind = self.wind.update(Va)

                # -------update viewer and video-------------
                self.mav_view.update(self.mav.true_state)  # plot body of MAV
                self.data_view.update(self.mav.true_state,  # true states
                                      self.mav.true_state,  # estimated states
                                      self.mav.true_state,  # commanded states
                                      self.sim_p.dt_simulation)

                if self.record_video:
                    self.video.update(sim_time)

                # -------increment time-------------
                sim_time += self.sim_p.dt_simulation

        if self.record_video:
            self.video.close()

        sys.exit(self.mav_view.app.exec_())


if __name__ == "__main__":
    simulator = DynamicsSimulator()
    simulator.simulate()
