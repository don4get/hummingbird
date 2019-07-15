import sys
import numpy as np
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter
from hummingbird.physics.fixed_wing_dynamics import FixedWingDynamics
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.tools.trim import compute_trim
from hummingbird.tools.compute_models import compute_tf_model


class TrimSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="trim.avi",
                                     bounding_box=(0, 0, 1000, 1000),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.mav_view = MavViewer()
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWingDynamics()

    def simulate(self):
        Va = 25.
        gamma = 0. * np.pi / 180.
        turn_radius = np.inf
        trim_state, trim_input = compute_trim(self.mav, Va, gamma, turn_radius)
        self.mav._state = trim_state  # set the initial state of the mav to the trim state
        self.mav.integrator.set_initial_value(self.mav._state, self.sim_p.start_time)
        self.mav._update_true_state()
        delta = np.copy(trim_input)  # set input to constant constant trim input
        print('trim_input:', trim_input)
        print('trim_state:', trim_state)
        # delta[-1] = 1.0

        # # compute the state space model linearized about trim
        # A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
        T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, \
        T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
            = compute_tf_model(self.mav, trim_state, trim_input)

        # initialize the simulation time
        sim_time = self.sim_p.start_time

        # main simulation loop
        print("Press Command-Q to exit...")
        while sim_time < self.sim_p.end_time:

            self.mav.update(delta)

            # -------update viewer-------------
            self.mav_view.update(self.mav.true_state)  # plot body of MAV
            if self.display_data:
                self.data_view.update(self.mav.true_state,  # true states
                                      self.mav.true_state,  # estimated states
                                      self.mav.true_state,  # commanded states
                                      self.sim_p.dt_simulation)

            # -------increment time-------------
            sim_time += self.sim_p.dt_simulation

        sys.exit(self.mav_view.app.exec_())


if __name__ == "__main__":
    simulator = TrimSimulator()
    simulator.simulate()
