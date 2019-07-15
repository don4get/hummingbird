import sys
import numpy as np
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter
from hummingbird.physics.fixed_wing_dynamics import FixedWingDynamics
from hummingbird.control.autopilot import Autopilot
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.tools.trim import compute_trim
from hummingbird.tools.compute_models import compute_tf_model
from hummingbird.tools.signals import Signals
from hummingbird.message_types.msg_autopilot import MsgAutopilot
from hummingbird.tools.rotations import Quaternion2Euler


class AutopilotTuningSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="trim.avi",
                                     bounding_box=(0, 0, 1000, 1000),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 500.
        self.mav_view = MavViewer()
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWingDynamics()
        self.ctrl = Autopilot(self.sim_p.dt_controller)

        # autopilot commands
        self.commands = MsgAutopilot()
        self.Va_command = Signals(dc_offset=25.0,
                                  amplitude=3.0,
                                  start_time=5.0,
                                  frequency=0.01)
        self.h_command = Signals(dc_offset=100.0,
                                 amplitude=10.0,
                                 start_time=0.0,
                                 frequency=0.01)
        self.chi_command = Signals(dc_offset=np.radians(0),
                                   amplitude=np.radians(45),
                                   start_time=5.0,
                                   frequency=0.2)

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

        reformated_trimmed_state = np.zeros(12)
        reformated_trimmed_state[0:6] = trim_state[0:6]
        reformated_trimmed_state[6:9] = Quaternion2Euler(trim_state[6:10])
        reformated_trimmed_state[9:12] = trim_state[10:13]
        self.ctrl.set_gains_from_trim(reformated_trimmed_state, trim_input)

        # main simulation loop
        print("Press Command-Q to exit...")
        while sim_time < self.sim_p.end_time:

            while self.sim_time < self.sim_p.end_time:

                # -------controller-------------
                estimated_state = self.mav.true_state  # uses true states in the control
                self.commands.airspeed_command = self.Va_command.square(self.sim_time)
                self.commands.course_command = self.chi_command.sinusoid(self.sim_time)
                self.commands.altitude_command = self.h_command.square(self.sim_time)
                delta, commanded_state = self.ctrl.update(self.commands, estimated_state)

                # -------physical system-------------
                self.mav.update(delta)  # propagate the MAV dynamics

                # -------update viewer-------------
                self.mav_view.update(self.mav.true_state)  # plot body of MAV
                if self.display_data:
                    self.data_view.update(self.mav.true_state,  # true states
                                          self.mav.true_state,  # estimated states
                                          commanded_state,  # commanded states
                                          self.sim_p.dt_simulation)

                # -------increment time-------------
                self.sim_time += self.sim_p.dt_simulation

        sys.exit(self.mav_view.app.exec_())


if __name__ == "__main__":
    simulator = AutopilotTuningSimulator()
    simulator.simulate()
