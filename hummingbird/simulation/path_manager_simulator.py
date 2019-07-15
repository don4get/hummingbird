import sys
import numpy as np
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter

from hummingbird.parameters import simulation_parameters as sim_p, planner_parameters as plan_p
from hummingbird.graphics.waypoint_viewer import WaypointViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.fixed_wing import FixedWing
from hummingbird.control.autopilot import Autopilot
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.path_follower import PathFollower
from hummingbird.guidance.path_manager import PathManager
from hummingbird.message_types.msg_waypoints import MsgWaypoints


class PathManagerSimulator(Simulator):
    """
    config: 'straight_line' 'fillet' 'dubins'

    """
    def __init__(self, record_video=False, display_data=True, config="fillet"):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="sensors.avi",
                                     bounding_box=(0, 0, 1000, 1000),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.waypoint_view = WaypointViewer()  # initialize the viewer
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWing()
        self.wind = WindSimulation()
        self.ctrl = Autopilot(sim_p.dt_controller)
        self.obsv = Observer(sim_p.dt_controller)
        self.measurements = self.mav.sensors.sensors

        self.path_follow = PathFollower()
        self.path_manage = PathManager()

        self.waypoints = MsgWaypoints()
        self.waypoints.type = config
        self.waypoints.num_waypoints = 4
        Va = plan_p.Va0
        self.waypoints.ned[:self.waypoints.num_waypoints] = np.array([[0, 0, -100],
                                                            [1000, 0, -100],
                                                            [0, 1000, -100],
                                                            [1000, 1000, -100]])
        self.waypoints.airspeed[:self.waypoints.num_waypoints] = np.array([Va, Va, Va, Va])
        self.waypoints.course[:self.waypoints.num_waypoints] = np.array([np.radians(0),
                                                               np.radians(45),
                                                               np.radians(45),
                                                               np.radians(-135)])

    def simulate(self):
        while self.sim_time < sim_p.end_time:

            # -------observer-------------
            measurements = self.mav.sensors.update_sensors(self.mav.dynamics.true_state,
                                                           self.mav.dynamics._forces)  # get sensor measurements
            estimated_state = self.obsv.update(measurements)  # estimate states from measurements

            # -------path manager-------------
            path = self.path_manage.update(self.waypoints, plan_p.R_min, estimated_state)

            # -------path follower-------------
            # autopilot_commands = path_follow.update(path, estimated_state)
            autopilot_commands = self.path_follow.update(path, self.mav.dynamics.true_state)

            # -------controller-------------
            delta, commanded_state = self.ctrl.update(autopilot_commands, estimated_state)

            # -------physical system-------------
            current_wind = self.wind.update()  # get the new wind vector
            self.mav.dynamics.update(delta, current_wind)  # propagate the MAV dynamics

            # -------update viewer-------------
            if not self.waypoint_view.plot_initialized:
                self.waypoint_view.update(self.waypoints, path, self.mav.dynamics.true_state)  # plot path and MAV
                path.flag_path_changed = True
                self.waypoint_view.update(self.waypoints, path, self.mav.dynamics.true_state)  # plot path and MAV
            else:
                self.waypoint_view.update(self.waypoints, path, self.mav.dynamics.true_state)  # plot path and MAV

            if self.display_data:
                self.data_view.update(self.mav.dynamics.true_state,  # true states
                                      estimated_state,  # estimated states
                                      commanded_state,  # commanded states
                                      sim_p.dt_simulation)

            # -------increment time-------------
            self.sim_time += sim_p.dt_simulation

        sys.exit(self.waypoint_view.app.exec_())

