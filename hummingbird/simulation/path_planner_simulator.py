import sys
from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.video_writer import VideoWriter

from hummingbird.parameters.planner_parameters import PlannerParameters
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.fixed_wing import FixedWing
from hummingbird.control.autopilot import Autopilot
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.path_follower import PathFollower
from hummingbird.guidance.path_manager import PathManager
from hummingbird.graphics.world_viewer import WorldViewer
from hummingbird.guidance.path_planner import PathPlanner
from hummingbird.message_types.msg_map import MsgMap


class PathPlannerSimulator(Simulator):
    def __init__(self, record_video=False, display_data=True):
        Simulator.__init__(self, record_video)

        if self.record_video:
            self.video = VideoWriter(video_name="path_planner.avi",
                                     bounding_box=(0, 0, 800, 600),
                                     output_rate=self.sim_p.dt_video)
        self.display_data = display_data

        self.sim_p.end_time = 50.
        self.world_view = WorldViewer()  # initialize the viewer
        self.data_view = DataViewer(800, 0)
        self.mav = FixedWing()
        self.wind = WindSimulation()
        self.ctrl = Autopilot(self.sim_p.dt_controller)
        self.obsv = Observer(self.sim_p.dt_controller)
        self.measurements = self.mav.sensors.sensors

        self.path_follow = PathFollower()
        self.path_manage = PathManager()
        self.plan_p = PlannerParameters()

        self.path_plan = PathPlanner()

        self.msg_map = MsgMap(self.plan_p)

    def simulate(self):
        while self.sim_time < self.sim_p.end_time:

            # -------observer-------------
            measurements = self.mav.sensors.update_sensors(self.mav.dynamics.true_state,
                                                           self.mav.dynamics._forces)  # get sensor measurements
            estimated_state = self.obsv.update(measurements)  # estimate states from measurements

            # -------path planner - ----
            if self.path_manage.flag_need_new_waypoints == 1:
                waypoints = self.path_plan.update(self.msg_map, estimated_state)

            # -------path manager-------------
            path = self.path_manage.update(waypoints, self.plan_p.R_min, estimated_state)

            # -------path follower-------------
            autopilot_commands = self.path_follow.update(path, estimated_state)

            # -------controller-------------
            delta, commanded_state = self.ctrl.update(autopilot_commands, estimated_state)

            # -------physical system-------------
            current_wind = self.wind.update()  # get the new wind vector
            self.mav.dynamics.update(delta, current_wind)  # propagate the MAV dynamics

            # -------update viewer-------------
            self.world_view.update(self.msg_map, waypoints, path, self.mav.dynamics.true_state)  # plot path and MAV
            if self.display_data:
                self.data_view.update(self.mav.dynamics.true_state,  # true states
                                      estimated_state,  # estimated states
                                      commanded_state,  # commanded states
                                      self.sim_p.dt_simulation)

            self.sim_time += self.sim_p.dt_simulation

            if self.record_video:
                self.video.update(self.sim_time)

        if self.record_video:
            self.video.close()

        sys.exit(self.world_view.app.exec_())


if __name__ == "__main__":
    simulator = PathPlannerSimulator()
    simulator.simulate()
