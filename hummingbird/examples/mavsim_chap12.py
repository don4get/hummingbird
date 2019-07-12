import numpy as np
from hummingbird.parameters import simulation_parameters as sim, planner_parameters\
    as plan
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.control.autopilot import Autopilot
from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.pathfollower import PathFollower
from hummingbird.guidance.path_manager \
    import PathManager
from hummingbird.graphics.world_viewer import WorldViewer
from hummingbird.guidance.path_planner import PathPlanner

# initialize the visualization
world_view = WorldViewer()  # initialize the viewer
DATA = True
if DATA:
    screen_pos = [2000, 0]  # x, y position on screen
    data_view = DataViewer(*screen_pos)  # initialize view of data plots

# initialize elements of the architecture
wind = WindSimulation(sim.ts_simulation)
mav = MavDynamics(sim.ts_simulation)
ctrl = Autopilot(sim.ts_simulation)
obsv = Observer(sim.ts_simulation)
path_follow = PathFollower()
path_manage = PathManager()
path_plan = PathPlanner()

from hummingbird.message_types.msg_map import MsgMap

map = MsgMap(plan)

# initialize the simulation time
sim_time = sim.start_time

delta = np.zeros(4)
mav.update(delta)  # propagate the MAV dynamics

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < sim.end_time:
    # -------observer-------------
    measurements = mav.update_sensors()  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    # -------path planner - ----
    if path_manage.flag_need_new_waypoints == 1:
        waypoints = path_plan.update(map, estimated_state)

    # -------path manager-------------
    path = path_manage.update(waypoints, plan.R_min, estimated_state)

    # -------path follower-------------
    autopilot_commands = path_follow.update(path, estimated_state)

    # -------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    world_view.update(map, waypoints, path, mav.true_state)  # plot path and MAV
    if DATA:
        data_view.update(mav.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         sim.ts_simulation)

    sim_time += sim.ts_simulation
