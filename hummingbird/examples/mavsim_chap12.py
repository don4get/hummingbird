import numpy as np
from hummingbird.parameters import simulation_parameters as sim, planner_parameters\
    as plan
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.control.autopilot import Autopilot
from hummingbird.physics.mav import Mav
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.path_follower import PathFollower
from hummingbird.guidance.path_manager \
    import PathManager
from hummingbird.graphics.world_viewer import WorldViewer
from hummingbird.guidance.path_planner import PathPlanner
from hummingbird.message_types.msg_map import MsgMap

enable_data = True

# initialize the visualization
world_view = WorldViewer()  # initialize the viewer
if enable_data:
    screen_pos = [0, 0]  # x, y position on screen
    data_view = DataViewer(*screen_pos)  # initialize view of data plots

# initialize elements of the architecture
wind = WindSimulation()
mav = Mav()
ctrl = Autopilot(sim.dt_controller)
obsv = Observer(sim.dt_observer)
path_follow = PathFollower()
path_manage = PathManager()
path_plan = PathPlanner()

msg_map = MsgMap(plan)

# initialize the simulation time
sim_time = sim.start_time

delta = np.zeros(4)
mav.dynamics.update(delta)  # propagate the MAV dynamics

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < sim.end_time:
    # -------observer-------------
    measurements = mav.sensors.update_sensors(mav.dynamics.true_state, mav.dynamics._forces)  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    # -------path planner - ----
    if path_manage.flag_need_new_waypoints == 1:
        waypoints = path_plan.update(msg_map, estimated_state)

    # -------path manager-------------
    path = path_manage.update(waypoints, plan.R_min, estimated_state)

    # -------path follower-------------
    autopilot_commands = path_follow.update(path, estimated_state)

    # -------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.dynamics.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    world_view.update(msg_map, waypoints, path, mav.dynamics.true_state)  # plot path and MAV
    if enable_data:
        data_view.update(mav.dynamics.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         sim.dt_simulation)

    sim_time += sim.dt_simulation
