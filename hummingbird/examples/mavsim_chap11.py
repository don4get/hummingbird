import numpy as np
from hummingbird.parameters import simulation_parameters as sim_p, planner_parameters as plan_p
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.graphics.waypoint_viewer import WaypointViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.mav import Mav
from hummingbird.control.autopilot import Autopilot
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.path_follower import PathFollower
from hummingbird.guidance.path_manager import PathManager
from hummingbird.message_types.msg_waypoints import MsgWaypoints

enable_data = True

# initialize the visualization
waypoint_view = WaypointViewer()  # initialize the viewer
if enable_data:
    screen_pos = [0, 0]  # x, y position on screen
    data_view = DataViewer(*screen_pos)  # initialize view of data plots

# initialize elements of the architecture
wind = WindSimulation()
mav = Mav()
ctrl = Autopilot(sim_p.dt_controller)
obsv = Observer(sim_p.dt_observer)
path_follow = PathFollower()
path_manage = PathManager()

# waypoint definition

waypoints = MsgWaypoints()
# waypoints.type = 'straight_line'
waypoints.type = 'fillet'
waypoints.type = 'dubins'
waypoints.num_waypoints = 4
Va = plan_p.Va0
waypoints.ned[:waypoints.num_waypoints] = np.array([[0, 0, -100],
                                                    [1000, 0, -100],
                                                    [0, 1000, -100],
                                                    [1000, 1000, -100]])
waypoints.airspeed[:waypoints.num_waypoints] = np.array([Va, Va, Va, Va])
waypoints.course[:waypoints.num_waypoints] = np.array([np.radians(0),
                                                       np.radians(45),
                                                       np.radians(45),
                                                       np.radians(-135)])

# initialize the simulation time
sim_time = sim_p.start_time

delta = np.zeros(4)
mav.dynamics.update(delta)  # propagate the MAV dynamics
mav.sensors.update_sensors(mav.dynamics.true_state, mav.dynamics._forces)
# main simulation loop
print("Press Q to exit...")
while sim_time < sim_p.end_time:
    # -------observer-------------
    measurements = mav.sensors.update_sensors(mav.dynamics.true_state, mav.dynamics._forces)  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    # -------path manager-------------
    path = path_manage.update(waypoints, plan_p.R_min, estimated_state)

    # -------path follower-------------
    # autopilot_commands = path_follow.update(path, estimated_state)
    autopilot_commands = path_follow.update(path, mav.dynamics.true_state)

    # -------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.dynamics.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    if not waypoint_view.plot_initialized:
        waypoint_view.update(waypoints, path, mav.dynamics.true_state)  # plot path and MAV
        path.flag_path_changed = True
        waypoint_view.update(waypoints, path, mav.dynamics.true_state)  # plot path and MAV
    else:
        waypoint_view.update(waypoints, path, mav.dynamics.true_state)  # plot path and MAV

    if enable_data:
        data_view.update(mav.dynamics.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         sim_p.dt_simulation)

    # -------increment time-------------
    sim_time += sim_p.dt_simulation
