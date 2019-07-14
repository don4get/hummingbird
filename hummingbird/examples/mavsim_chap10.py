import numpy as np
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.graphics.path_viewer import PathViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.control.autopilot import Autopilot
from hummingbird.estimation.observer import Observer
from hummingbird.guidance.path_follower import PathFollower
from hummingbird.message_types.msg_path import MsgPath

enable_data = True

# initialize the visualization
path_view = PathViewer()  # initialize the viewer
if enable_data:
    pos = [0, 0]  # x, y position on screen
    data_view = DataViewer(*pos)  # initialize view of data plots

# initialize elements of the architecture
wind = WindSimulation(sim_p.ts_simulation)
mav = MavDynamics(sim_p.ts_simulation)
ctrl = Autopilot(sim_p.ts_controller)
obsv = Observer(sim_p.ts_observer)
path_follow = PathFollower()
measurements = mav.sensors

# path definition

path = MsgPath()
# path.type = 'line'
path.type = 'orbit'
if path.type == 'line':
    path.line_origin = np.array([0.0, 0.0, -100.0])
    path.line_direction = np.array([0.5, 1.0, 0.0])
    path.line_direction = path.line_direction / np.linalg.norm(path.line_direction)
else:  # path.type == 'orbit'
    path.orbit_center = np.array([0.0, 0.0, -100.0])  # center of the orbit
    path.orbit_radius = 300.0  # radius of the orbit
    path.orbit_direction = 'CW'  # orbit direction: 'CW'==clockwise, 'CCW'==counter clockwise

# initialize the simulation time
sim_time = sim_p.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < sim_p.end_time:
    # -------observer-------------
    measurements = mav.update_sensors()  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    # -------path follower-------------
    autopilot_commands = path_follow.update(path, estimated_state)
    # autopilot_commands = path_follow.update(path, mav.true_state)  # for debugging

    # -------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    path_view.update(path, mav.true_state)  # plot path and MAV
    if enable_data:
        data_view.update(mav.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         sim_p.ts_simulation)

    # -------increment time-------------
    sim_time += sim_p.ts_simulation
