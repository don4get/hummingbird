import numpy as np
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.video_writer import VideoWriter
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.physics.wind_simulation import WindSimulation

# initialize the visualization
enable_video = False
enable_data = True

mav_view = MavViewer()  # initialize the mav viewer

if enable_data:
    data_view = DataViewer()  # initialize view of data plots
if enable_video:
    video = VideoWriter(video_name="chap4_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=sim_p.dt_video)

# initialize elements of the architecture
wind = WindSimulation(sim_p.dt_simulation)
mav = MavDynamics(sim_p.dt_simulation)
Va = 0

# initialize the simulation time
sim_time = sim_p.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < sim_p.end_time:
    # -------set control surfaces-------------
    delta_e = -0.1
    delta_a = 0.
    delta_r = 0.
    delta_t = 0.6

    delta = np.array([delta_e, delta_a, delta_r, delta_t])

    # -------physical system-------------
    Va = mav._Va  # grab updated Va from MAV dynamics
    current_wind = wind.update(Va)  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    if enable_data:
        data_view.update(mav.true_state,  # true states
                         mav.true_state,  # estimated states
                         mav.true_state,  # commanded states
                         sim_p.dt_simulation)
    if enable_video:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += sim_p.dt_simulation

if enable_video:
    video.close()
