import numpy as np
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.tools.trim import compute_trim
from hummingbird.tools.compute_models import compute_tf_model

enable_data = False

# initialize the visualization
mav_view = MavViewer()  # initialize the mav viewer
if enable_data:
    data_view = DataViewer()  # initialize view of data plots

# initialize elements of the architecture
wind = WindSimulation(sim_p.dt_simulation)
mav = MavDynamics(sim_p.dt_simulation)

# use compute_trim function to compute trim state and trim input
Va = 25.
gamma = 0. * np.pi / 180.
trim_state, trim_input = compute_trim(mav, Va, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
delta = np.copy(trim_input)  # set input to constant constant trim input
print('trim_input:', trim_input)
print('trim_state:', trim_state)
# delta[-1] = 1.0

# # compute the state space model linearized about trim
# A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, \
T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r \
    = compute_tf_model(mav, trim_state, trim_input)

# initialize the simulation time
sim_time = sim_p.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < sim_p.end_time:

    # -------physical system-------------
    # current_wind = wind.update()  # get the new wind vector
    forces_moments = mav._forces_moments(delta)
    mav._update_velocity_data(np.zeros(6))
    print("Forces and moments: {}".format(forces_moments))
    mav.update_true_state_from_forces_moments(forces_moments)

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    if enable_data:
        data_view.update(mav.true_state,  # true states
                         mav.true_state,  # estimated states
                         mav.true_state,  # commanded states
                         sim_p.dt_simulation)

    # -------increment time-------------
    sim_time += sim_p.dt_simulation
