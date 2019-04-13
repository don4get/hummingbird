"""
mavsim
    - Chapter 2 assignment for Beard & McLain, PUP, 2012
    - Update history:
        1/10/2019 - RWB
"""
import sys
sys.path.append('..')

from mav_viewer import MAV_Viewer
from mav_dynamics import mav_dynamics as Dynamics
import parameters.sim_params as SIM
from messages.state_msg import StateMsg
from data_viewer import data_viewer
from wind_simulation import wind_simulation
import numpy as np

# initialize dynamics object
dyn = Dynamics(SIM.ts_sim)
wind = wind_simulation(SIM.ts_sim)

mav_view = MAV_Viewer()
# data_view = data_viewer()

# initialize the simulation time
sim_time = SIM.t0

# main simulation loop
print("Press Ctrl-Q to exit...")
while sim_time < SIM.t_end:
    #---- Will set deltas for rudder and stuff to calculate forces
    delta_e = -0.3
    delta_t = 0.5
    delta_a = 0.0 #0.0
    delta_r = 0.0 #0.005

    if sim_time < 15:
        delta_r = 0.002
    elif sim_time < 25:
        delta_a = 0.025
    elif sim_time < 29.5:
        delta_a = -.025
    else:
        delta_t = 1.0

    delta = np.array([[delta_e, delta_t, delta_a, delta_r]]).T

    #---Get the wind here
    current_wind = wind.update(dyn._Va)
    dyn.update_state(delta, current_wind)
    #-------update viewer---------------
    mav_view.update(dyn.msg_true_state)
    # data_view.update(dyn.msg_true_state,
    #                 dyn.msg_true_state,
    #                 dyn.msg_true_state,
    #                 SIM.ts_sim)

    #-------increment time-------------
    sim_time += SIM.ts_sim
