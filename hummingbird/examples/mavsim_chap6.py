import numpy as np
from hummingbird import parameters as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from chap6.autopilot import autopilot
from hummingbird.tools import signals
from hummingbird.message_types.msg_autopilot import MsgAutopilot

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
DATA = True
if DATA:
    pos = [1500, 0]  # x, y position on screen
    data_view = data_viewer(*pos)  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)

# autopilot commands
commands = MsgAutopilot()
Va_command = signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.01)
h_command = signals(dc_offset=100.0,
                    amplitude=10.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = signals(dc_offset=np.radians(0),
                      amplitude=np.radians(45),
                      start_time=5.0,
                      frequency=0.015)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Q to exit...")
while sim_time < SIM.end_time:

    # -------controller-------------
    estimated_state = mav.true_state  # uses true states in the control
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)
    delta, commanded_state = ctrl.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    if DATA:
        data_view.update(mav.true_state,  # true states
                         mav.true_state,  # estimated states
                         commanded_state,  # commanded states
                         SIM.ts_simulation)

    # -------increment time-------------
    sim_time += SIM.ts_simulation

print("Finished")
