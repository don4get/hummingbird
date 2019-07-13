import numpy as np
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.physics.wind_simulation import WindSimulation
from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.control.autopilot import Autopilot
from hummingbird.tools.signals import Signals
from hummingbird.message_types.msg_autopilot import MsgAutopilot

enable_data = True

# initialize the visualization
mav_view = MavViewer()  # initialize the mav viewer
if enable_data:
    pos = [1500, 0]  # x, y position on screen
    data_view = DataViewer(*pos)  # initialize view of data plots

# initialize elements of the architecture
wind = WindSimulation(sim_p.ts_simulation)
mav = MavDynamics(sim_p.ts_simulation)
ctrl = Autopilot(sim_p.ts_controller)

# autopilot commands
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.01)
h_command = Signals(dc_offset=100.0,
                    amplitude=10.0,
                    start_time=0.0,
                    frequency=0.02)
chi_command = Signals(dc_offset=np.radians(0),
                      amplitude=np.radians(45),
                      start_time=5.0,
                      frequency=0.015)

# initialize the simulation time
sim_time = sim_p.start_time

# main simulation loop
print("Press Q to exit...")
while sim_time < sim_p.end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = chi_command.square(sim_time)
    commands.altitude_command = h_command.square(sim_time)

    # -------controller-------------
    estimated_state = mav.true_state  # uses true states in the control
    delta, commanded_state = ctrl.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics
    mav.update_sensors()  # update the sensors

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    if enable_data:
        data_view.update(mav.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         sim_p.ts_simulation)

    # -------increment time-------------
    sim_time += sim_p.ts_simulation
