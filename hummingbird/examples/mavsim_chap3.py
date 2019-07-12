import numpy as np
from hummingbird.physics.mav_dynamics import MavDynamics
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.graphics.data_viewer import DataViewer
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.message_types.msg_state import MsgState
from hummingbird.graphics.video_writer import VideoWriter



# initialize viewers and video
enable_video = True  # True==write video, False==don't write video
end_time = 50.
mav_view = MavViewer()
data_view = DataViewer()
mav = MavDynamics(sim_p.ts_simulation)
cmd_state = MsgState()
if enable_video:
    video = VideoWriter(video_name="chap3_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=sim_p.ts_video)

FM_list = ['fx', 'fy', 'fz', 'l', 'm', 'n']
# main simulation loop
for i in range(6):
    # initialize the simulation time
    sim_time = sim_p.start_time
    forces_moments = np.zeros(6)  # fx, fy, fz, l, m, n
    if i < 3:
        val = 100
        print('***** APPLYING FORCE {} OF {} N *****'.format(FM_list[i], val))
    else:
        val = 0.05
        print('***** APPLYING MOMENT {} OF {} N-m *****'.format(FM_list[i], val))
    forces_moments[i] = val
    mav.reset_state()
    while sim_time < end_time:
        # -------vary states to check viewer-------------
        mav.update_true_state_from_forces_moments(forces_moments)

        # -------update viewer and video-------------
        mav_view.update(mav.true_state)  # plot body of MAV
        data_view.update(mav.true_state,  # true states
                         mav.true_state,  # estimated states
                         mav.true_state,  # commanded states
                         sim_p.ts_simulation)

        if enable_video:
            video.update(sim_time)

        # -------increment time-------------
        sim_time += sim_p.ts_simulation

print("Press Ctrl-Q to exit...")
if enable_video:
    video.close()
