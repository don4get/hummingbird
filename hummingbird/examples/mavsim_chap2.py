from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.parameters import simulation_parameters as sim_p
from hummingbird.message_types.msg_state import MsgState
from hummingbird.graphics.video_writer import VideoWriter

# initialize messages
state = MsgState()  # instantiate state message

# initialize viewers and video
record_video = False  # True==write video, False==don't write video
# mav_view = spacecraft_viewer()
mav_view = MavViewer()
if record_video:
    video = VideoWriter(video_name="chap2_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=sim_p.ts_video)

# initialize the simulation time
sim_time = sim_p.start_time

def trans_rot(sim_time):
    if sim_time < sim_p.end_time / 6:
        state.pn += 10 * sim_p.ts_simulation
    elif sim_time < 2 * sim_p.end_time / 6:
        state.pe += 10 * sim_p.ts_simulation
    elif sim_time < 3 * sim_p.end_time / 6:
        state.h += 10 * sim_p.ts_simulation
    elif sim_time < 4 * sim_p.end_time / 6:
        state.phi += 0.1 * sim_p.ts_simulation
    elif sim_time < 5 * sim_p.end_time / 6:
        state.theta += 0.1 * sim_p.ts_simulation
    else:
        state.psi += 0.1 * sim_p.ts_simulation


def rot_trans(sim_time):
    if sim_time < sim_p.end_time / 6:
        state.phi += 0.1 * sim_p.ts_simulation
    elif sim_time < 2 * sim_p.end_time / 6:
        state.theta += 0.1 * sim_p.ts_simulation
    elif sim_time < 3 * sim_p.end_time / 6:
        state.psi += 0.1 * sim_p.ts_simulation
    elif sim_time < 4 * sim_p.end_time / 6:
        state.pn += 10 * sim_p.ts_simulation
    elif sim_time < 5 * sim_p.end_time / 6:
        state.pe += 10 * sim_p.ts_simulation
    else:
        state.h += 10 * sim_p.ts_simulation


# main simulation loop
T = 2.5
while sim_time < sim_p.end_time:
    # -------vary states to check viewer-------------
    # print(sim_time)

    trans_rot(sim_time)
    # rot_trans(sim_time)

    # -------update viewer and video-------------
    mav_view.update(state)
    if record_video:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += sim_p.ts_simulation

print("Press Ctrl-Q to exit...")
if record_video:
    video.close()
