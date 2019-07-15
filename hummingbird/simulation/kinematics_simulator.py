from hummingbird.simulation.simulator import Simulator
from hummingbird.graphics.mav_viewer import MavViewer
from hummingbird.message_types.msg_state import MsgState
from hummingbird.graphics.video_writer import VideoWriter


class KinematicsSimulator(Simulator):
    def __init__(self, record_video=False, config="translation"):
        Simulator.__init__(self, record_video)
        if self.record_video:
            self.video = VideoWriter(video_name="kinematics.avi",
                                     bounding_box=(0, 0, 800, 600),
                                     output_rate=self.sim_p.dt_video)
        self.msg_state = MsgState()
        self.mav_viewer = MavViewer()
        self.config = config

    def simulate(self):
        while self.sim_time < self.sim_p.end_time:
            if self.config == "translation":
                self.trans_rot(self.sim_time)
            elif self.config == "rotation":
                self.rot_trans(self.sim_time)

            self.mav_viewer.update(self.msg_state)
            if self.record_video:
                self.video.update(self.sim_time)

            self.sim_time += self.sim_p.dt_simulation

        if self.record_video:
            self.video.close()

    def trans_rot(self, t):
        if t < self.sim_p.end_time / 12:
            self.msg_state.pn += 10 * self.sim_p.dt_simulation
        elif t < 2 * self.sim_p.end_time / 12:
            self.msg_state.pe += 10 * self.sim_p.dt_simulation
        elif t < 3 * self.sim_p.end_time / 12:
            self.msg_state.h += 10 * self.sim_p.dt_simulation
        elif t < 4 * self.sim_p.end_time / 12:
            self.msg_state.phi += 0.2 * self.sim_p.dt_simulation
        elif t < 5 * self.sim_p.end_time / 12:
            self.msg_state.theta += 0.2 * self.sim_p.dt_simulation
        elif t < 6 * self.sim_p.end_time / 12:
            self.msg_state.psi += 0.2 * self.sim_p.dt_simulation
        elif t < 7 * self.sim_p.end_time / 12:
            self.msg_state.psi -= 0.2 * self.sim_p.dt_simulation
        elif t < 8 * self.sim_p.end_time / 12:
            self.msg_state.theta -= 0.2 * self.sim_p.dt_simulation
        elif t < 9 * self.sim_p.end_time / 12:
            self.msg_state.phi -= 0.2 * self.sim_p.dt_simulation
        elif t < 10 * self.sim_p.end_time / 12:
            self.msg_state.h -= 10 * self.sim_p.dt_simulation
        elif t < 11 * self.sim_p.end_time / 12:
            self.msg_state.pe -= 10 * self.sim_p.dt_simulation
        else:
            self.msg_state.pn -= 10 * self.sim_p.dt_simulation

    def rot_trans(self, t):
        if t < self.sim_p.end_time / 6:
            self.msg_state.phi += 0.1 * self.sim_p.dt_simulation
        elif t < 2 * self.sim_p.end_time / 6:
            self.msg_state.theta += 0.1 * self.sim_p.dt_simulation
        elif t < 3 * self.sim_p.end_time / 6:
            self.msg_state.psi += 0.1 * self.sim_p.dt_simulation
        elif t < 4 * self.sim_p.end_time / 6:
            self.msg_state.pn += 10 * self.sim_p.dt_simulation
        elif t < 5 * self.sim_p.end_time / 6:
            self.msg_state.pe += 10 * self.sim_p.dt_simulation
        else:
            self.msg_state.h += 10 * self.sim_p.dt_simulation


if __name__ == "__main__":
    simulator = KinematicsSimulator()
    simulator.simulate()
