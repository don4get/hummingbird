from hummingbird.parameters import simulation_parameters as sim_p
import sys


class Simulator:
    def __init__(self, record_video=False):
        self.mav = None
        self.autopilot = None
        self.sim_time = sim_p.start_time
        self.video = None
        self.record_video = record_video

        self.sim_p = sim_p


# TODO: Use this kind of decorator to run simulations
def simulate(loop_func, start_time, end_time, viewer):

    def run_loops():
        sim_time = start_time
        while sim_time < end_time:
            loop_func()
            sim_time += sim_p.dt_simulation

        sys.exit(viewer.app.exec_())

    return run_loops


