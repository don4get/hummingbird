import sys
from hummingbird.parameters.simulation_parameters import SimulationParameters


class Simulator:
    def __init__(self, record_video=False):
        self.mav = None
        self.autopilot = None
        self.sim_time = SimulationParameters().start_time
        self.video = None
        self.record_video = record_video

        self.sim_p = SimulationParameters()


# TODO: Use this kind of decorator to run simulations
def simulate(loop_func, start_time, end_time, viewer):

    def run_loops():
        sim_time = start_time
        while sim_time < end_time:
            loop_func()
            sim_time += SimulationParameters.dt_simulation

        sys.exit(viewer.app.exec_())

    return run_loops


