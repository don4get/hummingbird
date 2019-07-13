from hummingbird.parameters import simulation_parameters as sim_p


class Simulator:
    def __init__(self, record_video=False):
        self.mav = None
        self.autopilot = None
        self.sim_time = sim_p.start_time
        self.video = None
        self.record_video = record_video

        self.sim_p = sim_p

    def simulate(self):
        raise NotImplementedError
