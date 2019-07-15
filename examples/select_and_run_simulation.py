from hummingbird.simulation.kinematics_simulator import KinematicsSimulator
from hummingbird.simulation.dynamics_simulator import DynamicsSimulator
from hummingbird.simulation.physics_simulator import PhysicsSimulator
from hummingbird.simulation.trim_simulator import TrimSimulator
from hummingbird.simulation.autopilot_simulator import AutopilotSimulator
from hummingbird.simulation.sensor_simulator import SensorSimulator
from hummingbird.simulation.observer_simulator import ObserverSimulator
from hummingbird.simulation.path_follower_simulator import PathFollowerSimulator
from hummingbird.simulation.path_manager_simulator import PathManagerSimulator
from hummingbird.simulation.path_planner_simulator import PathPlannerSimulator
from examples.create_gifs import create_gifs

def main(conf, record_video):

    if conf == 2:
        simu = KinematicsSimulator(record_video=record_video)
    elif conf == 3:
        simu = DynamicsSimulator(record_video=record_video)
    elif conf == 4:
        simu = PhysicsSimulator(record_video=record_video)
    elif conf == 5:
        simu = TrimSimulator(record_video=record_video)
    elif conf == 6:
        simu = AutopilotSimulator(record_video=record_video)
    elif conf == 7:
        simu = SensorSimulator(record_video=record_video)
    elif conf == 8:
        simu = ObserverSimulator(record_video=record_video)
    elif conf == 10:
        simu = PathFollowerSimulator(record_video=record_video)
    elif conf == 11:
        simu = PathManagerSimulator(record_video=record_video)
    elif conf == 12:
        simu = PathPlannerSimulator(record_video=record_video)

    simu.sim_p.end_time = 100
    simu.simulate()


if __name__ == "__main__":
    main(12, True)
    create_gifs()
