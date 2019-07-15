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


def main():
    conf = 12

    if conf == 2:
        simu = KinematicsSimulator()
    elif conf == 3:
        simu = DynamicsSimulator()
    elif conf == 4:
        simu = PhysicsSimulator()
    elif conf == 5:
        simu = TrimSimulator()
    elif conf == 6:
        simu = AutopilotSimulator()
    elif conf == 7:
        simu = SensorSimulator()
    elif conf == 8:
        simu = ObserverSimulator()
    elif conf == 10:
        simu = PathFollowerSimulator()
    elif conf == 11:
        simu = PathManagerSimulator()
    elif conf == 12:
        simu = PathPlannerSimulator()

    simu.sim_p.end_time = 100
    simu.simulate()


if __name__ == "__main__":
    main()
