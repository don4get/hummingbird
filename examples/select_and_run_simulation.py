from hummingbird.simulation.kinematics_simulator import KinematicsSimulator
from hummingbird.simulation.dynamics_simulator import DynamicsSimulator


def main():
    conf = 2

    if conf == 1:
        simu = KinematicsSimulator()
    elif conf == 2:
        simu = DynamicsSimulator(True, config="windy")
    simu.sim_p.end_time = 100
    simu.simulate()


if __name__ == "__main__":
    main()
