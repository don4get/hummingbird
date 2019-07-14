from hummingbird.simulation.kinematics_simulator import KinematicsSimulator
from hummingbird.simulation.dynamics_simulator import DynamicsSimulator
from hummingbird.simulation.fixedwing_dynamics_simulator import FixedwingDynamicsSimulator
from hummingbird.simulation.fixedwing_physics_simulator import FixedwingPhysicsSimulator


def main():
    conf = 4

    if conf == 1:
        simu = KinematicsSimulator()
    elif conf == 2:
        simu = DynamicsSimulator(True, config="windy")
    elif conf == 3:
        simu = FixedwingDynamicsSimulator()
    elif conf == 4:
        simu = FixedwingPhysicsSimulator()
    simu.sim_p.end_time = 10
    simu.simulate()


if __name__ == "__main__":
    main()
