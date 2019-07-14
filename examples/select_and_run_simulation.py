from hummingbird.simulation.kinematics_simulator import KinematicsSimulator
from hummingbird.simulation.dynamics_simulator import DynamicsSimulator
from hummingbird.simulation.physics_simulator import PhysicsSimulator
from hummingbird.simulation.trim_simulator import TrimSimulator

def main():
    conf = 4

    if conf == 1:
        simu = KinematicsSimulator()
    elif conf == 2:
        simu = DynamicsSimulator()
    elif conf == 3:
        simu = PhysicsSimulator()
    elif conf == 4:
        simu = TrimSimulator()

    simu.sim_p.end_time = 100
    simu.simulate()


if __name__ == "__main__":
    main()
