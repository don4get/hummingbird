"""
compute_trim
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Rotation, Quaternion2Euler
from mav_dynamics import mav_dynamics as Dynamics

def compute_trim(mav, Va, gamma):
    # define initial state and input
    e = Euler2Quaternion(0, gamma, 0)
    state0 = np.array([[0., 0., -100., Va, 0., 0.,
                        e.item(0), e.item(1), e.item(2), e.item(3), 0., 0., 0.]]).T
    delta0 = np.array([[0., 0.5, 0., 0.]]).T
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7], # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9], # e3=0
                                x[10], # p=0  - angular rates should all be zero
                                x[11], # q=0
                                x[12], # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective, x0, method='SLSQP', args = (mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T #These inputs are the same. Do I need to recalculate them?
    return trim_state, trim_input

# objective function to be minimized
def trim_objective(x, mav, Va, gamma):
    state = x[:13]
    delta = x[13:]

    x_dot = np.array([[0., 0., -Va * np.sin(gamma), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

    mav._state = state
    mav._state = mav._state.reshape((13, 1))
    mav.updateVelocityData()
    forces_moments = mav.calcForcesAndMoments(delta)
    f = mav._derivatives(mav._state, forces_moments)

    temp = x_dot[2:] - f[2:]
    J = np.linalg.norm(temp)**2
    return J

if __name__ == "__main__":
    mav = Dynamics(.02)
    Va = 25.0
    gamma = 0.0
    mav._Va = Va

    # x = np.array([[0., 0., -100., Va, 0., 0., # last element is 0.1 for f and 0 for f_m, Va is 25 and gamma is 0
    #                1., 0., 0., 0., 0., 0., 0.]]).T
    # delta = np.array([[0., 0.5, 0., 0.]]).T
    # mav._state = x
    # mav.updateVelocityData()
    # f_m = mav.calcForcesAndMoments(delta)
    # f = mav._derivatives(mav._state, f_m)
    # print(delta)
    # # print(f)

    trim_state, trim_input = compute_trim(mav, Va, gamma) # Why don't I need R??
    phi, theta, psi = Quaternion2Euler(trim_state[6:10])
    # print('Phi: ', phi)
    # print('Theta: ', theta)
    # print('Psi: ', psi)
    print("State: ", trim_state)
    print("Inputs: ", trim_input)
