#!/usr/bin/env python3
"""
######################
Extended Kalman Filter
######################

"""

import numpy as np
from abc import ABCMeta, abstractmethod


class EKF(object):
    """Extended Kalman Filter.

    :param x: State vector
    :param P: State covariance Matrix
    :param Q: Input noise covariance matrix
    :param R: Measurement noise covariance matrix
    :param N: Integration factor, see __call__
    """
    __metaclass__ = ABCMeta

    def __init__(self, x, P, Q, R, N):
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R
        self.Identity = np.eye(len(x))
        self.N = N

    @abstractmethod
    def state_transition_function(self, u):
        pass

    @abstractmethod
    def state_transition_matrix(self, u):
        pass

    @abstractmethod
    def observation_function(self, u):
        pass

    @abstractmethod
    def observation_matrix(self, u):
        pass

    def __call__(self, dt, u, measurements=None):
        """ Step function.

        If no new measurements are passed,
        the function only run the predict step.

        :param dt: sampling time [s]
        :param u: argument of transition matrix
        :param measurements: input of the update step
        """
        dt_integration = dt / self.N
        for k in range(self.N):
            self.x += self.state_transition_function(u) * dt_integration
            F = self.state_transition_matrix(u)
            self.P += dt_integration(F * self.P * F.T + self.Q)

        if measurements is not None:
            H = self.observation_matrix(u)
            PHT = self.P * H.T
            S = H * PHT + self.R
            K = PHT * np.linalg.inv(S)
            y = measurements - self.observation_function(u)
            self.x = self.x + K * y
            self.P = (self.Identity - K * H) * self.P
