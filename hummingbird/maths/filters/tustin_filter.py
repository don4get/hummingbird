from warnings import warn

import numpy as np


class TustinFilter(object):
    def __init__(self, dt, tau, initial_value):
        self.dt = dt
        self.tau = tau
        self.last_value = initial_value
        self.filter_value_derivative = 0.

    def __call__(self, value):
        if np.isclose(2 * self.tau + self.dt, 0., 1e-10):
            warn("[Tustin filter] cannot divide by zero")

        self.filter_value_derivative = \
            (2 * self.tau - self.dt) / (2 * self.tau + self.dt) * self.filter_value_derivative + \
            2 / (2 * self.tau + self.dt) * (value - self.last_value)

        return self.filter_value_derivative
