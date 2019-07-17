from warnings import warn
import numpy as np


class Derivative(object):
    def __init__(self, dt, tau, initial_value):
        self.dt = dt
        self.tau = tau
        self.last_value = initial_value
        self.filter_value_derivative = 0.
        self._update_gains(dt)

    def _update_gains(self, dt):
        if np.isclose(2 * self.tau + self.dt, 0., 1e-10):
            warn("[Tustin filter] cannot divide by zero")
            self.a1 = 0.
            self.a2 = 1.
        else:
            self.a1 = (2. * self.tau - dt) / (2. * self.tau + self.dt)
            self.a2 = 2. / (2. * self.tau + self.dt)

    def __call__(self, value):
        self.filter_value_derivative = self.a1 * self.filter_value_derivative + self.a2 * (value - self.last_value)
        return self.filter_value_derivative
