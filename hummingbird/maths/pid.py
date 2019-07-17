"""
###########################################
Proportional Integral Derivative controller
###########################################
"""
from hummingbird.maths.filters.tustin_filter import Derivative
import numpy as np


class Pid:
    """Class describing a PID.

    This implementation considers:
    1. The Tustin bilinear transform to compute low pass filter of the derivative output.
    2. The antiwindup has a constraint on the absolute output of the PID.
        The output should not exceed the saturation.

    :param kp: Proportional gain
    :type kp: double, optional
    :param ki: Integral gain
    :type ki: double, optional
    :param kd: Derivative gain
    :type kd: double, optional
    :param limit: The absolute saturation of the output of the controller
    :type limit: double, optional
    :param dt: The sampling time used for output computation
    :type dt: double, optional
    :param sigma: The ´time constant´ of the low pass filter used for derivative ouput computation
    :type sigma: double, optional
    :param limit: The threshold to bound the maximum output (and also the minimum output in case lower_lim is None
    :type limit: double, optional
    :param lower_limit: The threshold to bound the minimum output
    :type lower_limit: double, optional
    """
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.01, sigma=0.05, limit=1.0, lower_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.upper_limit = limit
        self.lower_limit = lower_limit if lower_limit is not None else -limit
        self.integrator = 0.
        self.error_delay_1 = 0.
        self.y_dot = 0.
        self.sigma = sigma
        self.filter = Derivative(self.dt, self.sigma, self.y_dot)

    def __call__(self, y_ref, y, y_dot=None):
        """ Step function of the controller

                :param y_ref: command input, reference
                :param y: feedback
                :param y_dot: optional feedback derivative.

                In absence of it, the controller numerically compute the derivative of the feedback.
        """

        error = y_ref - y

        if self.ki != 0.:
            self._integrate_error(error)

        if y_dot is None:
            if self.kd != 0.:
                self.y_dot = self.filter(y)
        else:
            self.y_dot = y_dot

        u_unsat = self.kp * error + self.ki * self.integrator + self.kd * self.y_dot
        u_sat = np.clip(u_unsat, self.lower_limit, self.upper_limit)

        if self.ki != 0.:
            self._antiwindup(u_unsat, u_sat)

        return u_sat

    def _antiwindup(self, u_unsat, u_sat):
        # TODO: Improve the antiwindup. This integrator should not require
        #  more than a certain percentage of the total saturation.
        if self.ki != 0:
            self.integrator += self.dt / self.ki * (u_sat - u_unsat)

    def _integrate_error(self, e):
        # Use trapezoidal integration
        self.integrator += self.dt / 2. * (e + self.error_delay_1)
        self.error_delay_1 = e


