#!/usr/bin/env python3
"""
###########################################
Proportional Integral Derivative controller
###########################################
"""

import numpy as np
from hummingbird.maths.filters.tustin_filter import TustinFilter


class PID(object):
    """Class describing a PID.

    This implementation considers:
    1. The Tustin bilinear transform to compute low pass filter of the derivative output.
    2. The antiwindup has a constraint on the absolute output of the PID.
        The output should not exceed the saturation.

    :param kp: Proportional gain
    :param ki: Integral gain
    :param kd: Derivative gain
    :param limit: The absolute saturation of the output of the controller
    :param Ts: The sampling time used for output computation
    :param tau: The ´time constant´ of the low pass filter used for derivative ouput computation

    TODO: Improve the antiwindup. This integrator should not require more than a certain percentage of the total saturation.

    """

    def __init__(self, kp, ki=0., kd=0., limit=np.inf, dt=0.01, tau=0.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.dt = dt
        self.filtered_error_derivative = 0.
        self.derivative_filter = TustinFilter(self.dt, tau, self.filtered_error_derivative)
        self.integrated_error = 0.
        self.last_error = 0.

    def compute_control_input(self, reference, feedback, feedback_derivative=None):
        """ Step function of the controller

        :param reference: command input, reference
        :param feedback: feedback
        :param feedback_derivative: optional feedback derivative.

        In absence of it, the controller numerically compute the derivative of the feedback.
        """
        error = reference - feedback
        # Used trapezoidal transform for integral output.
        self.integrated_error += 0.5 * self.dt * (error + self.last_error)
        if feedback_derivative is not None:
            # Use measured feedback derivative
            # FIXME: the derivative error in the difference between the reference derivative and
            #  the feedback derivative
            self.filtered_error_derivative = self.kd * feedback_derivative
        else:
            # By default, compute 1st order low pass filter thanks to Tustin transform.
            self.filtered_error_derivative = self.derivative_filter(error)
        self.last_error = error
        # Bound the ouput.
        output_unsaturated = self.kp * error + self.ki * \
                             self.integrated_error + self.kd * self.filtered_error_derivative
        output = np.sign(output_unsaturated) * np.min([np.abs(output_unsaturated), self.limit])
        # Integrator anti wind up: If the ouput exceeds the saturation, reduce the integrator
        # so that the ouput stay in the saturation interval.
        if self.ki != 0:
            self.integrated_error += self.dt / self.ki * (output - output_unsaturated)
        return output
