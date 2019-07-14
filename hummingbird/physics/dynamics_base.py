#!/usr/bin/env python3
"""
########
Dynamics
########

Describe the second law of Newton and the way aerodynamics act on the drone.
"""
from scipy.integrate import ode


class DynamicsBase(object):
    """Dynamics Base

    :param x0: Initial state vector.
    :param t0: Initial time.
    :param dt_integration: Physical configuration of the drone
    """

    _default_dt_integration = 1e-3

    def __init__(self, x0, t0, dt_integration=_default_dt_integration):
        self.integrator = None
        self.dt = dt_integration
        self.x0 = x0
        self.t0 = t0
        self.x = x0
        self.t = t0

    def set_integrator(self, ode_func, integrator, jac=None, **kwargs):
        """Init function

        :param ode_func: Ordinary Differential Equation
        :param integrator: Function used to integrate Ordinary
                           Differential Equations
        :param jac: Jacobian of the ODE
        :param kwargs: optional arguments used to integrate equations
        """
        self.integrator = ode(ode_func, jac).set_integrator(
            integrator, **kwargs)
        self.integrator.set_initial_value(self.x0, self.t0)

    def integrate(self, t1):
        """Step function

        Integrate the ODE from intern integrator time till given final
        time.

        :param double t1: Final time of the integration

        :raise Exception: Check init of the integrator
        """
        if self.integrator is None:
            raise Exception('Initialize integrator first using set_integrator')
        while self.integrator.successful() and self.integrator.t < t1:
            self.integrator.integrate(self.integrator.t + self.dt)
            self.x = self.integrator.y
            self.t = self.integrator.t


