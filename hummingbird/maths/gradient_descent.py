import numpy as np
from hummingbird.parameters.constants import StateEnum, ActuatorEnum


def gradient_descent(J, alpha, beta, phi, max_iters=5000, epsilon=1e-8, kappa=1e-6):
    """ Gradient descent algorithm

    """
    iters = 0
    J0 = np.inf
    while iters < max_iters:
        alpha_plus = alpha + epsilon
        beta_plus = beta + epsilon
        phi_plus = phi + epsilon
        J0 = J(alpha, beta, phi)
        dJ_dalpha = (J(alpha_plus, beta, phi) - J0) / epsilon
        dJ_dbeta = (J(alpha, beta_plus, phi) - J0) / epsilon
        dJ_dphi = (J(alpha, beta, phi_plus) - J0) / epsilon
        alpha = alpha - kappa * dJ_dalpha
        beta = beta - kappa * dJ_dbeta
        phi = phi - kappa * dJ_dphi
        iters += 1
        if iters % 100 == 0:
            print("J: {0} at iteration {1}".format(J0, iters))
    return alpha, beta, phi


def linearize(self, nominal_state, nominal_control_input, epsilon=1e-8):
    """ Linearize the model around a trimmed point.

    """
    A = np.zeros((StateEnum.size, StateEnum.size), dtype=np.double)
    B = np.zeros((StateEnum.size, ActuatorEnum.size), dtype=np.double)
    f_nominal = self.f(nominal_state, nominal_control_input)
    state_mask = np.zeros((StateEnum.size,), dtype=np.double)
    input_mask = np.zeros((ActuatorEnum.size,), dtype=np.double)
    for col in range(StateEnum.size):
        state_mask[col] = 1.0
        f_ = self.f(nominal_state + epsilon *
                    state_mask, nominal_control_input)
        A[:, col] = (f_ - f_nominal) / epsilon
        state_mask[col] = 0.0
    for col in range(ActuatorEnum.size):
        input_mask[col] = 1.0
        f_ = self.f(nominal_state, nominal_control_input +
                    epsilon * input_mask)
        B[:, col] = (f_ - f_nominal) / epsilon
        input_mask[col] = 0.0
    return A, B
