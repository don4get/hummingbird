import numpy as np


def compute_sigmoid(alpha_0, alpha, M):
    c1 = np.exp(-M * (alpha - alpha_0))
    c2 = np.exp(M * (alpha + alpha_0))
    sigmoid_alpha = (1 + c1 + c2) / ((1 + c1) * (1 + c2))
    return sigmoid_alpha


def rotate(points, rotation_matrix):
    return points * rotation_matrix
