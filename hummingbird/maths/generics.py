import numpy as np
from warnings import warn


def compute_sigmoid(alpha_0, alpha, M):
    c1 = np.exp(-M * (alpha - alpha_0))
    c2 = np.exp(M * (alpha + alpha_0))
    sigmoid_alpha = (1 + c1 + c2) / ((1 + c1) * (1 + c2))
    return sigmoid_alpha


def rotate(points, rotation_matrix):
    return points * rotation_matrix


def normalize_vector(vect):
    norm = np.linalg.norm(vect)
    if np.isclose(norm, 0):
        warn("Impossible to normalize vector")
        return vect
    return vect / np.linalg.norm(vect)