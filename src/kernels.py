from math import exp
import numpy as np

from metrics import euclid_distance
from metrics import euclid_squared


def make_laplacian_kernel(sigma):
    def k(x, y):
        if x == y:
            return 1

        squared_distance = euclid_squared(x, y)
        power = -sigma*sigma*squared_distance
        return exp(power)

    return k


def gaussian_kernel(x, y):
    if x == y:
        return 1

    dot_product = np.dot(x, y)
    return exp(dot_product)

