from math import exp
from typing import Callable
from typing import Union

import numpy as np

from regimedetection.src.metrics import euclidean_distance
from regimedetection.src.metrics import squared_euclidean_distance


def make_rbf_kernel(sigma: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a vectorized implementation of the Radial Basis Function kernel,
    for a given value of sigma.

    Values passed to the returned kernel should be two-dimensional arrays,
    with the first axis being the batch dimension, and the second axis being
    the channels direction

    See e.g. https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    for a definition of the RBF kernel
    """

    def rbf_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or y.ndim != 2:
            raise np.AxisError("Inputs to RBF Kernel should have dimension 2")

        if x.shape != y.shape:
            raise ValueError("Inputs to RBF Kernel should have the same shape")

        differences = x - y
        squared_distances = (differences * differences).sum(axis=1)
        powers = -squared_distances / (2 * sigma * sigma)
        return np.exp(powers)

    return rbf_kernel


def _get_kernel_sum(
    x: np.ndarray, y: np.ndarray, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.float:
    """
    Compute the kernel of every element of x with every
    element of y, and take the sum

    Helper function for make_mmd_estimate_metric_from_kernel;
    not expected to be called directly by users.

    The kernel is expected to be a vectorized implementation,
    acting on two 2-dimensional np arrays
    """

    lhs = np.tile(x, reps=(len(y), 1))
    rhs = np.repeat(y, len(x), axis=0)
    kernel_values = kernel(lhs, rhs)

    return kernel_values.sum()


def make_mmd_estimate_metric_from_kernel(
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns the 'MMD Estimate Metric' from a specified kernel.

    The kernel provided is expected to be a vectorized implementation,
    acting on two 2-dimensional np arrays
    """

    def mmd_estimate_metric(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_kernel_sum = _get_kernel_sum(x, x, kernel)
        scaled_x_kernel_sum = x_kernel_sum / (len(x) * len(x))

        y_kernel_sum = _get_kernel_sum(y, y, kernel)
        scaled_y_kernel_sum = y_kernel_sum / (len(y) * len(y))

        cross_kernel_sum = _get_kernel_sum(x, y, kernel)
        scaled_cross_kernel_sum = (-2 / (len(x) * len(y))) * cross_kernel_sum

        return scaled_x_kernel_sum + scaled_y_kernel_sum + scaled_cross_kernel_sum

    return mmd_estimate_metric


# TODO: Do I want the Gaussian Kernel?
# TODO: Add documentation in the above


def gaussian_kernel(x: np.ndarray, y: np.ndarray) -> np.float:
    if x == y:
        return 1

    dot_product = np.dot(x, y)
    return exp(dot_product)
