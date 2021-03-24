import math
import numpy as np


NUMERIC_TYPES = (np.int, np.float, int, float)


def is_numeric(x):
    return isinstance(x, NUMERIC_TYPES)


def is_numpy(x):
    return type(x).__module__ == np.__name__


def squared_euclidean_distance(a, b):
    """Square of euclidean distance between two points in R^n, n >= 1"""
    # If a, b are scalars
    if is_numeric(a) and is_numeric(b):
        distance = (a - b) * (a - b)

    # Else a, b are lists / numpy arrays
    else:
        if len(a) != len(b):
            raise ValueError("Input lengths not equal in squared_euclidean_distance()")

        if is_numpy(a) and is_numpy(b):
            distance = ((a - b) * (a - b)).sum()
        else:
            distance = squared_euclidean_distance(np.array(a), np.array(b))

    return distance


def euclidean_distance(a, b):
    """Euclidean distance between two points in R^n, n >= 1"""
    return math.sqrt(squared_euclidean_distance(a, b))


def average_euclidean_distance(a, b):
    """
    Returns the average Euclidean distance between elements of a
    and elements of b

    Args:
    a, b (list of lists of numerics):
        represent collections of elements of R^n for some n
        need not be the same number of elements of R^n in a and b
    """

    total = 0
    for a_el in a:
        for b_el in b:
            total += euclidean_distance(a_el, b_el)

    n_combinations = len(a) * len(b)
    return total / n_combinations


def sum_of_all_kernel_vals(kernel, collection):
    """
    Computes the sum of kernel(x_i, x_j) for all x_i, x_j in the
    collection

    Args:
    kernel: function taking two points, returning a real number
    collection: collection of points
    """

    assert len(collection) > 0, "Require a non-empty collection in sum_kernels()"

    ret = 0

    # Compute the 'off-diagonal' sums. I.e. sum of k(x_i, x_j) for all j > i.
    for i in range(len(collection)):
        for j in range(i + 1, len(collection)):
            ret += kernel(collection[i], collection[j])

    # Double to add the sum of all k(x_i, x_j) for all j < i
    ret *= 2

    # Add elements on the diagonal to get all contributions
    ret += sum([kernel(x, x) for x in collection])

    return ret


def make_mmd_metric(kernel):
    def f(X, Y):
        sum_1 = sum_kernels(kernel, X)
        sum_3 = sum_kernels(kernel, Y)

        sum_2 = 0
        for x_i in X:
            for y_i in Y:
                sum_2 += kernel(x_i, y_i)

        sum_1 /= len(X) * len(X)
        sum_3 /= len(Y) * len(Y)
        sum_2 *= -2 / (len(X) * len(Y))

        total = sum_1 + sum_2 + sum_3
        if np.isclose(total, 0):
            return 0
        return math.sqrt(total)

    return f


def kl_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorised KL-divergence computations from vectors in b
    to vectors in a.

    a, b should be numpy arrays of the same shape. Returned value
    is a length-N 1-dimensional numpy array where N is the number
    of rows in the input arrays. The ith entry in the returned
    array is the KL-divergence from b[i] to a[i].
    """

    # If one of the inputs is one-dimensional, and the other is two-dimensional,
    # then tile the one-dimensional array into a 2-dimensional array of appropriate size
    if a.ndim == 1 and b.ndim == 2:
        a = np.tile(a, reps=(len(b), 1))

    elif a.ndim == 2 and b.ndim == 1:
        b = np.tile(b, reps=(len(a), 1))

    # Otherwise, if one-dimensional inputs are provided, reshape as 2-dimensional arrays
    if a.ndim == 1:
        a = a.reshape(1, -1)

    if b.ndim == 1:
        b = b.reshape(1, -1)

    # a and b should now be two-dimensional arrays of the same size if appropriate
    # inputs were provided
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            "Inputs to kl_divergence should be 1- or 2-dimensional arrays"
        )

    if a.shape != b.shape:
        raise ValueError("Inputs to kl_divergence to not have appropriate shapes")

    return (a * np.log(a / b)).sum(axis=1)
