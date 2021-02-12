import math
import numpy as np


NUMERIC_TYPES = (np.int, np.float, int, float)


def is_numeric(x):
    return isinstance(x, NUMERIC_TYPES)


def is_numpy(x):
    return type(x).__module__ == np.__name__


def squared_euclidean_distance(a, b):
    '''Square of euclidean distance between two points in R^n, n >= 1'''
    # If a, b are scalars
    if is_numeric(a) and is_numeric(b):
        distances = (a-b)*(a-b)

    # Else a, b are lists / numpy arrays
    else:
        if len(a) != len(b):
            raise ValueError('Input lengths not equal in squared_euclidean_distance()')

        if is_numpy(a) and is_numpy(b):
            distance = ((a-b)*(a-b)).sum()
        else:
            distance = squared_euclidean_distance(np.array(a), np.array(b))

    return distance


def euclidean_distance(a, b):
    '''Euclidean distance between two points in R^n, n >= 1'''
    return math.sqrt(squared_euclidean_distance(a, b))


def average_euclidean_distance(a, b):
    '''
    Returns the average Euclidean distance between elements of a
    and elements of b

    Args:
    a, b (list of lists of numerics):
        represent collections of elements of R^n for some n
        need not be the same number of elements of R^n in a and b
    '''
    
    total = 0
    for a_el in a:
        for b_el in b:
            total += euclidean_distance(a_el, b_el)

    n_combinations = len(a) * len(b)
    return total / n_combinations


def sum_of_all_kernel_vals(kernel, collection):
    '''
    Computes the sum of kernel(x_i, x_j) for all x_i, x_j in the
    collection

    Args:
    kernel: function taking two points, returning a real number
    collection: collection of points
    '''

    assert len(collection) > 0, 'Require a non-empty collection in sum_kernels()'

    ret = 0
    
    # Compute the 'off-diagonal' sums. I.e. sum of k(x_i, x_j) for all j > i.
    for i in range(len(collection)):
        for j in range(i+1, len(collection)):
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

        sum_1 /= (len(X) * len(X))
        sum_3 /= (len(Y) * len(Y))
        sum_2 *= (-2 / (len(X) * len(Y)))

        total = sum_1 + sum_2 + sum_3
        if np.isclose(total, 0):
            return 0
        return math.sqrt(total)

    return f


def kl_divergence(a, b):
    '''Return the KL-divergence from b to a'''
    if len(a) != len(b):
        raise ValueError(f'Expected inputs to kl_divergence to be of the same length, '
            'got {len(a)} and {len(b)}')

    return (a * np.log(a/b)).sum()

