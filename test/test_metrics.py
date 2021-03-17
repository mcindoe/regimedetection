import math
import numpy as np
import pytest
from pytest import approx

from regimedetection.src.metrics import euclidean_distance
from regimedetection.src.metrics import kl_divergence
from regimedetection.src.metrics import squared_euclidean_distance


def test_squared_euclidean_distance():
    a = [3, 4, 5]
    b = [10, 5, 0]

    assert squared_euclidean_distance(a, a) == approx(0)
    assert squared_euclidean_distance(a, b) == approx(75)

    assert squared_euclidean_distance(a, b) == squared_euclidean_distance(a, np.array(b))
    assert squared_euclidean_distance(a, b) == squared_euclidean_distance(a, tuple(b))

    with pytest.raises(ValueError):
        squared_euclidean_distance(a, str(a))

    with pytest.raises(ValueError):
        squared_euclidean_distance(a, [3, 4])

    # Scalar input
    assert squared_euclidean_distance(0, 2) == approx(4)
    assert squared_euclidean_distance(-1, -1) == approx(0)
    assert squared_euclidean_distance(5.25, 3.75) == approx(2.25)


def test_euclidean_distance():
    a = [1, 2, 3, 4]
    b = [6, 12, 18, 24]

    assert euclidean_distance(a, b) == approx(math.sqrt(squared_euclidean_distance(a, b)))


def slow_kl_divergence(a: np.ndarray, b: np.ndarray) -> np.float:
    """
    Non-vectorized, slow approach to calculating KL-divergence.
    Used only to compare vectorized approach

    a and b should be one-dimensional numpy arrays
    """

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Expected one-dimensional np arrays in slow_kl_divergence")

    if a.shape != b.shape:
        raise ValueError("a and b must be vectors of the same length in slow_kl_divergence")

    ret = 0
    for a_entry, b_entry in zip(a, b):
        ret += a_entry * math.log(a_entry / b_entry)

    return ret


def test_kl_divergence():
    a = np.array([[5, 4, 6, 7, 7], [9, 1, 8, 2, 3], [5, 8, 8, 8, 6]])
    b = np.array([[5, 3, 1, 4, 8], [5, 6, 3, 5, 6], [3, 2, 4, 2, 6]])

    slow_computation_result = np.empty(shape=(len(a)))
    for row_idx, (a_row, b_row) in enumerate(zip(a, b)):
        slow_computation_result[row_idx] = slow_kl_divergence(a_row, b_row)

    vectorised_computation_result = kl_divergence(a, b)

    assert np.allclose(slow_computation_result, vectorised_computation_result)

    # Check that appropriate shapes are handled correctly
    non_tiled_computation = kl_divergence(a, b[0])
    tiled_computation = kl_divergence(a, np.tile(b[0], reps=(len(a), 1)))
    assert np.allclose(non_tiled_computation, tiled_computation)

    non_tiled_computation_2 = kl_divergence(a[0], b)
    tiled_computation_2 = kl_divergence(np.tile(a[0], reps=(len(b), 1)), b)
    assert np.allclose(non_tiled_computation_2, tiled_computation_2)

    # Check that non-appropriate shapes are handled correctly
    with pytest.raises(ValueError):
        kl_divergence(a, b[:2])

    with pytest.raises(ValueError):
        kl_divergence(a[:2], b)
