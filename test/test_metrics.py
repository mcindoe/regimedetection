import math
import numpy as np
import pytest
from pytest import approx

from regimedetection.src.metrics import euclidean_distance
from regimedetection.src.metrics import squared_euclidean_distance

def test_squared_euclidean_distance():
    a = [3, 4, 5]
    b = [10, 5, 0]

    assert squared_euclidean_distance(a, a) == 0
    assert squared_euclidean_distance(a, b) == 75

    assert squared_euclidean_distance(a, b) == squared_euclidean_distance(
        a, np.array(b)
    )
    assert squared_euclidean_distance(a, b) == squared_euclidean_distance(
        a, tuple(b)
    )

    with pytest.raises(ValueError):
        squared_euclidean_distance(a, str(a))

    with pytest.raises(ValueError):
        squared_euclidean_distance(a, [3, 4])


def test_euclidean_distance():
    a = [1, 2, 3, 4]
    b = [6, 12, 18, 24]

    assert euclidean_distance(a, b) == approx(
        math.sqrt(squared_euclidean_distance(a, b))
    )

