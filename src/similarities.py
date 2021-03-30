"""
Provide similarity metrics as described in the paper
'A New Approach to Data-Driven Clustering'

Functions provided here should be non-negative, monotonically
decreasing functions
"""

import math
import numpy as np

from typing import Callable


def make_gaussian_similarity(sigma: np.float) -> Callable[[np.float], np.float]:
    """
    Produces the gaussian similarity function corresponding to the
    input sigma value.

    Used to create a function of one argument, as expected by the
    multiscale-k-prototypes algorithm in the clustering module,
    from a specified sigma
    """

    sigma_squared = sigma * sigma

    def gaussian_similarity(x):
        return math.exp(-x / sigma_squared)

    return gaussian_similarity


def make_gaussian_similarity_from_percentile(
    distances: np.ndarray, percentile: np.float
) -> Callable[[np.float], np.float]:
    """
    Produce the gaussian similarity function when sigma is chosen
    to be the specified percentile of the distances

    Percentile must be a value in [0, 100]
    """

    sigma = np.percentile(distances, percentile)
    return make_gaussian_similarity(sigma)
