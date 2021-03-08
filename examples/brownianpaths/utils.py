"""
Provides utilities specific to the Browian Paths example
"""

from typing import Optional

import numpy as np


def simulate_wiener_process(time_steps: np.ndarray, n_paths: int) -> np.ndarray:
    """
    Simulate `n_paths` paths of the Wiener process at time
    steps as provided in the `time_steps` array

    Returns:
    A np.ndarray of shape (n, len(time_steps)) with each row
    representing a realisation of the path
    """

    time_differences = np.ediff1d(time_steps)
    normal_noise = np.random.normal(size=(n_paths, len(time_differences))) * time_differences
    wiener_paths = np.insert(np.cumsum(normal_noise, axis=1), 0, 0, axis=1)

    return wiener_paths


def simulate_gbm_process(
    time_steps: np.ndarray,
    n_paths: int,
    mu: float,
    sigma: float,
    initial_value: Optional[float] = 1,
) -> np.ndarray:
    """
    Simulate `n_paths` of a Stochastic process driven by the
    Geometric Brownian Motion dynamics with return `mu` and
    volatility `sigma`, at the time steps as provided in the
    `time_steps` array. Optionally specify a starting value
    which is taken to be one by default.

    Returns:
    An np.ndarray of shape (n, len(time_steps)) with each row
    representing a relisation of the process's path at the
    specified nodes
    """

    drift_component = (mu - 0.5*sigma*sigma) * np.tile(time_steps, reps=(n_paths, 1))
    wiener_paths = simulate_wiener_process(time_steps, n_paths) 
    diffusion_component = sigma * wiener_paths
    gbm_paths = initial_value * np.exp(drift_component + diffusion_component)

    return gbm_paths
