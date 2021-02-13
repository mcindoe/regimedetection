"""
Provides utilities specific to the Gaussian Clouds example
"""

import numpy as np


def make_gaussian_clouds(cluster_centres, points_per_cluster, std_dev):
    """
    Generates 2-dimensional points in R2 which are normally distributed
    about the cluster centres provided.

    Args:
    cluster_centres (iter): collection of 2-dimensional pairs (x,y)
        to be used as centres of the clusters
    points_per_cluster (int): number of points to generate about each
        cluster centre
    std_dev (float): standard deviation of the normal distribution used
        to generate the points around each cluster

    There will be points_per_cluster * len(cluster_centres) points
    returned.
    """
    covariance_matrix = np.diag(np.repeat(std_dev, 2))
    cluster_points = [
        np.random.multivariate_normal(
            mean=np.array([x, y]),
            cov=covariance_matrix,
            size=points_per_cluster,
        )
        for x, y in cluster_centres
    ]
    return np.concatenate(cluster_points)
