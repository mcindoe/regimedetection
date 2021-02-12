import numpy as np

def make_gaussian_clouds(cluster_centres, points_per_cluster, std_dev):
    '''
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
    '''
    covariance_matrix = np.diag(np.repeat(std_dev, 2))
    cluster_points = [
        np.random.multivariate_normal(
            mean = np.array([x,y]),
            cov = covariance_matrix,
            size = points_per_cluster
        )
        for x, y in cluster_centres
    ]
    return np.concatenate(cluster_points)


if __name__ == '__main__':
    from src.clustering import K_prototypes
    from src.clustering import make_similarities_matrix
    from src.clustering import make_transition_matrix
    from src.clustering import star_shaped_init
    from src.clustering import random_init
    from src.metrics import squared_euclidean_distance
    from src.metrics import euclidean_distance
    from src.similarities import inverse_squared

    cluster_centres = [
        (0,0),
        # (0,3),
        # (3,0),
        (3,3),
    ]

    points = make_gaussian_clouds(
        cluster_centres,
        points_per_cluster = 2,
        std_dev = 0.1
    )
    similarities_matrix = make_similarities_matrix(
        points,
        metric = squared_euclidean_distance,
        similarity = inverse_squared
    )
    transition_matrix = make_transition_matrix(similarities_matrix)
    
    prototypes_init = star_shaped_init(transition_matrix, n_clusters=len(cluster_centres))
    partition, prototypes = K_prototypes(transition_matrix, prototypes_init)

    print('partition:')
    print(partition)

    print('prototypes:')
    print(prototypes)
