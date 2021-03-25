"""
Implements the Azran-Ghahramani clustering algorithm outlined in the
paper "A New Approach to Data Driven Clustering", which we refer to as
"the paper".
"""

import itertools
import numpy as np
import random

from regimedetection.src.metrics import kl_divergence

from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple


def get_space_distances(points: np.ndarray, metric: Callable[[Any, Any], np.float]) -> np.ndarray:
    """
    Compute all non-trivial distances between all points in a metric
    space. I.e. all distances except those of the form d(x,x) which is 0
    by definition

    Args:
    points: array of points in the metric space. These can be of any type
        as long as the metric is suitably defined to expect this type
    metric: a distance function between points in the space. Expected to
        take two points of the same type as in the points array and return
        a float

    Returns:
        A one-dimensional array of all distances between all non-trivial
        combinations of points in the array. Order of the distances is
        inherited from the order returned by itertools.combinations(points, 2),
        i.e. of the form [(x1, x2), (x1, x3), ..., (x2, x3), ...]

    Length of the returned array is n(n-1)/2 where n = len(points)
    """

    return np.fromiter(
        (metric(*comb) for comb in itertools.combinations(points, 2)),
        dtype=np.float,
        count=len(points) * (len(points) - 1) // 2,
    )


def get_space_similarities(
    points: np.ndarray,
    metric: Callable[[Any, Any], np.float],
    similarity: Callable[[np.float], np.float],
) -> np.ndarray:
    """
    Compute all similarities between all points in a metric space apart
    from the self-similarities

    This function provides access to a the array of similarities without
    needing to compute the distances array in memory if not separately required
    or already computed

    Args:
    points: array of points in the metric space. These can be of any type
        as long as the metric is suitably defined to expect this type
    metric: a distance function between points in the space. Expected to
        take two points of the same type as in the points array and return
        a float
    similarity: a unary function from np.float to np.float. Expected to be
        a monotonically decreasing function such that larger distances
        correspond to smaller similarities

    Returns:
        A one-dimensional array of all similarities between all non-trivial
        combinations of points in the array. Order of the distances is inherited
        from the order returned by itertools.combinations(points, 2), i.e. of the
        form [(x1, x2), (x1, x3), ..., (x2, x3), ...].

    Length of the returned array is n(n-1)/2 where n = len(points)
    """

    return np.fromiter(
        (similarity(metric(*comb)) for comb in itertools.combinations(points, 2)),
        dtype=np.float,
        count=len(points) * (len(points) - 1) // 2,
    )


def get_similarities_matrix(
    points: np.ndarray,
    metric: Callable[[Any, Any], np.float],
    similarity: Callable[[np.float], np.float],
    distances: Optional[np.ndarray] = None,
    self_similarity_multiplier: Optional[float] = 0.1,
) -> np.array:
    """
    Compute the matrix of similarities between points in a metric space.

    Args:
    points: array of points in the metric space. These can be of any type
        as long as the metric is suitably defined to expect this type
    metric: a distance function between points in the space. Expected to
        take two points of the same type as in the points array and return
        a float
    similarity: a unary function from np.float to np.float. Expected to be
        a monotonically decreasing function such that larger distances
        correspond to smaller similarities
    distances: allows the specification of the distances between
        points in the space. Expected to be a np.ndarray of length n(n-1)/2
        where n = len(points), the form returned by get_space_distances
    self_similarity_multiplier: the similarity of a particle with itself is
        set to self_similarity_multplier * min(similarities). Smaller values
        encourage exploration away from self.

    Returns:
        2-dimensional np.ndarray which has (i,j)-entry equal to the similarity
        of the distance between point i and point j.
    """

    if distances is None:
        similarities = get_space_similarities(points, metric, similarity)

    else:
        similarities = np.vectorize(similarity)(distances)

    # the similarity of a particle with itself is set to a multiple of
    # the smallest similarity observed, specified by self_similarity_multiplier
    self_similarity = min(similarities) * self_similarity_multiplier

    similarities_matrix = np.zeros(shape=(len(points), len(points)))

    upper_indices = np.triu_indices(len(points), 1)
    diag_indices = np.diag_indices(len(points))

    similarities_matrix[upper_indices] = similarities
    similarities_matrix += similarities_matrix.T
    similarities_matrix[diag_indices] = self_similarity

    # W is assumed to be full rank
    if np.linalg.matrix_rank(similarities_matrix) != len(points):
        print("WARN: Similarities matrix is not full rank")

    return similarities_matrix


def get_transition_matrix_from_similarities_matrix(
    similarities_matrix: np.array,
) -> np.array:
    """
    Compute a transition matrix from a matrix of similarities.

    Divide each row of the similarities matrix by the row sum to give
    a transition matrix with each row representing a probability
    distribution.

    Implements equation (2) of the paper.

    Args:
    similarities_matrix: a 2-dimensional np.ndarray which has (i,j)-entry
        equal to the similarity of the distance between point i and point j.
        Of the form returned by get_similarities_matrix

    Returns:
        A 2-dimensional np.ndarray which has (i,j)-entry equal to the
        transition probability from point X_i to X_j
    """

    row_sums = similarities_matrix.sum(axis=1)
    transition_matrix = similarities_matrix / row_sums[:, np.newaxis]
    return transition_matrix


def k_prototypes(transition_matrix: np.array, prototypes_init: np.array):
    """
    Finds the K-clustering of the points evolving under the specified
    transition matrix, with initial prototypes prototypes_init. K is
    inferred from the length of the prototypes_init parameter

    Args:
    transition_matrix: a 2-dimensional np.ndarray which has (i,j)-entry
        equal to the probability of transitioning from particle X_i to
        particle X_j in the next time step
    prototypes_init: a list of lists. The ith list is the indexes of
        points which begin in partition i

    Returns:
        A list of lists representing the converged partition. The ith
        element contains indices of the points which belong to the ith
        partition

    Implements Algorithm 1 of the paper
    """

    n_points = len(transition_matrix)
    n_clusters = len(prototypes_init)

    prototypes = prototypes_init
    previous_partition = list()

    while True:
        partition = [[] for _ in range(n_clusters)]

        # Compute KL-divergence from each row to each cluster centre
        divergences = kl_divergence(
            np.repeat(transition_matrix, len(prototypes), axis=0),
            np.tile(prototypes, reps=(len(transition_matrix), 1)),
        )
        divergences = np.split(divergences, len(transition_matrix))
        closest_clusters = np.argmin(divergences, axis=1)

        new_prototypes = np.empty((n_clusters, n_points))
        for k in range(n_clusters):
            partition[k] = list(np.where(closest_clusters == k)[0])

            if partition[k]:
                new_prototypes[k] = np.average(transition_matrix[partition[k]], axis=0)

        # If we have an empty cluster in new_prototypes, perform star-shaped initialisation
        # and overwrite corresponding prototype
        if any(len(cluster) == 0 for cluster in partition):
            empty_cluster_indices = tuple(
                idx for idx, cluster in enumerate(partition) if len(cluster) == 0
            )

            non_empty_cluster_indices = list(
                set(range(n_clusters)).difference(set(empty_cluster_indices))
            )

            existing_prototypes = new_prototypes[non_empty_cluster_indices]

            for cluster_idx in empty_cluster_indices:
                # Find the transition matrix prototype which is furthest away (wrt KL-divergence)
                # from all currently-used prototypes
                divergences = kl_divergence(
                    np.repeat(transition_matrix, len(existing_prototypes), axis=0),
                    np.tile(existing_prototypes, reps=(len(transition_matrix), 1)),
                )
                divergences = np.array(np.split(divergences, len(transition_matrix)))
                min_divergences = divergences.min(axis=1)

                # Choose the row from the transition matrix with maximal min-divergence
                chosen_prototype_idx = np.argmax(min_divergences)

                chosen_prototype = transition_matrix[chosen_prototype_idx]
                new_prototypes[cluster_idx] = chosen_prototype

                # Update partition to include the corresponding point as a singleton cluster
                for cluster in partition:
                    if chosen_prototype_idx in cluster:
                        cluster.remove(chosen_prototype_idx)

                partition[cluster_idx] = [chosen_prototype_idx]

                # If there is another iteration to perform, update the existing_prototypes arr
                if cluster_idx != empty_cluster_indices[-1]:
                    existing_prototypes = np.append(
                        existing_prototypes, chosen_prototype.reshape(1, -1), axis=0
                    )

        # If no empty clusters and the cluster has not changed, steady-state has been reached
        elif sorted(previous_partition) == sorted(partition):
            return partition

        previous_partition = partition
        prototypes = new_prototypes


def star_shaped_init(transition_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Compute the "star-shaped" prototypes initialisation with n_clusters
    clusters, where the points evolve according to transition_matrix.

    We take the mean of the transition matrix rows for the first prototype.
    The remaining rows are chosen from the transition matrix so as to maximise
    the minimum KL-divergence to the already-computed prototypes.

    Implements Algorithm 2 of the paper
    """

    n_points = len(transition_matrix)
    prototypes = np.empty((n_clusters, n_points))
    prototypes[0] = np.average([transition_matrix[n] for n in range(n_points)], axis=1)

    # Record which transition-matrix rows have been used as a prototype
    # already so as to avoid unncessary KL-divergence computation
    row_numbers_used = set()

    for k in range(1, n_clusters):
        min_divergences = np.empty(shape=(n_points))

        for n in range(n_points):
            if n in row_numbers_used:
                min_divergences[n] = 0
            else:
                previous_prototypes = transition_matrix[:k]
                divergences = kl_divergence(transition_matrix[n], previous_prototypes)
                min_divergences[n] = divergences.min()

        z = np.argmax(min_divergences)
        prototypes[k] = transition_matrix[z]
        row_numbers_used.add(z)

    return prototypes


def random_init(transition_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Compute a random initialisation of the n_clusters prototypes
    from the transition_matrix
    """

    n_points = len(transition_matrix)
    prototypes = np.empty((n_clusters, n_points))
    chosen_rows = random.sample(range(len(transition_matrix)), n_clusters)

    for k in range(n_clusters):
        prototypes[k] = transition_matrix[chosen_rows[k]]

    return prototypes


def delta_k_t(k_values: np.ndarray, t_values: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """
    For each passed k, compute the t-th order k-eigengap.
    Implements a vectorised version of Equation (11) of the paper.

    Returns:
        A two-dimensional numpy array with (i,j) entry equal to
        delta_k(t), where:
            k = k_values[i],
            t = t_values[j]
    """

    if isinstance(k_values, list):
        k_values = np.array(k_values)
    if isinstance(t_values, list):
        t_values = np.array(t_values)
    if isinstance(eigenvalues, list):
        eigenvalues = np.array(eigenvalues)

    if np.any(k_values < 0):
        raise ValueError("Input k_values to delta_k_t must be positive")

    if np.any(k_values >= len(eigenvalues) - 1):
        raise ValueError(f"Input k_values to delta_k_t must be <= {len(eigenvalues)-1}")

    abs_evalues = abs(eigenvalues).reshape(-1, 1)

    # k eigenvalues are at indexes 0, ..., k-1, so in 0-index convention,
    # equation (11) reads abs(lambda_k-1)**t - abs(lambda_k)**t
    lower_evalues = abs_evalues[k_values - 1]
    upper_evalues = abs_evalues[k_values]

    lower_evalues_matrix = np.tile(lower_evalues, (1, len(t_values)))
    upper_evalues_matrix = np.tile(upper_evalues, (1, len(t_values)))

    lower_power_matrix = lower_evalues_matrix ** t_values
    upper_power_matrix = upper_evalues_matrix ** t_values

    return lower_power_matrix - upper_power_matrix


def multiscale_k_prototypes(
    transition_matrix: np.ndarray,
    max_steps_power: np.ndarray,
    max_clusters: Optional[np.int] = None,
) -> List[Tuple]:
    """
    Suggest good partitions of the dataset, given the transition matrix
    P of equation (2) of the paper.

    Implements Algorithm 3 of the paper.

    Args:
    transition_matrix: N*N matrix with (i,j) entry equal to the
        probability of moving from point i to point j in one
        time step. As returned by get_transition_matrix_from_similarities_matrix
    max_steps_power: The upper bound on the number of steps to consider in order
        to find good partitions
    max_clusters: An optional upper bound of the number of clusters in the
        returned partitions

    Returns:
        List of tuples. Each tuple is of the form (partition, suitability),
        where partition is a list of lists containing the indices for each
        cluster, and suitability is the associated eigengap separation value
    """

    if max_clusters is None:
        max_clusters = len(transition_matrix) - 2

    eigenvalues = np.linalg.eig(transition_matrix)[0]
    sorting_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorting_indices]

    k_values = np.arange(1, max_clusters + 1)
    t_values = np.logspace(start=0, stop=max_steps_power, num=1000)

    delta_k_t_values = delta_k_t(k_values, t_values, eigenvalues)
    steps_to_best_reveal_indices = delta_k_t_values.argmax(axis=1)

    considered_n_clusters = []

    for k_idx, steps_to_best_reveal_k_idx in enumerate(steps_to_best_reveal_indices):
        # If k is the cluster best revealed by this number of steps
        index_best_revealed = delta_k_t_values[:, steps_to_best_reveal_k_idx].argmax(axis=0)

        if index_best_revealed == k_idx:
            k = k_values[k_idx]
            steps_to_best_reveal_k = np.int(t_values[steps_to_best_reveal_k_idx])
            partition_suitability = delta_k_t_values[k_idx, steps_to_best_reveal_k_idx]

            considered_n_clusters.append((k, steps_to_best_reveal_k, partition_suitability))

    suggested_clusters = []

    for n_clusters, n_steps, partition_suitability in considered_n_clusters:
        if n_clusters == 1:
            continue

        transition_matrix_power = np.linalg.matrix_power(transition_matrix, n_steps)

        Q_init = star_shaped_init(transition_matrix_power, n_clusters)
        partition = k_prototypes(transition_matrix_power, Q_init)

        suggested_clusters.append((partition, partition_suitability))

    return suggested_clusters
