'''
Implementation of the Azran-Ghahramani clustering as outlined in the paper
"A New Approach to Data Driven Clustering", which is referred to here as
"the paper".
'''

from math import ceil
from math import floor
import numpy as np
import pickle
import random


def get_space_distances(points, metric):
    '''
    Compute distances between all points in a metric space.

    Returned as a list of lists. If points contains n elements,
    the returned list is of length n-1, with the ith entry being
    a list of the distances from point i to points with indices
    {i+1, i+2, ..., n}
    '''
    n_points = len(points)
    distances = [
        [metric(points[i], points[j]) for j in range(i+1, n_points)]
        for i in range(n_points-1)
    ]
    return distances


def make_similarities_matrix(points, metric, similarity, distances=None):
    '''
    Returns a np.array which has (i,j)-entry equal to the similarity of the
    distance between point i and point j.

    If distances is passed, the computation is not repeated, or if not already
    computed this arg may be left as None.
    '''
    if distances is None:
        distances = get_space_distances(points, metric)

    W = np.empty((n_points, n_points))

    for i in range(n_points):
        for j in range(i, n_points):
            if i == j:
                W[i][j] = 1
            else:
                distance = distances[i][j-i-1]
                W[i][j] = similarity(distance)
                W[j][i] = W[i][j]

    # W is assumed to be full rank
    if np.linalg.matrix_rank(W) != len(W):
        print('WARN: Similarities matrix is not full rank')

    return W


def kl_divergence(a, b):
    '''Return the KL-divergence from b to a'''
    if len(a) != len(b):
        raise ValueError(f'Expected inputs to kl_divergence to be of the same length, '
            'got {len(a)} and {len(b)}')

    return (a * np.log(a/b)).sum()


def K_prototypes(transition_matrix, prototypes_init):
    '''
    Finds the K-clustering of the points evolving under the
    specified transition matrix, with initial prototypes
    prototypes_init.

    Algorithm 1 of the paper.
    '''

    n_rows, n_cols = transition_matrix.shape
    prototypes = prototypes_init
    previous_partition = None

    while True:
        partition = [[] for _ in range(len(prototypes))]

        for row in range(n_rows):
            divergences = [
                kl_divergence(
                    transition_matrix[row],
                    prototypes[k]
                )
                for k in range(n_clusters)
            ]
            closest_cluster = np.argmin(divergences)
            partition[closest_cluster].append(row)

        new_prototypes = np.empty((n_clusters, n_cols))
        for k in range(n_clusters):
            # NB: Trying out a new method here of keeping old row if no partition elements
            if partition[k]:
                new_prototypes[k] = np.average(
                    [transition_matrix[m] for m in partition[k]],
                    axis = 1
                )
            else:
                new_prototypes[k] = prototypes[k]

        if previous_partition is not None and set(previous_partition) == set(partition):
            return partition, prototypes

        previous_partition = partition
        prototypes = new_prototypes


def star_shaped_init(transition_matrix, n_clusters):
    '''
    Compute the "star-shaped" prototypes initialisation with
    n_clusters clusters, where the points evolve according
    to transition_matrix.

    Algorithm 2 of the paper. We take the mean of the transition
    matrix rows for the first prototype. The remaining rows
    are chosen from the transition matrix so as to maximise the
    minimum KL-divergence to the already-computed prototypes.
    '''

    n_points = len(transition_matrix)
    prototypes = np.empty((n_clusters, n_points))
    prototypes[0] = np.average(
        [transition_matrix[n] for n in range(n_points)],
        axis = 1
    )

    # Record which transition-matrix rows have been used as a prototype
    # already so as to avoid unncessary KL-divergence computation
    row_numbers_used = set() 
    
    for k in range(1, n_clusters):
        min_divergences = []

        for n in range(n_points):
            if n in row_numbers_used:
                min_divergences.append(0)
            else:
                divergences = [
                    kl_divergence(transition_matrix[n], prototypes[j])
                    for j in range(k)
                ]
                min_divergences.append(min(divergences))

        z = np.argmax(min_divergences)
        prototypes[k] = transition_matrix[z]
        row_numbers_used.add(z)

    return prototypes


def random_init(transition_matrix, n_clusters):
    '''
    Compute a random initialisation of the n_clusters prototypes
    from the transition_matrix
    '''

    n_points = len(transition_matrix)
    prototypes = np.empty((n_clusters, n_points))
    chosen_rows = random.sample(range(len(transition_matrix)), n_clusters)

    for k in range(n_clusters):
        prototypes[k] = transition_matrix[chosen_rows[k]]

    return prototypes


def closest_even_integer(x):
    '''
    Computes closest even integer to an input float or integer.
    If x is an integer, the output is x+1 if x is odd else x
    '''
    if not isinstance(x, (int, float)):
        raise ValueError(f'Expected {x} to be an integer or a float')

    if isinstance(x, int):
        if x%2 == 0:
            return x
        return x+1

    lower = floor(x)
    upper = ceil(x)

    # If the closest integer to x is smaller than x
    if abs(lower - x) < abs(upper - x):
        return lower if lower%2 == 0 else upper
    return upper if upper%2 == 0 else lower


def delta_k_t(k, t, eigenvalues):
    '''
    Compute the t-th order eigengap between the kth and (k+1)th eigenvalue.
    Equation 11 from the paper.
    '''
    if k < 0:
        raise ValueError('Input k to delta_k_t() must be a positive integer')
    if k >= len(eigenvalues)-1:
        raise ValueError('Received an input k value to delta_k_t() larger than number of eigenvalues')

    this_evalue = eigenvalues[k]
    next_evalue = eigenvalues[k+1]

    return abs(this_evalue)**t - abs(next_evalue)**t


def maximal_eigengap(t, eigenvalues):
    '''
    Compute the maximal t-th order eigengap.
    Equation 14 of the paper
    '''
    eigengaps = [
        delta_k_t(k, t, eigenvalues)
        for k in range(len(eigenvalues)-1)
    ]
    return max(eigengaps)


def n_clusters_best_revealed(t, eigenvalues, max_clusters=None):
    '''
    Find the number of clusters best revealed by t steps.
    Equation 15 of the paper.
    '''
    if max_clusters is None:
        max_clusters = len(eigenvalues)-1

    delta_k_values = [
        delta_k_t(k, t, eigenvalues)
        for k in range(max_clusters)
    ]
    return np.argmax(delta_k_values) + 1


