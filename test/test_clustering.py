import numpy as np

from regimedetection.src.clustering import get_space_distances
from regimedetection.src.clustering import get_space_similarities
from regimedetection.src.clustering import get_similarities_matrix
from regimedetection.src.similarities import make_gaussian_similarity
from regimedetection.src.metrics import euclidean_distance
from regimedetection.src.metrics import squared_euclidean_distance

# metrics used to test get_space_distances() and get_space_similarities()
tested_metrics = [euclidean_distance, squared_euclidean_distance]
tested_similarities = [make_gaussian_similarity(2)]
tested_points = np.array([[0, 1], [3, 3], [5, 2], [2.34, 4.32], [-1.44, 9.51]])


def test_get_space_distances():
    def get_expected_space_distances(points, metric):
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                first_point = points[i]
                second_point = points[j]

                distances.append(metric(first_point, second_point))

        return distances

    for metric in tested_metrics:
        expected = get_expected_space_distances(tested_points, metric)
        observed = get_space_distances(tested_points, metric)

        assert np.isclose(expected, observed).all()


def test_get_space_similarities():
    def get_expected_space_similarities(points, metric, similarity):
        distances = get_space_distances(points, metric)
        return [similarity(d) for d in distances]

    for metric in tested_metrics:
        for similarity in tested_similarities:
            expected = get_expected_space_similarities(
                tested_points, metric, similarity
            )
            observed = get_space_similarities(
                tested_points, metric, similarity
            )

            assert np.isclose(expected, observed).all()


def test_get_similarities_matrix():
    def get_expected_similarities_matrix(
        points, metric, similarity, self_similarity_multiplier
    ):
        n_points = len(points)
        similarities_matrix = np.empty((n_points, n_points))

        similarities = get_space_similarities(points, metric, similarity)
        self_similarity = min(similarities) * self_similarity_multiplier

        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    similarities_matrix[i, j] = self_similarity
                else:
                    similarities_matrix[i, j] = similarity(
                        metric(points[i], points[j])
                    )

        return similarities_matrix

    for metric in tested_metrics:
        for similarity in tested_similarities:
            for self_similarity_multiplier in [0.1, 0.5, 1]:
                expected = get_expected_similarities_matrix(
                    tested_points,
                    metric,
                    similarity,
                    self_similarity_multiplier,
                )
                observed = get_similarities_matrix(
                    tested_points,
                    metric,
                    similarity,
                    self_similarity_multiplier=self_similarity_multiplier,
                )

                assert np.isclose(expected, observed).all()
