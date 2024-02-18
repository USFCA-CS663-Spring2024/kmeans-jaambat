"""
James Ambat
CS-663
Assignment 2
"""
from cluster import cluster
from typing import Tuple
import random


class KMeans(cluster):
    """
    Class for implementing KMeans clustering algorithm.
    """

    def __init__(self, k=5, max_iterations=100):
        """
        Class constructor for KMeans.
        :param k: int for the target number of cluster centroids.
        :param max_iterations: int max number of times to attempt convergence.
        """
        super().__init__()

        if k < 2:
            raise ValueError("Must provide k >= 2 for clustering at least 2 groups.")
        if max_iterations < 100:
            raise ValueError("Must provide at max_iterations >= 100.")

        self.num_clusters = k
        self.max_iterations = max_iterations

    def fit(self, x: list) -> Tuple[list, list]:
        """
        Fits data into the number of clusters as specified by self.num_clusters and the number of convergence
        attempts as specified by self.max_iterations.
        :param x: list of n instances with d number of features for each instance. This is a matrix.
        :return: Tuple[list, list].
                 Tuple[0] is the list of cluster hypotheses for each instance.
                 Tuple[1] is a list of lists of the cluster centroid values.
        """
        if len(x) <= 0 or type(x) is not list:
            raise ValueError("Must provide a non-empty list for x.")

        # I assume dimensions are consistent across all rows of the matrix. Inspect the first row to verify n_dimensions
        # [
        #     [dimension_1, dimension_2, dimension_3, ... ],
        #     [dimension_1, dimension_2, dimension_3, ... ],
        #                      ...
        # ]

        num_dimensions = len(x[0])

        if num_dimensions == 0:
            raise ValueError("Must provide instances with at least one dimension to cluster.")

        # Dimensions validated, assign random clusters positions using a min_value of -100 and max_value of 100 so that
        # clusters have reasonable space for dispersion when random values are chosen for each dimension.
        clusters_coordinates = {}
        for i in range(self.num_clusters):
            random_coordinates = [None] * num_dimensions
            random_coordinates = [random.randint(-100, 100) for _ in random_coordinates]

            clusters_coordinates[i] = random_coordinates

        return [], []




