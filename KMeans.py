"""
James Ambat
CS-663
Assignment 2
"""
import math

from matplotlib.colors import ListedColormap

from cluster import cluster
from typing import Tuple
import random
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
                 Tuple[1] is a list of lists of the cluster centroid coordinates.
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
        closest_centroids = None

        if num_dimensions == 0:
            raise ValueError("Must provide instances with at least one dimension to cluster.")

        # Dimensions validated, assign random cluster positions using a min_value of -100 and max_value of 100 so that
        # clusters have reasonable space for dispersion when random values are chosen for each dimension.
        centroid_coordinate_map = {}
        for i in range(self.num_clusters):
            random_coordinates = [None] * num_dimensions
            random_coordinates = [random.randint(-1000, 1000) for _ in random_coordinates]

            centroid_coordinate_map[i] = random_coordinates

        # Initial centroid coordinates placed, begin convergence
        for i in range(self.max_iterations):
            # Get the closest centroids for each of the instances
            closest_centroids = self.get_closest_centroids(instances=x, centroid_coordinate_map=centroid_coordinate_map)

            # Update the centroids based on coordinates of the closest centroids
            self.update_centroids(instances=x,
                                  closest_centroids=closest_centroids,
                                  centroid_coordinate_map=centroid_coordinate_map)

        if closest_centroids is None:
            raise ValueError("There was an error calculating the closest centroids for each instance.")
        if len(centroid_coordinate_map) == 0:
            raise ValueError("There was an error acquiring the centroids.")

        # Return a list of centroid coordinate per assignment requirements.
        centroid_coordinates = [centroid_coordinate_map.get(centroid_num, None)
                                for centroid_num in centroid_coordinate_map]

        return closest_centroids, centroid_coordinates

    def get_closest_centroids(self, instances: list, centroid_coordinate_map: dict) -> list:
        """
        Helper method to map instances to centroids based on the min distance from instance to centroid.
        Distances are straight line distance in n-dimensions.
        :param instances: list instance of n_dimensions.
        :param centroid_coordinate_map: dictionary of the centroid to their coordinates.
        :return: list of each centroid number for each index of instances.
        """
        centroid_numbers_per_instance = [None] * len(instances)

        # Scan through each of the instances and calculate distance to each centroid
        for i in range(len(instances)):

            # Get the straight line distance to each of the centroids.
            instance = instances[i]
            min_distance = sys.maxsize
            min_centroid_num = 0
            for centroid_num in centroid_coordinate_map:
                centroid_coordinates = centroid_coordinate_map.get(centroid_num, None)
                distances_squared = [(instance[i] - centroid_coordinates[i]) ** 2 for i in range(len(instance))]

                distance = math.sqrt(sum(distances_squared))
                if distance < min_distance:
                    # track the min centroid and the distance
                    min_centroid_num = centroid_num
                    min_distance = distance

            # Record the centroid number for each instance
            centroid_numbers_per_instance[i] = min_centroid_num

        return centroid_numbers_per_instance

    def update_centroids(self, instances: list, closest_centroids: list, centroid_coordinate_map: dict):
        """
        Helper method to update the centroid_coordinate_map with coordinates of those instances that are closest to it.
        Uses the instance list and closest_centroids list to identify the closest instances for making updates.
        :param instances: list of instances and their features.
        :param closest_centroids: list of the closest centroid in relation to the indexed instance.
        :param centroid_coordinate_map: dictionary mapping centroids to their coordinates.
        :return: dictionary of an updated centroid map.
        """
        clustered_centroid_instance_map = {}

        for i in range(len(instances)):
            instance = instances[i]
            closest_centroid = closest_centroids[i]
            accumulated_coordinates = clustered_centroid_instance_map.get(closest_centroid, [])
            accumulated_coordinates.append(instance)
            clustered_centroid_instance_map[closest_centroid] = accumulated_coordinates

        # For each centroid, accumulate the dimensions and get each dimension's average.
        for centroid_num in clustered_centroid_instance_map:
            clustered_instances = clustered_centroid_instance_map.get(centroid_num, None)

            if clustered_instances is None:
                continue

            # accumulate_dimensions:      [  sum(x's),  sum(y's),   sum(z's),   ...  ]
            accumulated_dimensions = [0] * len(clustered_instances[0])
            for j in range(len(clustered_instances)):
                curr_instance = clustered_instances[j]

                for dimension in range(len(curr_instance)):
                    accumulated_dimensions[dimension] += curr_instance[dimension]

            # Dimensions accumulated, take the average of dimensions to get the new_centroid coordinates
            # average dimensions:         [  sum(x's) / |x|,  sum(y's) / |y|,   sum(z's) / |z|,   ...  ]

            num_in_cluster = len(clustered_instances)
            average_dimensions = [item / num_in_cluster for item in accumulated_dimensions]

            # Update the old averages with the new ones (if any)
            centroid_coordinate_map[centroid_num] = average_dimensions

    def plot_clustered_instances(self, instances: list, closest_centroids: list, scatter_plot_dot_size: int = 150,
                                 scatter_plot_title="Scatter Plot of Clustered Instances"):
        """
        Method to plot instances in their respective clusters.
        :param instances: list of instance data of n-dimensions.
        :param closest_centroids: list of closest centroids whose indexes are directly related to instance indexes.
        :param centroid_coordinates: list of coordinates for each centroid.
        :param scatter_plot_dot_size: int of the dot size for the scatter plot.
        :param scatter_plot_title: str of the title for the scatter plot.
        """

        if len(instances) <= 0:
            raise ValueError("Must provide an instances list with length >= 0")

        # Create a data frame to render a scatter plot from
        num_dimensions = len(instances[0])
        df_column_names = ["Feature %d" % (num + 1) for num in range(num_dimensions)]

        instances_data_frame = pd.DataFrame(instances)
        instances_data_frame.columns = df_column_names

        instances_data_frame["Closest Centroids"] = closest_centroids
        figures, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        sns.scatterplot(data=instances_data_frame, x="Feature 1", y="Feature 2",
                        s=scatter_plot_dot_size, hue="Closest Centroids", palette="Set1")

        axes.set_xlabel("Feature 1", fontweight="bold", fontsize=15)
        axes.set_ylabel("Feature 2", fontweight="bold", fontsize=15)

        plt.title(scatter_plot_title, fontweight="bold", fontsize=20)
        plt.legend(bbox_to_anchor=(1, 1), title="Cluster of Closest Centroid")
        plt.show()
        print()

