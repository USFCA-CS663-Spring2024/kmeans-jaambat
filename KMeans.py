"""
James Ambat
CS-663
Assignment 2
"""
import math

from cluster import cluster
from typing import Tuple

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import random
import seaborn as sns
import sys


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

    def fit(self, x: list, x_range: tuple = None, y_range: tuple = None) -> Tuple[list, list]:
        """
        Fits data into the number of clusters as specified by self.num_clusters and the number of convergence
        attempts as specified by self.max_iterations.
        :param x: list of n instances with d number of features for each instance. This is a matrix.
        :param x_range: tuple of the centroid's x range (min, max). Randomized centroid positions could possibly be
                        dispersed too far given the range of the data. Selecting centroids within a reasonable range
                        initialize centroids within reasonable ranges of the actual data.
        :param y_range: tuple of the centroid's x range (min, max). Randomized centroid positions could possibly be
                        dispersed too far given the range of the data. Selecting centroids within a reasonable range
                        initialize centroids within reasonable ranges of the actual data.
        :return: Tuple[list, list].
                 Tuple[0] is the list of cluster hypotheses for each instance.
                 Tuple[1] is a list of lists of the cluster centroid coordinates.
        """
        if type(x) is numpy.ndarray:
            x = x.tolist()

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

        centroid_coordinate_map = self.get_randomly_dispersed_centroids(num_dimensions=num_dimensions,
                                                                        x_range=x_range,
                                                                        y_range=y_range)
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

    def get_randomly_dispersed_centroids(self, num_dimensions: int, x_range: tuple, y_range: tuple) -> dict:
        """
        Helper method to get randomly dispersed centroids. Assign random cluster positions using a min_value and
        max_value so that clusters have reasonable space for dispersion when random values are chosen for each dimension
        Additionally, the centroids are selecting with minimum dispersion of 0.6 standard deviations to ensure adequate
        dispersion of randomly selected centroids. Dispersion thresholds are incrementally adjusted if the 0.6 threshold
        is too high of a value.
        :return: dictionary of initially dispersed cluster centroid coordinates.
        """
        # Dimensions validated,
        good_dispersion = False
        std_tolerance = 0.65
        centroid_coordinate_map = None

        while not good_dispersion:
            centroid_coordinate_map = {}
            for i in range(self.num_clusters):
                random_coord = [0] * num_dimensions

                x_min_range = -1000
                x_max_range = 1000

                y_min_range = -1000
                y_max_range = 1000

                if x_range:
                    x_min_range = x_range[0]
                    x_max_range = x_range[1]

                if y_range:
                    y_min_range = x_range[0]
                    y_max_range = x_range[1]

                # random_coordinates: [x, y, z, ... ]
                random_coord[0] = random.randint(x_min_range, x_max_range)
                random_coord[1] = random.randint(y_min_range, y_max_range)

                # Store a random coordinate per centroid number
                centroid_coordinate_map[i] = random_coord

            # Check the dispersion of the points to ensure they are not too close. If so, select other random centroids.
            centroid_coordinates = [centroid_coordinate_map.get(centroid_num, None)
                                    for centroid_num in centroid_coordinate_map]
            coordinates_data_frame = pd.DataFrame(centroid_coordinates)
            less_dispersion_counter = 0
            for i in range(num_dimensions):
                # normalize the coordinates + get std to determine dispersion
                coordinates_dimension_i = coordinates_data_frame.iloc[:, i]
                coordinates_max = coordinates_dimension_i.max()
                coordinates_min = coordinates_dimension_i.min()
                coordinates_delta = coordinates_max - coordinates_min
                dimension_i_normalized = (coordinates_dimension_i - coordinates_min) / coordinates_delta
                dimension_standard_deviation = dimension_i_normalized.std()

                # Make sure the dimensions are not too far.
                if dimension_standard_deviation < std_tolerance:
                    less_dispersion_counter += 1

            if less_dispersion_counter == 0:
                good_dispersion = True
            else:
                # Bring-in the standard deviation tolerance
                std_tolerance -= 0.001

        if centroid_coordinate_map is None:
            raise ValueError("There was an error creating the centroid_coordinate_map.")

        return centroid_coordinate_map

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

    def plot_clustered_instances(self, instances: list,
                                 closest_centroids: list,
                                 scatter_plot_dot_size: int = 150,
                                 scatter_plot_title="Scatter Plot of Clustered Instances",
                                 compare_against_make_blob: bool = False,
                                 make_blobs_cluster_assignments: list = None,
                                 figure_file_name=None):
        """
        Method to plot instances in their respective clusters.
        :param instances: list of instance data of n-dimensions.
        :param closest_centroids: list of closest centroids whose indexes are directly related to instance indexes.
        :param scatter_plot_dot_size: int of the dot size for the scatter plot.
        :param scatter_plot_title: str of the title for the scatter plot.
        :param compare_against_make_blob: bool to indicate if comparing against cluster assignments.
                                                    This option will create a subplot for the comparison.
        :param make_blobs_cluster_assignments: list of scikit-learn's cluster assignments as compared to my
                                               closest_centroids list.
        :param figure_file_name: str of the figure file name.
        """
        if type(instances) is numpy.ndarray:
            instances = instances.tolist()

        if len(instances) <= 0:
            raise ValueError("Must provide an instances list with length >= 0")

        # Create a data frame to render a scatter plot from
        num_dimensions = len(instances[0])
        df_column_names = ["Feature %d" % (num + 1) for num in range(num_dimensions)]

        instances_data_frame = pd.DataFrame(instances)
        instances_data_frame.columns = df_column_names

        instances_data_frame["Closest Centroids"] = closest_centroids

        if compare_against_make_blob:
            # Plotting against the cluster_assignments, make 1 x 2 subplots.
            figures, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            # First plot the KMeans implementation.
            sns.scatterplot(data=instances_data_frame, x="Feature 1", y="Feature 2",
                            s=scatter_plot_dot_size, hue="Closest Centroids", palette="Set1",
                            ax=axes[0])

            axes[0].set_title("KMeans.fit() Performance", fontweight="bold", fontsize=15)
            axes[0].set_xlabel("x's", fontweight="bold", fontsize=15)
            axes[0].set_ylabel("y's", fontweight="bold", fontsize=15)

            # Plot the cluster_assignments
            if make_blobs_cluster_assignments is None:
                raise ValueError("Must provide a list of scikit-learn's cluster assignments.")

            if type(make_blobs_cluster_assignments) is numpy.ndarray:
                make_blobs_cluster_assignments = make_blobs_cluster_assignments.tolist()

            instances_data_frame["make_blob() Cluster Assignments"] = make_blobs_cluster_assignments
            sns.scatterplot(data=instances_data_frame, x="Feature 1", y="Feature 2",
                            s=scatter_plot_dot_size, hue="make_blob() Cluster Assignments", palette="Set2",
                            ax=axes[1])
            axes[1].set_title("make_blob() Clustering", fontweight="bold", fontsize=15)
            axes[1].set_xlabel("x's", fontweight="bold", fontsize=15)
            axes[1].set_ylabel("y's", fontweight="bold", fontsize=15)

            plt.legend(bbox_to_anchor=(1, 1), title="Cluster Assignments")
            if figure_file_name:
                plt.savefig(figure_file_name)

            plt.show()
        else:
            # Only plot data without comparing performance.
            figures, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            sns.scatterplot(data=instances_data_frame, x="Feature 1", y="Feature 2",
                            s=scatter_plot_dot_size, hue="Closest Centroids", palette="Set1")

            axes.set_xlabel("x's", fontweight="bold", fontsize=15)
            axes.set_ylabel("y's", fontweight="bold", fontsize=15)

            plt.title(scatter_plot_title, fontweight="bold", fontsize=20)
            plt.legend(bbox_to_anchor=(1, 1), title="Cluster Assignments")

            if figure_file_name:
                axes.figure.savefig(figure_file_name)

            plt.show()


