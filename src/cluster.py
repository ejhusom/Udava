#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clustering for historical data validation.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 12:05:02

"""
import sys

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)

from annotations import *
from config import *
from preprocess_utils import find_files, move_column


def cluster(dir_path=""):

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    learning_method = params["cluster"]["learning_method"]
    n_clusters = params["cluster"]["n_clusters"]
    max_iter = params["cluster"]["max_iter"]
    use_predefined_centroids = params["cluster"]["use_predefined_centroids"]
    fix_predefined_centroids = params["cluster"]["fix_predefined_centroids"]
    annotations_dir = params["cluster"]["annotations_dir"]
    min_segment_length = params["cluster"]["min_segment_length"]

    # Find data files and load feature_vectors.
    filepaths = find_files(dir_path, file_extension=".npy")
    feature_vectors = np.load(filepaths[0])

    model = build_model(learning_method, n_clusters, max_iter)

    if use_predefined_centroids:
        try:
            annotations_data_filepath = find_files(
                ANNOTATIONS_PATH / annotations_dir, file_extension=".csv"
            )[0]
        except:
            raise FileNotFoundError(
                "Annotation data not found. Cannot create predefined clusters without annotation data."
            )

        try:
            annotations_filepath = find_files(
                ANNOTATIONS_PATH / annotations_dir, file_extension=".json"
            )[0]
        except:
            raise FileNotFoundError(
                "Annotations not found. Cannot create predefined clusters without annotations."
            )

        annotation_data = pd.read_csv(annotations_data_filepath, index_col=0)
        annotations = read_annotations(annotations_filepath)
        predefined_centroids_dict = create_cluster_centers_from_annotations(
            annotation_data, annotations
        )

        with open(PREDEFINED_CENTROIDS_PATH, "w") as f:
            json.dump(predefined_centroids_dict, f)

        labels, model = fit_predict_with_predefined_centroids(
            feature_vectors,
            model,
            n_clusters,
            predefined_centroids_dict,
            fix_predefined_centroids,
            max_iter=max_iter,
        )
    else:
        labels, model = fit_predict(feature_vectors, model)

    if min_segment_length > 0:
        distances_to_centers, sum_distance_to_centers = calculate_distances(
            feature_vectors, model
        )
        labels = filter_segments(labels, distances_to_centers, min_segment_length)
    
    segments = find_segments(labels)
    event_log = create_event_log(segments)
    event_log["case"] = params["featurize"]["dataset"]

    MODELS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(OUTPUT_PATH / "labels.csv")
    pd.DataFrame(model.cluster_centers_).to_csv(OUTPUT_PATH / "cluster_centers.csv")
    pd.DataFrame(feature_vectors).to_csv(OUTPUT_PATH / "feature_vectors.csv")
    event_log.to_csv(OUTPUT_PATH / "event_log.csv")

    dump(model, MODELS_FILE_PATH)

    cluster_names = generate_cluster_names(model)

    if use_predefined_centroids:
        for i, key in enumerate(predefined_centroids_dict):
            cluster_names["cluster_name"][i] = (
                str(cluster_names["cluster_name"][i].split(":")[0])
                + ": "
                + f" {key}, ".upper()
                + str(cluster_names["cluster_name"][i].split(":")[1])
            )

    cluster_names.to_csv(OUTPUT_PATH / "cluster_names.csv")


def build_model(learning_method, n_clusters, max_iter):
    """Build clustering model.

    Args:
        n_clusters (int): Number of clusters.
        max_iter (int): Maximum iterations.

    Returns:
        model: sklearn clustering model.

    """

    if learning_method == "meanshift":
        model = MeanShift()
    elif learning_method == "minibatchkmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter)
    else:
        model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter)
    # model = DBSCAN(eps=0.30, min_samples=3)
    # model = GaussianMixture(n_components=1)
    # model = AffinityPropagation(damping=0.5)

    return model


def fit_predict(feature_vectors, model):

    labels = model.fit_predict(feature_vectors)

    return labels, model


def fit_predict_with_predefined_centroids(
    feature_vectors,
    model,
    n_clusters,
    predefined_centroids_dict,
    fix_predefined_centroids=False,
    max_iter=100,
):
    """Fit a model with fixe cluster centroids.

    Args:
        predefined_centroids_dict (dict): A dictionary containing predefined
            clusters, where the keys are the names of each cluster, and the
            values are an array containing the cluster centroids.

    Returns:

    """

    n_predefined_centroids = len(predefined_centroids_dict)
    predefined_centroids = []

    # Get predefined centroids from dictionary to array.
    for key in predefined_centroids_dict:
        predefined_centroids.append(predefined_centroids_dict[key])

    predefined_centroids = np.array(predefined_centroids)

    # If the number of predefined clusters is greater than the parameter
    # n_clusters, the former will override the latter.
    if n_clusters <= predefined_centroids.shape[0]:

        if n_clusters != predefined_centroids.shape[0]:
            n_clusters = predefined_centroids.shape[0]
            print(
                f"""Number of clusters changed from {n_clusters} to
            {predefined_centroids.shape[0]} in order to match the
            number of predefined clusters."""
            )

        # If the predefined centroids should be fixed, simply use the
        # predefined centroids as the model's cluster centers.
        if fix_predefined_centroids:
            model = MiniBatchKMeans(max_iter=1, n_clusters=n_clusters)
            model.fit(feature_vectors)
            model.cluster_centers_ = predefined_centroids
            labels = model.predict(feature_vectors)

        else:
            model = MiniBatchKMeans(
                max_iter=max_iter, n_clusters=n_clusters, init=predefined_centroids
            )
            labels = model.fit_predict(feature_vectors)

        return labels, model

    # If the number of predefined clusters is less than the parameter
    # n_clusters, we need some extra random centroids.
    elif predefined_centroids.shape[0] < n_clusters:

        # Run one iteration of clustering to obtain initial centroids.
        model = MiniBatchKMeans(max_iter=1, n_clusters=n_clusters)
        model.fit(feature_vectors)
        initial_centroids = model.cluster_centers_

        # Overwrite some of the initial centroids with the predefined ones.
        initial_centroids[0 : predefined_centroids.shape[0], :] = predefined_centroids

        if fix_predefined_centroids:
            current_centroids = initial_centroids

            for i in range(max_iter):
                model = MiniBatchKMeans(
                    max_iter=1, n_clusters=n_clusters, init=current_centroids
                )

                model.fit(feature_vectors)
                current_centroids = model.cluster_centers_

                # Overwrite some of the current centroids with the predefined ones.
                current_centroids[
                    0 : predefined_centroids.shape[0], :
                ] = predefined_centroids

            labels = model.predict(feature_vectors)

        else:
            model = MiniBatchKMeans(
                max_iter=max_iter, n_clusters=n_clusters, init=initial_centroids
            )
            labels = model.fit_predict(feature_vectors)

        return labels, model


def predict(feature_vectors, model):

    labels = model.predict(feature_vectors)

    return labels


def calculate_distances(feature_vectors, model):

    distances_to_centers = model.transform(feature_vectors)
    sum_distance_to_centers = distances_to_centers.sum(axis=1)

    return distances_to_centers, sum_distance_to_centers


def generate_cluster_names(model):
    """Generate cluster names based on the characteristics of each cluster.

    Args:
        model: Cluster model trained on input data.

    Returns:
        cluster_names (list of str): Names based on feature characteristics.

    """

    num_clusters = model.cluster_centers_.shape[0]
    levels = ["lowest", "low", "medium", "high", "highest"]
    cluster_names = []

    for i in range(num_clusters):
        # cluster_names.append(str(i) + ": ")
        cluster_names.append(f"{i} ({COLORS[i]}): ")

    maxs = model.cluster_centers_.argmax(axis=0)
    mins = model.cluster_centers_.argmin(axis=0)

    for i in range(len(FEATURE_NAMES)):
        # cluster_names[maxs[i]] += "_highest_" + FEATURE_NAMES[i]
        # cluster_names[mins[i]] += "_lowest_" + FEATURE_NAMES[i]
        cluster_names[maxs[i]] += "highest " + FEATURE_NAMES[i] + ", "
        cluster_names[mins[i]] += "lowest " + FEATURE_NAMES[i] + ", "

    # if any(c == "" for c in cluster_names):
    #     for i in range(len(FEATURE_NAMES)):
    #         cluster_names[maxs[i]] += "highest_" + FEATURE_NAMES[i] + "_"
    #         cluster_names[mins[i]] += "lowest_" + FEATURE_NAMES[i] + "_"

    cluster_names = pd.DataFrame(cluster_names, columns=["cluster_name"])

    return cluster_names


def find_segments(labels):
    """Find segments in array of labels.

    By segments we mean a continuous sequence of the same label. 

    Args:
        labels (1d array): Array of labels.

    Returns:
        segments (2d array): Array of segments.

    Example:
        Let's say the array of labels looks like this:

        [0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2]

        In this example we have four segments:

        1. Two zeros: [0, 0]
        2. Three ones: [1, 1, 1]
        3. Four zeros: [0, 0, 0, 0]
        4. Three twos: [2, 2, 2]

        The array `segments` has the following format:

        [segment_index, label, segment_length, start_index, end_index]

        `start_index` and `end_index` are indeces of the `labels` array.
        In this example it will then return the following:

        [[0, 0, 2, 0, 1],
         [1, 1, 3, 2, 4],
         [2, 0, 4, 5, 8],
         [3, 2, 3, 9, 11]]


    """

    segments = []

    current_label = labels[0]
    current_length = 1
    start_idx = 0
    segment_idx = 0

    # Count the length of each segment
    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_length += 1
            if i == len(labels)-1:
                end_idx = i
                segments.append(
                    [segment_idx, current_label, current_length, start_idx, end_idx]
                )
        else:
            end_idx = i - 1
            segments.append(
                [segment_idx, current_label, current_length, start_idx, end_idx]
            )
            segment_idx += 1
            current_label = labels[i]
            current_length = 1
            start_idx = i

    return np.array(segments)

def create_event_log(segments):

    events = []
    feature_vector_timestamps = np.load(OUTPUT_PATH / "feature_vector_timestamps.npy")

    for i in range(len(segments)):

        current_segment = segments[i, :]
        label = current_segment[1]
        start_timestamp = feature_vector_timestamps[current_segment[3]]
        stop_timestamp = feature_vector_timestamps[current_segment[4]]

        events.append([start_timestamp, label, "started"])
        events.append([stop_timestamp, label, "completed"])

    event_log = pd.DataFrame(events, columns=["timestamp", "label", "status"])

    return event_log

def filter_segments(labels, distances_to_centers, min_segment_length):

    # Array for storing updated labels after short segments are filtered out.
    new_labels = labels.copy()

    segments = find_segments(labels)

    segments_sorted_on_length = segments[segments[:, 2].argsort()]

    shortest_segment = np.min(segments[:,2])
    number_of_segments = len(segments)

    # Filter out the segments which are too short
    # for i in range(len(segments)):
    while shortest_segment < min_segment_length:
        # Get the information of current segment
        current_segment = segments_sorted_on_length[0]
        segment_idx = current_segment[0]
        label = current_segment[1]
        length = current_segment[2]
        start_idx = current_segment[3]
        end_idx = current_segment[4]

        # If the segment length is long enough, break the loop
        if length >= min_segment_length:
            break


        current_distances = distances_to_centers[start_idx : end_idx + 1, :]

        # Set the original smallest distance to be above the maximum, in order
        # to find the second closest cluster center
        current_distances[:, label] = np.max(current_distances) + 1

        # Find the second closest cluster center of the current data points
        second_closest_cluster_centers = current_distances.argmin(axis=1)

        # Find the most frequent second closest cluster center in the segment
        counts = np.bincount(second_closest_cluster_centers)
        most_frequent_second_closest_cluster_center = np.argmax(counts)
        current_new_labels = np.ones_like(second_closest_cluster_centers) * most_frequent_second_closest_cluster_center

        # Find the neighboring segment labels
        if segment_idx == 0:
            # If it is the first segment, then the label of the previous
            # segment will be set to equal the one for the next segment
            label_of_previous_segment = segments[segment_idx + 1][1]
            label_of_next_segment = segments[segment_idx + 1][1]
        elif segment_idx == len(segments) - 1:
            # If it is the last segment, then the label of the next
            # segment will be set to equal the one for the previous segment
            label_of_previous_segment = segments[segment_idx - 1][1]
            label_of_next_segment = segments[segment_idx - 1][1]
        else:
            label_of_previous_segment = segments[segment_idx - 1][1]
            label_of_next_segment = segments[segment_idx + 1][1]

        # If the most frequent second closest cluster center in the segment is
        # different from the previous and next segment, the current segment
        # will be split in half, and each half will be "swallowed" by the
        # neighboring segments. Otherwise the label is set to either the
        # previous or next segment label.
        if most_frequent_second_closest_cluster_center == label_of_previous_segment:
            current_new_labels[:] = label_of_previous_segment
        elif most_frequent_second_closest_cluster_center == label_of_next_segment:
            current_new_labels[:] = label_of_next_segment
        else:
            current_new_labels[: length // 2] = label_of_previous_segment
            current_new_labels[length // 2 :] = label_of_next_segment

        # Update with new labels
        new_labels[start_idx : end_idx + 1] = current_new_labels

        # Recompute segments, since they now have changed
        segments = find_segments(new_labels)
        segments_sorted_on_length = segments[segments[:, 2].argsort()]
        shortest_segment = np.min(segments[:,2])

        if len(segments) == number_of_segments:
            print("Could not remove any more segments.")
            break

        number_of_segments = len(segments)

    return new_labels


if __name__ == "__main__":

    cluster(sys.argv[1])
