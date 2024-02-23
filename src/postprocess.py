#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess result of training cluster model.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 13:40:31

"""
import json

from datetime import timedelta
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

# from catch22 import catch22_all
from mpl_toolkits import mplot3d
from plotly.subplots import make_subplots
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)
from sklearn.metrics import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cluster_utils import (
    calculate_distances,
    calculate_model_metrics,
    create_event_log,
    filter_segments,
    filter_segments_plot_snapshots,
    find_segments,
    plot_labels_over_time,
)
from config import *
from preprocess_utils import find_files


def filter_outliers(labels, distances, percentile=95, separate_thresholds=False):
    """Filter outliers from labels.

    The outliers are defined as data points that are further away from their
    cluster center than the given percentile.

    Args:
        labels (np.array): Labels.
        distances (np.array): Distances from each data point to each cluster
            center.
        percentile (int): Percentile to use for outlier detection.
        separate_thresholds (bool): If True, each cluster will have its own
            threshold for outlier detection. If False, all clusters will use
            the same threshold.

    Returns:
        np.array: Labels with outliers filtered out.

    """

    local_distances = np.take_along_axis(
        distances, labels.reshape(len(labels), 1), axis=1
    ).flatten()

    outlier_indeces = np.empty((0, labels.shape[0]), dtype=np.int64)

    if separate_thresholds:
        # Calculate percentiles
        for c in np.unique(labels):
            current_indeces = np.where(labels == c)[0]
            current_distances = distances[current_indeces, c].flatten()
            current_percentile = np.percentile(current_distances, percentile)

            new_outlier_indeces = []
            # TODO: Vectorize this stuff
            for i in current_indeces:
                # print(distances[i,c])
                if distances[i, c] > current_percentile:
                    new_outlier_indeces.append(int(i))

            outlier_indeces = np.append(outlier_indeces, new_outlier_indeces)

        outlier_indeces = np.array(outlier_indeces, dtype=np.int64)

    else:
        outlier_indeces = np.where(
            local_distances > np.percentile(local_distances, percentile)
        )

    labels[outlier_indeces] = -1

    return labels


def visualize_clusters(
    labels,
    feature_vectors,
    model,
    dim1=0,
    dim2=1,
    dim3=None,
    width=10,
    height=10,
    mark_outliers=False,
    label_data_points=False,
):
    """Plot data point and cluster centers in a reduced feature space.

    Args:
        labels (np.array): Labels.
        feature_vectors (np.array): Feature vectors.
        model (sklearn.cluster): Cluster model.
        dim1 (int): Index of first feature to use in plot.
        dim2 (int): Index of second feature to use in plot.
        dim3 (int): Index of third feature to use in plot. If None (which
            is default), the plot will be in 2D. If not None, the plot will
            be in 3D.
        width (int): Width of plot.
        height (int): Height of plot.
        mark_outliers (bool): If True, outliers will be marked with a grey
            color.
        label_data_points (bool): If True, data points will be labeled with

    Returns:
        None.

    """

    clusters = np.unique(labels)
    cluster_centers = pd.read_csv(
        OUTPUT_PATH / "cluster_centers.csv", index_col=0
    ).to_numpy()

    if mark_outliers:
        # dist = model.transform(feature_vectors)
        dist = euclidean_distances(feature_vectors, cluster_centers)
        labels = filter_outliers(labels, dist)

    if dim3 is None:
        plt.figure(figsize=(width, height))

        for c in clusters:
            current_cluster_indeces = np.where(labels == c)
            current_cluster_points = feature_vectors[current_cluster_indeces]
            plt.scatter(
                current_cluster_points[:, dim1],
                current_cluster_points[:, dim2],
                color=COLORS[c],
                label=f"Cluster {c}",
            )

            if label_data_points:
                for i in range(current_cluster_points.shape[0]):
                    plt.annotate(
                        c,
                        (
                            current_cluster_points[i, dim1],
                            current_cluster_points[i, dim2],
                        ),
                    )

        if mark_outliers:
            current_cluster_indeces = np.where(labels == -1)
            current_cluster_points = feature_vectors[current_cluster_indeces]
            plt.scatter(
                current_cluster_points[:, dim1],
                current_cluster_points[:, dim2],
                color="grey",
            )

        for i, c in enumerate(clusters):
            plt.scatter(
                cluster_centers[i, dim1],
                cluster_centers[i, dim2],
                s=90,
                c=COLORS[c],
                edgecolors="black",
                marker="d",
            )

            if label_data_points:
                plt.annotate(c, (cluster_centers[i, dim1], cluster_centers[i, dim2]))

        plt.legend()

    else:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.scatter(
            feature_vectors[:, dim1],
            feature_vectors[:, dim2],
            feature_vectors[:, dim3],
            alpha=0.1,
        )
        ax.scatter(
            cluster_centers[:, dim1],
            cluster_centers[:, dim2],
            cluster_centers[:, dim3],
            alpha=1.0,
        )

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_PATH / "clusters.png", dpi=300)
    # plt.show()

def plot_cluster_center_distance(feature_vector_timestamps, feature_vectors, model):
    """Plot the distance of each data point to its cluster center.

    Args:
        feature_vector_timestamps (np.array): Timestamps of feature vectors.
        feature_vectors (np.array): Feature vectors.
        model (sklearn.cluster): Cluster model.

    Returns:
        None.
        
    """

    cluster_centers = pd.read_csv(
        OUTPUT_PATH / "cluster_centers.csv", index_col=0
    ).to_numpy()

    # dist = model.transform(feature_vectors)
    dist = euclidean_distances(feature_vectors, cluster_centers)
    dist = dist.sum(axis=1)
    avg_dist = pd.Series(dist).rolling(50).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(dist, label="dist")
    plt.plot(avg_dist, label="avg_dist")

    plt.legend()
    plt.show()

    return dist, avg_dist

def generate_cluster_names(model, cluster_centers):
    """Generate cluster names based on the characteristics of each cluster.

    Args:
        model: Cluster model trained on input data.
        cluster_centers (np.array): Cluster centers.

    Returns:
        cluster_names (list of str): Names based on feature characteristics.

    """

    levels = ["lowest", "low", "medium", "high", "highest"]
    cluster_labels = []
    cluster_names = []
    cluster_characteristics = []
    cluster_colors = []

    n_clusters = cluster_centers.shape[0]

    # Add index and color for each cluster
    for i in range(n_clusters):
        cluster_colors.append(COLORS[i])
        cluster_labels.append(i)
        cluster_names.append("")
        cluster_characteristics.append("")

    maxs = cluster_centers.argmax(axis=0)
    mins = cluster_centers.argmin(axis=0)

    for i in range(len(FEATURE_NAMES)):
        # cluster_names[maxs[i]] += "highest " + FEATURE_NAMES[i] + ", "
        # cluster_names[mins[i]] += "lowest " + FEATURE_NAMES[i] + ", "
        cluster_characteristics[maxs[i]] += "highest " + FEATURE_NAMES[i] + ", "
        cluster_characteristics[mins[i]] += "lowest " + FEATURE_NAMES[i] + ", "

    print(cluster_labels)
    print(cluster_names)
    print(cluster_characteristics)

    # cluster_names = pd.DataFrame([cluster_labels, cluster_names, cluster_characteristics], columns=["cluster_label", "cluster_name", "cluster_characteristics"])
    cluster_names = pd.DataFrame({
        "cluster_label": cluster_labels,
        "cluster_name": cluster_names,
        "cluster_characteristics": cluster_characteristics
        })

    return cluster_names

def postprocess(model, cluster_centers, feature_vectors, labels):
    """Postprocess the cluster labels.

    This function postprocess the output of the clustering algorithm. It
    filters out the segments that are too short, and it also merges the
    segments that are too close to each other.

    Args:
        model: Cluster model trained on input data.
        cluster_centers (np.array): Cluster centers.
        feature_vectors (np.array): Feature vectors.
        labels (np.array): Cluster labels.

    Returns:
        labels (np.array): Postprocessed cluster labels.

    """

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    learning_method = params["train"]["learning_method"]
    n_clusters = params["train"]["n_clusters"]
    max_iter = params["train"]["max_iter"]
    use_predefined_centroids = params["train"]["use_predefined_centroids"]
    fix_predefined_centroids = params["train"]["fix_predefined_centroids"]
    annotations_dir = params["train"]["annotations_dir"]
    min_segment_length = params["postprocess"]["min_segment_length"]

    # If the minimum segment length is set to be a non-zero value, we need to
    # filter the segments.
    if min_segment_length > 0:
        distances_to_centers, sum_distance_to_centers = calculate_distances(
            feature_vectors, model, cluster_centers
        )
        labels = filter_segments(labels, min_segment_length, distances_to_centers)

        # # Code to provide snapshots during filtering
        # original_data = pd.read_csv(ORIGINAL_TIME_SERIES_PATH, index_col=0)
        # feature_vector_timestamps = np.load(FEATURE_VECTOR_TIMESTAMPS_PATH)
        # model = joblib.load(MODELS_FILE_PATH)

        # labels = filter_segments2(labels, min_segment_length,
        #         feature_vector_timestamps, feature_vectors, original_data,
        #         model, distances_to_centers)

    # Create event log
    event_log = create_event_log(labels, identifier=params["featurize"]["dataset"])

    # Hardcoded expectations (adhoc solution)
    try:
        with open("assets/data/expectations/" + params["featurize"]["dataset"] + "/expectations.json", "r") as f:
            # expectations = json.load(f) 
            expectations = eval(f.read())
    except:
        expectations = None
        print("No expectations found.")

    # Create and save cluster names
    cluster_names = generate_cluster_names(model, cluster_centers)

    # Read predefined centroids from file
    with open(PREDEFINED_CENTROIDS_PATH, "r") as f:
        predefined_centroids_dict = json.load(f)

    # Use cluster names from annotated data, if the number of clusters still
    # matches the number of unique annotation label (the number of clusters
    # might change when using cluster algorithms that automatically decide on a
    # suitable number of clusters.
    if use_predefined_centroids:
        if len(predefined_centroids_dict) == n_clusters:
            for i, key in enumerate(predefined_centroids_dict):
                # cluster_names["cluster_name"][i] = (
                #     str(cluster_names["cluster_name"][i].split(":")[0])
                #     + ": "
                #     + f" {key}, ".upper()
                #     + str(cluster_names["cluster_name"][i].split(":")[1])
                # )
                cluster_names["cluster_name"][i] = key.upper()

                if expectations != None:
                    # Add number to expectations
                    for expectation in expectations:
                        if expectation["name"].lower() == key.lower():
                            expectation["label"] = i

    cluster_names["source"] = params["featurize"]["dataset"]
    cluster_names.to_csv(OUTPUT_PATH / "cluster_names.csv", index=False)

    if expectations != None:
        event_log_score(event_log, expectations)

    event_log.to_csv(OUTPUT_PATH / "event_log.csv")


    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = calculate_model_metrics(model, feature_vectors, labels)

    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(metrics, f)

    return labels

def event_log_score(event_log, expectations):
    """Iterate through the event log and check whether the events matches the
    expected order and duration.

    Arguments:
        event_log ():
        expectations (list): List of expected events.

    Returns:
        score

    """

    hits = 0
    misses = 0

    # print(event_log)
    print(expectations)

    event_log["duration_correct"] = 0
    event_log["next_event_correct"] = 0

    if isinstance(event_log["timestamp"][0], np.ndarray):
        event_log["timestamp"] = event_log["timestamp"].apply(lambda x: x[0])

    if isinstance(event_log["timestamp"][0], str):
        event_log["timestamp"] = pd.to_datetime(event_log["timestamp"])

    # Loop through every second row in dataframe
    for i, event in event_log.iloc[1::2].iterrows():
        event_label = event["label"]

        # Find duration
        event_duration = event["timestamp"] - event_log.iloc[i - 1]["timestamp"]
        if isinstance(event_duration, np.float64):
            seconds = event_duration
        else:
            seconds = event_duration.seconds
        event_duration = timedelta(seconds=seconds)


        # Find expectations
        expected_duration_ranges = []
        expected_next_events = []
        for j, expectation in enumerate(expectations):
            if expectation["label"] == event_label:

                # expected_duration_ranges.append(expectation["duration"])
                expected_duration_range = expectation["duration"]

                if j < len(expectations) - 1:
                    expected_next_events.append(expectations[j + 1]["label"])
                # If this is the last event in the cycle, the next expected
                # event is the first in the cycle.
                else:
                    expected_next_events.append(expectations[0]["label"])

        # Compare duration
        # Convert to timedelta
        min_duration = timedelta(seconds=expected_duration_range[0])
        max_duration = timedelta(seconds=expected_duration_range[1])
        if event_duration < min_duration or event_duration > max_duration:
            # print(
            #     f"Event {event_label} has duration {event_duration} which is not in the expected range {expected_duration_range} (timestamp: {event['timestamp']})"
            # )
            misses += 1
        else:
            hits += 1
            event_log["duration_correct"][i] = 1

        # Find next event
        # If this is the last event in the event log, there is no next event.
        if i == len(event_log) - 1:
            continue
        else:
            next_event = event_log.iloc[i + 1]
            next_event_label = next_event["label"]

        if next_event["label"] in expected_next_events:
            hits += 1
            event_log["next_event_correct"][i] = 1
        else:
            # print(
            #     f"Event {event_label} has next event {next_event_label} which is not in the expected events {expected_next_events} (timestamp: {event['timestamp']})"
            # )
            misses += 1

    score = hits / (hits + misses)
    print(event_log)

    print(f"Event log score: {score}")

    with open("assets/output/event_log_score.txt", "w") as f:
        f.write(str(score))

    return score, event_log

if __name__ == "__main__":

    # Load data
    labels = pd.read_csv(LABELS_PATH).iloc[:, -1].to_numpy()
    original_data = pd.read_csv(ORIGINAL_TIME_SERIES_PATH, index_col=0)
    feature_vectors = np.load(FEATURE_VECTORS_PATH)
    feature_vector_timestamps = np.load(FEATURE_VECTOR_TIMESTAMPS_PATH)
    cluster_centers = pd.read_csv(CLUSTER_CENTERS_PATH, index_col=0).to_numpy()
    model = joblib.load(MODELS_FILE_PATH)

    labels = postprocess(model, cluster_centers, feature_vectors, labels)

    # visualize_clusters(
    #     labels, feature_vectors, model, dim1=0, dim2=4, mark_outliers=False
    # )

    plot_labels_over_time(
        feature_vector_timestamps,
        labels,
        feature_vectors,
        original_data,
        model,
        mark_outliers=False,
        show_local_distance=False,
    )

    # plot_cluster_center_distance(feature_vector_timestamps, feature_vectors, model)
