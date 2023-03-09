#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess result of training cluster model.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 13:40:31

"""


import joblib
import json
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

from config import *
from cluster_utils import filter_segments, create_event_log, calculate_model_metrics, calculate_distances, find_segments
from preprocess_utils import find_files


def filter_outliers(labels, distances, percentile=95, separate_thresholds=False):

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
        dim1 (int): Index of first feature to use in plot.
        dim2 (int): Index of second feature to use in plot.
        dim3 (int): Index of third feature to use in plot. If None (which
            is default), the plot will be in 2D. If not None, the plot will
            be in 3D.

    """

    clusters = np.unique(labels)
    cluster_centers = pd.read_csv(OUTPUT_PATH / "cluster_centers.csv",
            index_col=0).to_numpy()

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
                label=f"Cluster {c}"
            )

            if label_data_points:
                for i in range(current_cluster_points.shape[0]):
                    plt.annotate(c, (current_cluster_points[i, dim1],
                        current_cluster_points[i, dim2]))

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
                plt.annotate(c, (cluster_centers[i, dim1],
                    cluster_centers[i, dim2]))

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


def plot_labels_over_time(
    feature_vector_timestamps,
    labels,
    feature_vectors,
    original_data,
    model,
    mark_outliers=False,
    show_local_distance=False,
    reduce_plot_size=False,
):

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    columns = params["featurize"]["columns"]

    cluster_centers = pd.read_csv(OUTPUT_PATH / "cluster_centers.csv",
            index_col=0).to_numpy()

    if type(columns) is str:
        columns = [columns]

    step = window_size - overlap

    # dist = model.transform(feature_vectors)
    dist = euclidean_distances(feature_vectors, cluster_centers)
    sum_dist = dist.sum(axis=1)

    if mark_outliers:
        labels = filter_outliers(labels, dist)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    n_features = len(columns)
    n_labels = len(labels)

    timestamps = original_data.index

    if n_labels > 3000:
        reduce_plot_size = True

    # If reduce plot size, take only the nth data point, where nth is set to be
    # a fraction of the window size. Large fraction of the window size is
    # small, and small fraction if the window size is large.
    nth = min(int(window_size / np.log(window_size)), window_size)
    nth = 1500

    j = 0

    for i in range(n_features):
        # for j in range(n_labels):
        while j < n_labels:

            start = j * step
            stop = start + window_size
            t = timestamps[start:stop]
            y = original_data[columns[i]].iloc[start:stop]

            cluster = labels[j]

            if cluster == -1:
                color = "grey"
            else:
                color = COLORS[cluster]

            if reduce_plot_size:
                t = t[::nth]
                y = y[::nth]
                # j += 10

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=original_data[columns[i]].iloc[start:stop],
                    line=dict(color=color),
                    showlegend=False,
                ),
            )

            j += 1

    if show_local_distance and not reduce_plot_size:
        label_indeces = labels.reshape(len(labels), 1)
        local_distance = np.take_along_axis(dist, label_indeces, axis=1).flatten()
        fig.add_trace(
            go.Scatter(x=feature_vector_timestamps, y=local_distance),
            secondary_y=True,
        )

        # Plot distance to each cluster center
        for i in range(dist.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=feature_vector_timestamps,
                    y=dist[:, i],
                    line=dict(color=COLORS[i]),
                    showlegend=False,
                ),
                secondary_y=True,
            )

    # Plot deviation metric
    fig.add_trace(
        go.Scatter(
            # x=timestamps[::step],
            x=feature_vector_timestamps,
            y=sum_dist,
            name="Deviation metric",
            line=dict(color="black"),
        ),
        secondary_y=True,
    )

    fig.update_layout(title_text="Cluster labels over time")
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="Deviation metric", secondary_y=True)
    fig.update_yaxes(title_text="Sensor data unit", secondary_y=False)

    fig.write_html(str(PLOTS_PATH / "labels_over_time.html"))
    fig.write_html("src/templates/prediction.html")
    fig.write_image(str(PLOTS_PATH / "labels_over_time.png"), height=500, width=860)

    return fig.to_html(full_html=False)


def plot_cluster_center_distance(feature_vector_timestamps, feature_vectors, model):

    cluster_centers = pd.read_csv(OUTPUT_PATH / "cluster_centers.csv",
            index_col=0).to_numpy()

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

    Returns:
        cluster_names (list of str): Names based on feature characteristics.

    """

    levels = ["lowest", "low", "medium", "high", "highest"]
    cluster_names = []
    n_clusters = cluster_centers.shape[0]

    for i in range(n_clusters):
        cluster_names.append(f"{i} ({COLORS[i]}): ")

    maxs = cluster_centers.argmax(axis=0)
    mins = cluster_centers.argmin(axis=0)

    for i in range(len(FEATURE_NAMES)):
        cluster_names[maxs[i]] += "highest " + FEATURE_NAMES[i] + ", "
        cluster_names[mins[i]] += "lowest " + FEATURE_NAMES[i] + ", "

    cluster_names = pd.DataFrame(cluster_names, columns=["cluster_name"])

    return cluster_names


def postprocess(model, cluster_centers, feature_vectors, labels):

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    learning_method = params["cluster"]["learning_method"]
    n_clusters = params["cluster"]["n_clusters"]
    max_iter = params["cluster"]["max_iter"]
    use_predefined_centroids = params["cluster"]["use_predefined_centroids"]
    fix_predefined_centroids = params["cluster"]["fix_predefined_centroids"]
    annotations_dir = params["cluster"]["annotations_dir"]
    min_segment_length = params["postprocess"]["min_segment_length"]

    # If the minimum segment length is set to be a non-zero value, we need to
    # filter the segments.
    if min_segment_length > 0:
        distances_to_centers, sum_distance_to_centers = calculate_distances(
            feature_vectors, model, cluster_centers
        )
        labels = filter_segments(labels, min_segment_length, distances_to_centers)

    # Create event log
    event_log = create_event_log(
            labels, 
            identifier=params["featurize"]["dataset"]
    )

    event_log.to_csv(OUTPUT_PATH / "event_log.csv")

    # Create and save cluster names
    cluster_names = generate_cluster_names(model, cluster_centers)

    # Use cluster names from annotated data, if the number of clusters still
    # matches the number of unique annotation label (the number of clusters
    # might change when using cluster algorithms that automatically decide on a
    # suitable number of clusters.
    if use_predefined_centroids:
        if len(predefined_centroids_dict) == n_clusters:
            for i, key in enumerate(predefined_centroids_dict):
                cluster_names["cluster_name"][i] = (
                    str(cluster_names["cluster_name"][i].split(":")[0])
                    + ": "
                    + f" {key}, ".upper()
                    + str(cluster_names["cluster_name"][i].split(":")[1])
                )

    cluster_names.to_csv(OUTPUT_PATH / "cluster_names.csv")

    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = calculate_model_metrics(model, feature_vectors, labels)

    with open(METRICS_FILE_PATH, "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":

    labels = pd.read_csv(LABELS_PATH).iloc[:, -1].to_numpy()
    original_data = pd.read_csv(ORIGINAL_TIME_SERIES_PATH, index_col=0)
    feature_vectors = np.load(FEATURE_VECTORS_PATH)
    feature_vector_timestamps = np.load(FEATURE_VECTOR_TIMESTAMPS_PATH)
    cluster_centers = pd.read_csv(CLUSTER_CENTERS_PATH,
            index_col=0).to_numpy()
    model = joblib.load(MODELS_FILE_PATH)

    postprocess(model, cluster_centers, feature_vectors, labels)

    visualize_clusters(
        labels, feature_vectors, model, dim1=0, dim2=4, mark_outliers=False
    )

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