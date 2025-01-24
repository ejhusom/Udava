#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities related to clustering and post-processing.

Author:
    Erik Johannes Husom

Created:
    2023-03-08 onsdag 09:09:52 

Description:
    This module contains functions for clustering and post-processing of
    clustering results.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    euclidean_distances,
    silhouette_score,
)

from config import *

def filter_segments(labels, min_segment_length, distances_to_centers=None):
    """Filter out segments which are too short.

    This function filters out segments which are too short. If the segment is
    too short, it will be merged with the neighboring segment. If the segment
    is too short, and the neighboring segments have different labels, the
    segment will be split in half, and each half will be "swallowed" by the
    neighboring segments.

    Args:
        labels (np.array): Array of labels.
        min_segment_length (int): Minimum length of a segment.
        distances_to_centers (np.array): Array of distances to cluster centers.

    Returns:
        np.array: Array of updated labels.

    """

    # Array for storing updated labels after short segments are filtered out.
    new_labels = labels.copy()

    segments = find_segments(labels)

    segments_sorted_on_length = segments[segments[:, 2].argsort()]

    shortest_segment = np.min(segments[:, 2])
    number_of_segments = len(segments)

    # Filter out the segments which are too short
    while shortest_segment < min_segment_length:

        current_segment = segments_sorted_on_length[0]
        length = current_segment[2]

        # If the segment length is long enough, break the loop
        if length >= min_segment_length:
            break

        # Get the information of current segment
        segment_idx = current_segment[0]
        label = current_segment[1]
        start_idx = current_segment[3]
        end_idx = current_segment[4]

        current_distances = distances_to_centers[start_idx : end_idx + 1, :]

        # Set the original smallest distance to be above the maximum, in order
        # to find the second closest cluster center
        current_distances[:, label] = np.max(current_distances) + 1

        # Find the second closest cluster center of the current data points
        second_closest_cluster_centers = current_distances.argmin(axis=1)

        # Find the most frequent second closest cluster center in the segment
        counts = np.bincount(second_closest_cluster_centers)
        most_frequent_second_closest_cluster_center = np.argmax(counts)
        current_new_labels = (
            np.ones_like(second_closest_cluster_centers)
            * most_frequent_second_closest_cluster_center
        )

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
        shortest_segment = np.min(segments[:, 2])

        if len(segments) == number_of_segments:
            print("Could not remove any more segments.")
            break

        number_of_segments = len(segments)

    return new_labels

def filter_segments_plot_snapshots(labels, min_segment_length, feature_vector_timestamps,
        feature_vectors, original_data, model, distances_to_centers=None):
    """Filter out segments which are too short.

    This function filters out segments which are too short. If the segment is
    too short, it will be merged with the neighboring segment. If the segment
    is too short, and the neighboring segments have different labels, the
    segment will be split in half, and each half will be "swallowed" by the
    neighboring segments.

    Args:
        labels (np.array): Array of labels.
        min_segment_length (int): Minimum length of a segment.
        distances_to_centers (np.array): Array of distances to cluster centers.

    Returns:
        np.array: Array of updated labels.

    """

    # Array for storing updated labels after short segments are filtered out.
    new_labels = labels.copy()

    segments = find_segments(labels)

    segments_sorted_on_length = segments[segments[:, 2].argsort()]

    shortest_segment = np.min(segments[:, 2])
    number_of_segments = len(segments)

    counter = 0

    # Filter out the segments which are too short
    while shortest_segment < min_segment_length:

        current_segment = segments_sorted_on_length[0]
        length = current_segment[2]

        # If the segment length is long enough, break the loop
        if length >= min_segment_length:
            break

        # Get the information of current segment
        segment_idx = current_segment[0]
        label = current_segment[1]
        start_idx = current_segment[3]
        end_idx = current_segment[4]

        current_distances = distances_to_centers[start_idx : end_idx + 1, :]

        # Set the original smallest distance to be above the maximum, in order
        # to find the second closest cluster center
        current_distances[:, label] = np.max(current_distances) + 1

        # Find the second closest cluster center of the current data points
        second_closest_cluster_centers = current_distances.argmin(axis=1)

        # Find the most frequent second closest cluster center in the segment
        counts = np.bincount(second_closest_cluster_centers)
        most_frequent_second_closest_cluster_center = np.argmax(counts)
        current_new_labels = (
            np.ones_like(second_closest_cluster_centers)
            * most_frequent_second_closest_cluster_center
        )

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
        shortest_segment = np.min(segments[:, 2])

        if len(segments) == number_of_segments:
            print("Could not remove any more segments.")
            break

        number_of_segments = len(segments)

        if counter % 1 == 0:
            plot_labels_over_time(
                feature_vector_timestamps,
                new_labels,
                feature_vectors,
                original_data,
                model,
                mark_outliers=False,
                show_local_distance=False,
                filename=f"labels_over_time_{counter}.html"
            )

        counter += 1

    return new_labels

def create_event_log_from_segments(segments,
        feature_vector_timestamps=None):
    """Create an event log from segments.

    Args:
        segments (np.array): Array of segments.

    Returns:
        pd.DataFrame: Event log.

    """

    events = []

    if feature_vector_timestamps is None:
        feature_vector_timestamps = np.load(FEATURE_VECTOR_TIMESTAMPS_PATH)

    for i in range(len(segments)):

        current_segment = segments[i, :]
        label = current_segment[1]
        start_timestamp = feature_vector_timestamps[current_segment[3]]
        stop_timestamp = feature_vector_timestamps[current_segment[4]]

        events.append([start_timestamp, label, "started"])
        events.append([stop_timestamp, label, "completed"])

    event_log = pd.DataFrame(events, columns=["timestamp", "label", "status"])

    return event_log

def calculate_model_metrics(model, feature_vectors, labels):
    """Evaluate the cluster model.

    Silhouette score: Bounded between -1 for incorrect clustering and +1 for
        highly dense clustering. Scores around zero indicate overlapping clusters.
    Calinski-Harabasz Index: Higher when clusters are dense and well separated.
    Davies-Bouldin Index: Zero is the lowest score. Lower scores indicate a
        better partition.

    Args:
        model (sklearn.cluster): Cluster model.
        feature_vectors (np.array): Feature vectors.
        labels (np.array): Cluster labels.

    Returns:
        dict: Dictionary with the model metrics.

    """

    # Set invalid default values to indicate that they were not computed
    metrics = {
        "silhouette_score": -1000,
        "calinski_harabasz_score": -1000,
        "davies_bouldin_score": 1000
    }


    n_detected_clusters = np.unique(labels)

    if len(n_detected_clusters) == 1:
        print("Only one cluster detected. Skipping evaluation.")
        return metrics

    silhouette = silhouette_score(feature_vectors, labels)
    chs = calinski_harabasz_score(feature_vectors, labels)
    dbs = davies_bouldin_score(feature_vectors, labels)

    metrics = {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": chs,
        "davies_bouldin_score": dbs,
    }

    return metrics


def calculate_distances(feature_vectors, model, cluster_centers):

    distances_to_centers = euclidean_distances(feature_vectors, cluster_centers)
    sum_distance_to_centers = distances_to_centers.sum(axis=1)

    return distances_to_centers, sum_distance_to_centers


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

    if len(labels) == 0:
        return np.array(segments)

    current_label = labels[0]
    current_length = 1
    start_idx = 0
    segment_idx = 0

    # Count the length of each segment
    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_length += 1
        else:
            end_idx = i - 1
            segments.append(
                [segment_idx, current_label, current_length, start_idx, end_idx]
            )
            segment_idx += 1
            current_label = labels[i]
            current_length = 1
            start_idx = i

    # Append the last segment
    end_idx = len(labels) - 1
    segments.append(
        [segment_idx, current_label, current_length, start_idx, end_idx]
    )

    return np.array(segments)

def create_event_log(labels, identifier="",
        feature_vector_timestamps=None):
    """Create an event log from labels.

    This function creates an event log from an array of labels. The event log
    has the following format:

    timestamp, label, status

    Args:
        labels (np.array): Array of labels.
        identifier (str): Case identifier.

    Returns:
        pd.DataFrame: Event log.

    """

    if identifier == "":
        identifier = str(uuid.uuid4())

    segments = find_segments(labels)
    event_log = create_event_log_from_segments(segments,
            feature_vector_timestamps)
    event_log["source"] = identifier
    event_log["case"] = ""

    return event_log

def post_process_labels(
        model,
        cluster_centers,
        feature_vectors,
        labels,
        identifier,
        min_segment_length=0,
    ):
    """Post-process labels.

    This function post-processes the labels by filtering out segments that are
    too short. It also creates an event log from the labels.

    Args:
        model (sklearn.cluster): Cluster model.
        cluster_centers (np.array): Cluster centers.
        feature_vectors (np.array): Feature vectors.
        labels (np.array): Cluster labels.
        identifier (str): Case identifier.
        min_segment_length (int): Minimum segment length.

    Returns:
        tuple: Tuple containing: (1) array of labels, (2) event log.
        
    """

    if min_segment_length > 0:
        distances_to_centers, sum_distance_to_centers = calculate_distances(
            feature_vectors, model, cluster_centers
        )
        labels = filter_segments(labels, min_segment_length, distances_to_centers)

    event_log = create_event_log(labels, identifier=identifier)

    return labels, event_log

def plot_labels_over_time(
    feature_vector_timestamps,
    labels,
    feature_vectors,
    original_data,
    model,
    mark_outliers=False,
    show_local_distance=False,
    reduce_plot_size=False,
    filename=None,
    return_fig=False,
    png_only=False,
):
    """Plot labels over time.

    This function plots the labels over time. It also plots the local
    distance of each data point to its cluster center.

    Args:
        feature_vector_timestamps (np.array): Timestamps of feature vectors.
        labels (np.array): Labels.
        feature_vectors (np.array): Feature vectors.
        original_data (pd.DataFrame): Original data.
        model (sklearn.cluster): Cluster model.
        mark_outliers (bool): If True, outliers will be marked with a grey
            color.
        show_local_distance (bool): If True, the local distance of each
            data point to its cluster center will be plotted.
        reduce_plot_size (bool): If True, the plot will be reduced in size.

    Returns:
        None.

    """

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    columns = params["featurize"]["columns"]

    cluster_centers = pd.read_csv(
        OUTPUT_PATH / "cluster_centers.csv", index_col=0
    ).to_numpy()

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
        print("Reducing the plot size")

    # If reduce plot size, take only the nth data point, where nth is set to be
    # a fraction of the window size. Large fraction if the window size is
    # small, and small fraction if the window size is large.
    nth = min(int(window_size / np.log(window_size)), window_size)
    nth = 100000
    print("=====")
    print(n_labels)
    print(len(original_data))

    # # Reshape labels to match the DataFrame length
    # expanded_labels = np.repeat(labels, window_size)[:len(original_data)]

    # # Normalize the expanded_labels to range [0,1]
    # normalized_labels = (expanded_labels - expanded_labels.min()) / (expanded_labels.max() - expanded_labels.min())

    # # Create a custom color scale
    # color_scale = [(label / max(expanded_labels), color) for label, color in enumerate(COLORS) if label in np.unique(expanded_labels)]

    # Plot the data using scattergl for better performance with large datasets
    # fig.add_trace(
    #     go.Scattergl(
    #         x=original_data.index,
    #         y=original_data['Channel_4_Data'],
    #         mode='markers+lines',  # Use both markers and lines
    #         marker=dict(
    #             color=normalized_labels,  # Set color of the markers as the normalized labels
    #             colorscale=color_scale,  # Define custom color scale
    #             colorbar=dict(title='Labels'),  # Optional: to show a color bar
    #             size=3,  # Optional: adjust marker size
    #             cmin=0,  # Set min for color scale
    #             cmax=1,  # Set max for color scale
    #         ),
    #         line=dict(shape='hv')  # Use a horizontal-vertical step line
    #     )
    # )




    for i in range(n_features):
        # for j in range(n_labels):
        j = 0
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
                    y=y,
                    line=dict(color=color),
                    showlegend=False,
                ),
            )

            j += 1

            # if j % 100:
            #     print(start)


    
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

    if filename is None:
        fig.write_image(str(PLOTS_PATH / "labels_over_time.png"), height=500, width=860)
        if not png_only:
            fig.write_html(str(PLOTS_PATH / "labels_over_time.html"))
            fig.write_html("src/templates/prediction.html")
    else:
        fig.write_html(filename)

    if return_fig:
        return fig
    else:
        return fig.to_html(full_html=False)


def plot_labels_over_time_matplotlib(
    feature_vector_timestamps,
    labels,
    feature_vectors,
    original_data,
    model,
    mark_outliers=False,
    show_local_distance=False,
    reduce_plot_size=False,
    filename=None,
    return_fig=False,
):
    """Plot labels over time.

    This function plots the labels over time. It also plots the local
    distance of each data point to its cluster center.

    Args:
        feature_vector_timestamps (np.array): Timestamps of feature vectors.
        labels (np.array): Labels.
        feature_vectors (np.array): Feature vectors.
        original_data (pd.DataFrame): Original data.
        model (sklearn.cluster): Cluster model.
        mark_outliers (bool): If True, outliers will be marked with a grey
            color.
        show_local_distance (bool): If True, the local distance of each
            data point to its cluster center will be plotted.
        reduce_plot_size (bool): If True, the plot will be reduced in size.

    Returns:
        None.

    """

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    columns = params["featurize"]["columns"]

    cluster_centers = pd.read_csv(
        OUTPUT_PATH / "cluster_centers.csv", index_col=0
    ).to_numpy()

    if type(columns) is str:
        columns = [columns]

    step = window_size - overlap

    # dist = model.transform(feature_vectors)
    dist = euclidean_distances(feature_vectors, cluster_centers)
    sum_dist = dist.sum(axis=1)

    if mark_outliers:
        labels = filter_outliers(labels, dist)

    fig, ax1 = plt.subplots(figsize=(10, 6))  # You can adjust the figure size
    ax2 = ax1.twinx()  # Create a second y-axis to plot the deviation metric

    n_features = len(columns)
    n_labels = len(labels)

    timestamps = original_data.index

    if n_labels > 3000:
        reduce_plot_size = True

    # If reduce plot size, take only the nth data point, where nth is set to be
    # a fraction of the window size. Large fraction of the window size is
    # small, and small fraction if the window size is large.
    nth = min(int(window_size / np.log(window_size)), window_size)
    nth = 10000


    for i in range(n_features):
        # for j in range(n_labels):
        j = 0
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

            ax1.plot(t, y, color=color)

            j += 1

    if show_local_distance and not reduce_plot_size:
        label_indeces = labels.reshape(len(labels), 1)
        local_distance = np.take_along_axis(dist, label_indeces, axis=1).flatten()
        ax2.plot(feature_vector_timestamps, local_distance, color='blue')


        # Plot distance to each cluster center
        for i in range(dist.shape[1]):
            ax2.plot(feature_vector_timestamps, dist[:, i], color=COLORS[i])


    # Plot deviation metric
    ax2.plot(feature_vector_timestamps, sum_dist, color='black', label="Deviation metric")

    ax1.set_title("Cluster labels over time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sensor data unit")
    ax2.set_ylabel("Deviation metric")

    fig.tight_layout()  # Adjust the layout

    plt.savefig(str(PLOTS_PATH / "labels_over_time.png"))  # Save the figure

    if return_fig:
        return fig
    # else:
    #     plt.show()  # Show the plot
