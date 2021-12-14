#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate result from clustering.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 13:40:31

"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sn
from catch22 import catch22_all
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
from mpl_toolkits import mplot3d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml

from config import *


def visualize_clusters(labels, fingerprints, model, dim1=0, dim2=1, dim3=None,
        width=10, height=10):
    """Plot data point and cluster centers in a reduced feature space.

    Args:
        dim1 (int): Index of first feature to use in plot.
        dim2 (int): Index of second feature to use in plot.
        dim3 (int): Index of third feature to use in plot. If None (which
            is default), the plot will be in 2D. If not None, the plot will
            be in 3D.

    """

    clusters = np.unique(labels)

    if dim3 is None:
        plt.figure(figsize=(width, height))

        for cluster in clusters:
            current_cluster_indeces = np.where(labels == cluster)
            current_cluster_points = fingerprints[current_cluster_indeces]
            plt.scatter(current_cluster_points[:, dim1], current_cluster_points[:, dim2])

        plt.scatter(
            model.cluster_centers_[:, dim1], model.cluster_centers_[:, dim2],
            s=70,
            c="black",
            edgecolors="white"
        )
        plt.show()
    else:
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection ='3d')
        ax.scatter(
                fingerprints[:, dim1], 
                fingerprints[:, dim2], 
                fingerprints[:, dim3],
                alpha=0.1
        )
        ax.scatter(
            model.cluster_centers_[:, dim1], 
            model.cluster_centers_[:, dim2],
            model.cluster_centers_[:, dim3],
            alpha=1.0
        )
        plt.show()

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_PATH / "clusters.png")

def plot_labels_over_time(fp_timestamps, labels, fingerprints,
        original_data, model):

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    columns = params["featurize"]["columns"]

    print(columns)
    if type(columns) is str:
        columns = [columns]

    step = window_size - overlap

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # fig.add_trace(
    #     go.Scatter(x=fp_timestamps, y=labels,
    #         name="Cluster labels"),
    #     secondary_y=False,
    # )

    # n_features = fingerprints.shape[0]
    n_features = len(columns)
    n_labels = len(labels)
    colors = ["red", "green", "blue", "brown", "yellow", "purple", "grey",
            "black", "pink", "orange"]

    timestamps = original_data.index

    for i in range(n_features):
        for j in range(n_labels):

            start = j * step
            stop = start + window_size
            # t = original_data.index.iloc[start:stop]
            t = timestamps[start:stop]

            cluster = labels[j]

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=original_data[columns[i]].iloc[start:stop],
                    # name=self.input_columns[i],
                    # mode="markers"
                    line=dict(color=colors[cluster]),
                    showlegend=False,
                ),
                secondary_y=True,
            )

        # fig.add_trace(
        #     go.Scatter(
        #         x=timestamps,
        #         y=data[self.input_columns[i]],
        #         name=self.input_columns[i],
        #         # mode="markers"
        #     ),
        #     secondary_y=True,
        # )
        # if j > 100:
        #     break

    dist = model.transform(fingerprints)
    sum_dist = dist.sum(axis=1)

    # for i in range(dist.shape[1]):
    #     fig.add_trace(
    #             go.Scatter(
    #                 x=timestamps[::self.step],
    #                 y=dist[:,i],
    #             ),
    #             secondary_y=True,
    #     )

    fig.add_trace(
            go.Scatter(
                x=timestamps[::step],
                y=sum_dist,
                name="Distance sum",
            ),
            secondary_y=True,
    )


    fig.update_layout(title_text="Cluster labels over time")
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="Cluster label number", secondary_y=False)
    fig.update_yaxes(title_text="Input unit", secondary_y=True)

    fig.write_html(str(PLOTS_PATH / "labels_over_time.html"))

def plot_cluster_center_distance(fp_timestamps, fingerprints, model):

    dist = model.transform(fingerprints)
    dist = dist.sum(axis=1)
    avg_dist = pd.Series(dist).rolling(50).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(dist, label="dist")
    plt.plot(avg_dist, label="avg_dist")

    plt.legend()
    plt.show()

    return dist, avg_dist


if __name__ == '__main__': 

    labels = pd.read_csv(OUTPUT_PATH / "labels.csv").iloc[:,-1].to_numpy()
    original_data = pd.read_csv(OUTPUT_PATH / "combined.csv", index_col=0)
    fingerprints = np.load(DATA_FEATURIZED_PATH / "featurized.npy")
    fingerprint_timestamps = np.load(OUTPUT_PATH /
    "fingerprint_timestamps.npy")
    model = joblib.load(MODELS_FILE_PATH)

    visualize_clusters(labels, fingerprints, model)
    plot_labels_over_time(fingerprint_timestamps, labels, fingerprints,
            original_data, model)
    # plot_cluster_center_distance(fingerprint_timestamps, fingerprints, model)
