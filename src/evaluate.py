#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate result from clustering.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 13:40:31

"""

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

def plot_labels_over_time(fp_timestamps, labels, fingerprints, timestamps,
        data):

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    window_size = params["featurize"]["window_size"]



    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # fig.add_trace(
    #     go.Scatter(x=fp_timestamps, y=labels,
    #         name="Cluster labels"),
    #     secondary_y=False,
    # )

    n_features = fingerprints.shape[0]
    n_labels = len(labels)
    colors = ["red", "green", "blue", "brown", "yellow", "purple", "grey",
            "black", "pink", "orange"]

    for i in range(n_features):
        for j in range(n_labels):

            start = j * self.step
            stop = start + window_size
            t = timestamps.iloc[start:stop]

            cluster = labels[j]

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=data[self.input_columns[i]].iloc[start:stop],
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
                x=timestamps[::self.step],
                y=sum_dist
                name="Distance sum",
            ),
            secondary_y=True,
    )


    fig.update_layout(title_text="Cluster labels over time")
    fig.update_xaxes(title_text="date")
    fig.update_yaxes(title_text="Cluster label number", secondary_y=False)
    fig.update_yaxes(title_text="Input unit", secondary_y=True)

    fig.write_html(f"plot_{data_set}.html")

def plot_cluster_center_distance(self, data_set="train"):

    if data_set == "test":
        start_idx = self.test_start_idx
        stop_idx = self.test_stop_idx
        X = self.test_fingerprints
        timestamps = self.test_fp_timestamps
    elif data_set == "train":
        start_idx = self.train_start_idx
        stop_idx = self.train_stop_idx
        X = fingerprints
        timestamps = self.train_fp_timestamps
    else:
        raise ValueError("Must specify train or test data set.")

    dist = model.transform(X)
    dist = dist.sum(axis=1)
    avg_dist = pd.Series(dist).rolling(50).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(dist, label="dist")
    plt.plot(avg_dist, label="avg_dist")

    plt.legend()
    plt.plot()

    # print(dist.shape)
    # print(avg_dist.shape)

    return dist, avg_dist
