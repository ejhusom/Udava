#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsupervised learning for historic data validation.

Author:
    Erik Johannes Husom

Created:
    2021-10-06

"""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sn
from mpl_toolkits import mplot3d
from plotly.subplots import make_subplots
from pycatch22 import catch22_all
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import *

# pd.options.plotting.backend = "plotly"


class Udava:
    def __init__(
        self,
        df,
        train_start_idx=None,
        train_stop_idx=None,
        test_start_idx=None,
        test_stop_idx=None,
        timestamp_column_name="Timestamp",
    ):

        self.df = df

        self.timestamp_column_name = timestamp_column_name

        if self.df.empty:
            raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")

        # Ensure Timestamp column exists or create it early
        if self.timestamp_column_name not in self.df.columns:
            print(f"Warning: '{self.timestamp_column_name}' column not found. Using row indices as timestamps.")
            self.df[self.timestamp_column_name] = self.df.index

        self.train_start_idx = 0 if train_start_idx is None else train_start_idx
        self.train_stop_idx = (
            self.df.shape[0] - 1 if train_stop_idx is None else train_stop_idx
        )
        self.test_start_idx = 0 if test_start_idx is None else test_start_idx
        self.test_stop_idx = (
            self.df.shape[0] - 1 if test_stop_idx is None else test_stop_idx
        )

        self.train_start_date = self.df[self.timestamp_column_name].iloc[
            self.train_start_idx
        ]
        self.train_stop_date = self.df[self.timestamp_column_name].iloc[
            self.train_stop_idx
        ]
        self.test_start_date = self.df[self.timestamp_column_name].iloc[
            self.test_start_idx
        ]
        self.test_stop_date = self.df[self.timestamp_column_name].iloc[
            self.test_stop_idx
        ]

        self.model = None

        # Ensure train/test indices are within bounds for non-empty DataFrame
        if self.train_start_idx >= self.df.shape[0] or self.train_stop_idx >= self.df.shape[0]:
            raise IndexError("Train indices are out of bounds for the DataFrame.")

        if self.test_start_idx >= self.df.shape[0] or self.test_stop_idx >= self.df.shape[0]:
            raise IndexError("Test indices are out of bounds for the DataFrame.")

    def create_train_test_set(self, columns):

        self.input_columns = columns

        self.train_set = self.df[columns].iloc[
            self.train_start_idx : self.train_stop_idx
        ]
        self.test_set = self.df[columns].iloc[self.test_start_idx : self.test_stop_idx]

        self.train_timestamps = self.df[self.timestamp_column_name].iloc[
            self.train_start_idx : self.train_stop_idx
        ]
        self.test_timestamps = self.df[self.timestamp_column_name].iloc[
            self.test_start_idx : self.test_stop_idx
        ]

        return self.train_set, self.test_set

    def create_fingerprints(self, window_size=600, overlap=0):
        """Wrapper method for creating fingerprints.

        See function _create_fingerprints for full description.

        """

        self.window_size = window_size
        self.overlap = overlap

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0") 

        self.train_fingerprints, self.train_fp_timestamps = self._create_fingerprints(
            self.train_set,
            self.train_timestamps,
            window_size=window_size,
            overlap=overlap,
        )

        self.test_fingerprints, self.test_fp_timestamps = self._create_fingerprints(
            self.test_set,
            self.test_timestamps,
            window_size=window_size,
            overlap=overlap,
        )

        self.scaler = StandardScaler()
        self.train_fingerprints = self.scaler.fit_transform(self.train_fingerprints)
        self.test_fingerprints = self.scaler.transform(self.test_fingerprints)

        return self.train_fingerprints, self.test_fingerprints

    def _create_fingerprints(self, df, timestamps, window_size, overlap):
        """Create fingerprints of time series data.

        The fingerprint is based on statistical properties.

        Args:
            df (DataFrame): The data frame to create fingerprints from.
            timestamps (Series): The timestamps corresponding to the time series in
                df.
            window_size (int): Number of time steps to include when calculating
                a fingerprint.
            overlap (int): How much overlap between windows of the time series
                data.

        Returns:
            fingerprints (Numpy array): An array of fingerprints for the time
                series data. The array has size n*m, where n is the number of
                fingerprints (number of subsequences), and m is the number of
                features in the fingerprint.

        """

        n_features = df.shape[1]
        n_rows_raw = df.shape[0]
        n_rows = n_rows_raw // window_size

        step = window_size - overlap
        self.step = step

        # Initialize descriptive feature matrices
        mean = np.zeros((n_rows - 1, n_features))
        median = np.zeros((n_rows - 1, n_features))
        std = np.zeros((n_rows - 1, n_features))
        rms = np.zeros((n_rows - 1, n_features))
        var = np.zeros((n_rows - 1, n_features))
        minmax = np.zeros((n_rows - 1, n_features))
        frequency = np.zeros((n_rows - 1, n_features))
        # fingerprint_timestamps = np.zeros(n_rows - 1)

        # cfp = np.zeros((n_rows - 1, 22, n_features))

        # Loop through all observations and calculate features within window
        for i in range(n_rows - 1):
            start = i * step
            stop = start + window_size

            window = np.array(df.iloc[start:stop, :])
            # fingerprint_timestamps[i] = timestamps[stop - (step // 2)]

            mean[i, :] = np.mean(window, axis=0)
            median[i, :] = np.median(window, axis=0)
            std[i, :] = np.std(window, axis=0)
            # rms[i, :] = np.sqrt(np.mean(np.square(window, axis=0)))
            var[i, :] = np.var(window, axis=0)
            minmax[i, :] = np.max((window), axis=0) - np.min((window), axis=0)
            frequency[i, :] = np.linalg.norm(np.fft.rfft(window, axis=0), axis=0, ord=2)

            # for j in range(n_features):
            #     cfp[i, :, j] = catch22_all(window)["values"]

        features = np.concatenate((mean, median, std, var, minmax, frequency), axis=1)
        # cfp = np.nan_to_num(cfp)

        # features = cfp.reshape(n_rows - 1, 22*n_features)

        # print(f"Mean shape: {mean.shape}")
        # print(f"cfp shape: {cfp.shape}")
        # print(f"Features shape: {features.shape}")

        fingerprint_timestamps = timestamps[::step]

        return features, fingerprint_timestamps

    def build_model(self, method="minibatchkmeans", n_clusters=2, max_iter=100):
        """Build clustering model.

        Args:
            n_clusters (int): Number of clusters.
            max_iter (int): Maximum iterations.

        Returns:
            self.model: sklearn clustering model.

        """

        valid_methods = ["meanshift", "minibatchkmeans"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Supported methods are: {valid_methods}")

        if method == "meanshift":
            self.model = MeanShift()
        else:
            self.model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter)
        # self.model = DBSCAN(eps=0.30, min_samples=3)
        # self.model = GaussianMixture(n_components=1)
        # self.model = AffinityPropagation(damping=0.5)

        return self.model

    def load_model(self, filepath):

        self.model = joblib.load(filepath)

    def fit_predict(self):

        if self.model is None:
            self.build_model()

        self.train_labels = self.model.fit_predict(self.train_fingerprints)
        self.test_labels = self.model.predict(self.test_fingerprints)
        self.clusters = np.unique(self.train_labels)

        # Calculate distances to cluster centers for both train and test set
        self.train_dist = self.model.transform(self.train_fingerprints)
        self.test_dist = self.model.transform(self.test_fingerprints)
        self.train_dist_sum = self.train_dist.sum(axis=1)
        self.test_dist_sum = self.test_dist.sum(axis=1)

    def predict(self):

        self.train_labels = self.model.predict(self.train_fingerprints)
        self.test_labels = self.model.predict(self.test_fingerprints)
        self.clusters = np.unique(self.train_labels)

        # Calculate distances to cluster centers for both train and test set
        self.train_dist = self.model.transform(self.train_fingerprints)
        self.test_dist = self.model.transform(self.test_fingerprints)
        self.train_dist_sum = self.train_dist.sum(axis=1)
        self.test_dist_sum = self.test_dist.sum(axis=1)

    def calculate_distance_to_cluster_center(self, data_set="train"):

        distances = []

        for l, d in zip(self.train_labels, self.train_dist):
            distances.append(d[l])

        self.train_distance_to_cluster_center = distances

    def visualize_clusters(self, dim1=0, dim2=1, dim3=None, width=10, height=10):
        """Plot data point and cluster centers in a reduced feature space.

        Args:
            dim1 (int): Index of first feature to use in plot.
            dim2 (int): Index of second feature to use in plot.
            dim3 (int): Index of third feature to use in plot. If None (which
                is default), the plot will be in 2D. If not None, the plot will
                be in 3D.

        """
        print(self.train_labels)
        print("CLUSTERS")
        print(self.clusters)

        if dim3 is None:
            plt.figure(figsize=(width, height))

            for cluster in self.clusters:
                current_cluster_indeces = np.where(self.train_labels == cluster)
                current_cluster_points = self.train_fingerprints[
                    current_cluster_indeces
                ]

                print(current_cluster_points.shape)
                print(current_cluster_points)
                plt.scatter(
                    current_cluster_points[:, dim1], current_cluster_points[:, dim2]
                )

            # plt.scatter(self.train_fingerprints[:, dim1], self.train_fingerprints[:, dim2])

            plt.scatter(
                self.model.cluster_centers_[:, dim1],
                self.model.cluster_centers_[:, dim2],
                s=70,
                c="black",
                edgecolors="white",
            )
            # plt.show()
        else:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection="3d")
            ax.scatter(
                self.train_fingerprints[:, dim1],
                self.train_fingerprints[:, dim2],
                self.train_fingerprints[:, dim3],
                alpha=0.1,
            )
            ax.scatter(
                self.model.cluster_centers_[:, dim1],
                self.model.cluster_centers_[:, dim2],
                self.model.cluster_centers_[:, dim3],
                alpha=1.0,
            )

            plt.savefig("visualize_clusters.png")
            # plt.show()

    def plot_labels_over_time(self, data_set="train"):

        if data_set == "train":
            fp_timestamps = self.train_fp_timestamps
            labels = self.train_labels
            fingerprints = self.train_fingerprints
            timestamps = self.train_timestamps
            data = self.train_set
            start_idx = self.train_start_idx
            stop_idx = self.train_stop_idx
        elif data_set == "test":
            fp_timestamps = self.test_fp_timestamps
            labels = self.test_labels
            fingerprints = self.test_fingerprints
            timestamps = self.test_timestamps
            data = self.test_set
            start_idx = self.test_start_idx
            stop_idx = self.test_stop_idx

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # fig.add_trace(
        #     go.Scatter(x=fp_timestamps, y=labels,
        #         name="Cluster labels"),
        #     secondary_y=False,
        # )

        n_features = len(self.input_columns)
        print(labels.shape)
        n_labels = len(labels)
        colors = [
            "red",
            "green",
            "blue",
            "brown",
            "yellow",
            "purple",
            "grey",
            "black",
            "pink",
            "orange",
        ]

        for i in range(n_features):
            for j in range(n_labels):

                start = j * self.step
                stop = start + self.window_size
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

        dist = self.model.transform(fingerprints)
        # # dist = dist.max(axis=1)

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
                x=timestamps[:: self.step],
                y=dist.sum(axis=1),
                name="Distance sum",
            ),
            secondary_y=True,
        )
        # fig.add_trace(
        #         go.Scatter(
        #             x=timestamps[::self.step],
        #             y=dist.mean(axis=1),
        #             name="Distance mean",
        #         ),
        #         secondary_y=True,
        # )
        # fig.add_trace(
        #         go.Scatter(
        #             x=timestamps[::self.step],
        #             y=dist.max(axis=1),
        #             name="Greatest difference to a cluster center",
        #         ),
        #         secondary_y=True,
        # )

        # pn = np.array(pd.DataFrame(self.df["PN"].iloc[start_idx:stop_idx])).reshape(-1)

        # fig.add_trace(
        #         go.Scatter(
        #             x=timestamps,
        #             y=pn,
        #             name="PN",
        #         ),
        #         secondary_y=True,
        # )

        fig.update_layout(title_text="Cluster labels over time")
        fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text="Cluster label number", secondary_y=False)
        fig.update_yaxes(title_text="Input unit", secondary_y=True)

        fig.write_html(f"src/templates/prediction.html")
        # fig.write_html(PLOTS_PATH / f"plot_{data_set}.html")

    def plot_cluster_center_distance(self, data_set="train"):

        if data_set == "test":
            start_idx = self.test_start_idx
            stop_idx = self.test_stop_idx
            X = self.test_fingerprints
            timestamps = self.test_fp_timestamps
        elif data_set == "train":
            start_idx = self.train_start_idx
            stop_idx = self.train_stop_idx
            X = self.train_fingerprints
            timestamps = self.train_fp_timestamps
        else:
            raise ValueError("Must specify train or test data set.")

        # original_data = pd.DataFrame(self.df["TN"].iloc[start_idx:stop_idx])
        # original_data["new_index"] = np.linspace(
        #     0, (stop_idx - start_idx) / self.window_size, stop_idx - start_idx
        # )
        # original_data = original_data.set_index("new_index")

        dist = self.model.transform(X)
        dist = dist.sum(axis=1)
        avg_dist = pd.Series(dist).rolling(50).mean()

        plt.figure(figsize=(15, 5))
        # plt.plot(original_data / 40, "--", label="TN", alpha=0.5)
        # plt.plot(input_data[:,0], "--", label="input")
        plt.plot(dist, label="dist")
        plt.plot(avg_dist, label="avg_dist")

        plt.legend()
        plt.plot()

        # print(original_data.iloc[::600].shape)
        # print(dist.shape)
        # print(avg_dist.shape)

        # plot_data = pd.DataFrame([original_data.iloc[::600], dist, avg_dist],
        #         columns=["TN", "dist", "rolling_dist"])

        return dist, avg_dist
        # return dist, avg_dist, original_data

    # def run_analysis(self):

    #     self.build_model(n_clusters=3)
    #     self.create_train_test_set(columns=["Spindle_Torque"])
    #     self.create_fingerprints(window_size=500, overlap=0)
    #     self.fit_predict()
    #     self.visualize_clusters()
    #     self.plot_cluster_center_distance("test")


def validate_params(params):
    """Validate parameters from params.yaml."""

    # Validate featurize parameters
    featurize = params.get("featurize", {})
    if not isinstance(featurize.get("columns"), (str, list)):
        raise ValueError("'columns' must be a string or a list of strings.")
    if not isinstance(featurize.get("convert_timestamp_to_datetime"), bool):
        raise ValueError("'convert_timestamp_to_datetime' must be a boolean.")
    if not isinstance(featurize.get("dataset"), str):
        raise ValueError("'dataset' must be a string.")
    if not isinstance(featurize.get("overlap"), int) or featurize["overlap"] < 0:
        raise ValueError("'overlap' must be a non-negative integer.")
    if not isinstance(featurize.get("timestamp_column"), str):
        raise ValueError("'timestamp_column' must be a string.")
    if not isinstance(featurize.get("window_size"), int) or featurize["window_size"] <= 0:
        raise ValueError("'window_size' must be a positive integer.")

    # Validate postprocess parameters
    postprocess = params.get("postprocess", {})
    if not isinstance(postprocess.get("min_segment_length"), int) or postprocess["min_segment_length"] <= 0:
        raise ValueError("'min_segment_length' must be a positive integer.")

    # Validate train parameters
    train = params.get("train", {})
    if train.get("annotations_dir") not in [None, ""] and not isinstance(train["annotations_dir"], str):
        raise ValueError("'annotations_dir' must be a string or None.")
    if not isinstance(train.get("fix_predefined_centroids"), bool):
        raise ValueError("'fix_predefined_centroids' must be a boolean.")
    if not isinstance(train.get("max_iter"), int) or train["max_iter"] <= 0:
        raise ValueError("'max_iter' must be a positive integer.")
    if not isinstance(train.get("n_clusters"), int) or train["n_clusters"] <= 0:
        raise ValueError("'n_clusters' must be a positive integer.")
    if not isinstance(train.get("use_predefined_centroids"), bool):
        raise ValueError("'use_predefined_centroids' must be a boolean.")

    print("All parameters are valid.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_file", help="data file", required=True)
    parser.add_argument(
        "-t",
        "--timestamp_column_name",
        help="file containing predictions",
        default="Timestamp",
    )
    parser.add_argument(
        "-c", "--column", help="Which column to use", default="OP390_NC_SP_Torque"
    )
    parser.add_argument("-w", "--window_size", help="window size", default=100)
    parser.add_argument("-o", "--overlap", help="overlap", default=0)
    parser.add_argument("-n", "--n_clusters", help="Number of clusters", default=4)

    args = parser.parse_args()

    df = pd.read_csv(args.data_file)

    analysis = Udava(df, timestamp_column_name=args.timestamp_column_name)
    analysis.create_train_test_set(columns=[args.column])
    analysis.create_fingerprints(window_size=args.window_size, overlap=args.overlap)
    # analysis.build_model(n_clusters=int(args.n_clusters))
    # analysis.fit_predict()
    # joblib.dump(analysis.model, "model.pkl")
    analysis.model = joblib.load("model.pkl")
    analysis.predict()
    analysis.visualize_clusters()
    analysis.plot_labels_over_time()
    analysis.plot_cluster_center_distance()
