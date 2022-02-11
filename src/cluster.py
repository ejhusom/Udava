#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clustering for historical data validation.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 12:05:02

"""
import sys
import yaml

import numpy as np
import pandas as pd
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

from config import *
from preprocess_utils import find_files, move_column

def cluster(dir_path=""):

    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    learning_method = params["cluster"]["learning_method"]
    n_clusters = params["cluster"]["n_clusters"]
    max_iter = params["cluster"]["max_iter"]

    filepaths = find_files(dir_path, file_extension=".npy")
    fingerprints = np.load(filepaths[0])

    model = build_model(learning_method, n_clusters, max_iter)
    labels, model = fit_predict(fingerprints, model)
    distances_to_centers, sum_distance_to_centers = calculate_distances(fingerprints, model)

    MODELS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(OUTPUT_PATH / "labels.csv")
    pd.DataFrame(fingerprints).to_csv(OUTPUT_PATH / "fingerprints.csv")

    dump(model, MODELS_FILE_PATH)

    cluster_names = generate_cluster_names(model)
    pd.DataFrame(cluster_names).to_csv(OUTPUT_PATH / "cluster_names.csv")


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


def fit_predict(fingerprints, model):

    labels = model.fit_predict(fingerprints)

    return labels, model

def predict(fingerprints, model):

    labels = model.predict(fingerprints)

    return labels

def calculate_distances(fingerprints, model):

    distances_to_centers = model.transform(fingerprints)
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
        cluster_names.append(str(i))

    print(model.cluster_centers_)

    maxs = model.cluster_centers_.argmax(axis=0)
    mins = model.cluster_centers_.argmin(axis=0)

    for i in range(len(FEATURE_NAMES)):
        cluster_names[maxs[i]] += "_highest_" + FEATURE_NAMES[i]
        cluster_names[mins[i]] += "_lowest_" + FEATURE_NAMES[i]

    # if any(c == "" for c in cluster_names):
    #     for i in range(len(FEATURE_NAMES)):
    #         cluster_names[maxs[i]] += "highest_" + FEATURE_NAMES[i] + "_"
    #         cluster_names[mins[i]] += "lowest_" + FEATURE_NAMES[i] + "_"

    return cluster_names


if __name__ == '__main__':

    cluster(sys.argv[1])
