#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process time series annotations made by human.

Author:
    Erik Johannes Husom

Created:
    2022-02-23 Wednesday 13:35:06 

"""
import json
import sys

import joblib
import matplotlib.pyplot as plt

# import plotext as plt
import numpy as np
import pandas as pd

from config import *
from featurize import *


def read_annotations(filepath, verbose=False):
    """Read annotations from file.

    Args:
        filepath (str): Filepath to annotations file.

    Returns:
        annotations (DataFrame): Annotations in a Pandas DataFrame.

    """

    df = pd.read_json(filepath)
    annotations = pd.json_normalize(df["label"].iloc[0])

    # The column 'timeserieslabels' is originally a list of one string, and is
    # changed to be just a string to make it easier to process.
    for i in range(len(annotations)):
        annotations["timeserieslabels"].iat[i] = annotations["timeserieslabels"].iloc[
            i
        ][0]

    if verbose:
        print(annotations)

    return annotations


def create_cluster_centers_from_annotations(data, annotations):

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    dataset = params["featurize"]["dataset"]
    columns = params["featurize"]["columns"]
    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    timestamp_column = params["featurize"]["timestamp_column"]

    # for col in data.columns:
    #     if col not in columns:
    #         del data[col]

    #     # Remove feature if it is non-numeric
    #     elif not is_numeric_dtype(data[col]):
    #         del data[col]

    scaler = joblib.load(INPUT_SCALER_PATH)

    categories = np.unique(annotations["timeserieslabels"])
    cluster_centers = {}

    for category in categories:

        current_annotations = annotations[annotations["timeserieslabels"] == category]

        features_list = []

        for start, end in zip(current_annotations["start"], current_annotations["end"]):

            # Select the data covered by the current annotation.
            current_data = data.loc[start:end]

            # plt.figure()
            # plt.plot(current_data)
            # plt.savefig("plot.png")
            # plt.show()

            # Featurize the current data.
            features, feature_vector_timestamps = create_feature_vectors(
                current_data, current_data.index, window_size, overlap
            )

            features_list.append(features)

        # Combine all the data that is annotated with the same category.
        combined_feature_vectors = np.concatenate(features_list, axis=0)

        # Scale the feature vectors with the scaler already fitted on the full data
        # set (not only the data used for annotations).
        scaled_feature_vectors = scaler.transform(combined_feature_vectors)

        # Take the average of the feature vectors, to find a representative
        # feature vector for the current category.
        average_feature_vector = np.average(scaled_feature_vectors, axis=0)
        cluster_centers[category] = list(average_feature_vector)

    print(cluster_centers)
    return cluster_centers


if __name__ == "__main__":

    data_filepath = sys.argv[1]
    annotations_filepath = sys.argv[2]

    data = pd.read_csv(data_filepath, index_col="ts")
    annotations = read_annotations(annotations_filepath)
    create_cluster_centers_from_annotations(data, annotations)
