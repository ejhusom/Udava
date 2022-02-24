#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process time series annotations made by human.

Author:
    Erik Johannes Husom

Created:
    2022-02-23 Wednesday 13:35:06 

"""
import sys

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        annotations["timeserieslabels"].iat[i] = annotations["timeserieslabels"].iloc[i][0]

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

    # categories = []
    # for category in annotations["timeserieslabels"]:
    #     categories.append(category)
    # categories = np.unique(categories)
    categories = np.unique(annotations["timeserieslabels"])

    for category in categories:
    #     for j in range(len(annotations)):
    #         if annotations["timeserieslabels"].iloc[j] == category:
    #             print(category)
    #             print(j)

        current_annotations = annotations[annotations["timeserieslabels"] == category]

        for start, end in zip(current_annotations["start"], current_annotations["end"]):

            current_data = data.loc[start:end]
            # print(current_data)
            features, fingerprint_timestamps = create_fingerprints(current_data, current_data.index, window_size, overlap)


    print(annotations)


if __name__ == '__main__': 

    data_filepath = sys.argv[1]
    annotations_filepath = sys.argv[2]

    data = pd.read_csv(data_filepath, index_col="ts")
    annotations = read_annotations(annotations_filepath)
    create_cluster_centers_from_annotations(data, annotations)

