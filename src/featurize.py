#!/usr/bin/env python3
"""Create summary statistics from time series data.

Author:
    Erik Johannes Husom

Date:
    2021-11-29 Monday 11:31:22 

"""
import json
import os
import sys
import joblib

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml

from config import *
from preprocess_utils import find_files, move_column

def featurize(dir_path="", inference=False, inference_df=None):
    """Create vectors of summary statistics based on sliding windows across
    time series data.

    Args:
        dir_path (str): Path to directory containing files.
        inference (bool): When creating a virtual sensor, the
            results should be saved to file for more efficient reruns of the
            pipeline. When running the virtual sensor, there is no need to save
            these intermediate results to file.

    """

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    dataset = params["featurize"]["dataset"]
    columns = params["featurize"]["columns"]
    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    timestamp_column = params["featurize"]["timestamp_column"]
    print(columns)

    # If no name of data set is given, all files present in 'assets/data/raw'
    # will be used.
    if dataset != None:
        dir_path += "/" + dataset

    if inference:
        featurized_df = _featurize(
            inference_df,
            columns,
            window_size,
            overlap,
            timestamp_column
        )

        return featurized_df

    else:
        filepaths = find_files(dir_path, file_extension=".csv")

        dfs = []
        featurized_dfs = []
        # timestamps = np.array([])

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)
        SCALER_PATH.mkdir(parents=True, exist_ok=True)

        for filepath in filepaths:

            # Read csv
            df = pd.read_csv(filepath, index_col=0)

            # timestamps = np.concatenate([timestamps, df.index])

            featurized_df = _featurize(df, columns, window_size, overlap,
                    timestamp_column)


            dfs.append(df)
            featurized_dfs.append(featurized_df)

        combined_df = pd.concat(dfs)
        combined_featurized_df = pd.concat(featurized_dfs)
        fp_timestamps = combined_featurized_df.index

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # Save the timestamps for each fingerprint, in order to use it for
        # plotting later.
        np.save(
                OUTPUT_PATH / "fingerprint_timestamps.npy",
                fp_timestamps
        )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined_featurized_df.to_numpy())
        joblib.dump(scaler, INPUT_SCALER_PATH)

        combined_df.to_csv(OUTPUT_PATH / "combined.csv")
        np.save(DATA_FEATURIZED_PATH / "featurized.npy", scaled)


def _featurize(df, columns, window_size, overlap, timestamp_column):
    """Process individual DataFrames."""

    # If no features are specified, use all columns as features
    if type(columns) is str:
        columns = [columns]
    # if not isinstance(columns, list):
    #     columns = df.columns

    print(columns)

    # Check if wanted features from params.yaml exists in the data
    for column in columns:
        if column not in df.columns:
            print(f"Column {column} not found!")

    for col in df.columns:
        # Remove feature from input. This is useful in the case that a raw
        # feature is used to engineer a feature, but the raw feature itself
        # should not be a part of the input.
        if col not in columns:
            del df[col]

        # Remove feature if it is non-numeric
        elif not is_numeric_dtype(df[col]):
            del df[col]

    features, fingerprint_timestamps = _create_fingerprints(df,
            df.index,
            window_size, overlap)

    df = pd.DataFrame(features,
            index=fingerprint_timestamps[-len(features):])

    # return features, fingerprint_timestamps
    return df


def _create_fingerprints(df, timestamps, window_size, overlap):
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

    feature_names = [
            "mean",
            "median",
            "std",
            # "rms",
            "var",
            "minmax",
            "frequency"
    ]

    # Initialize descriptive feature matrices
    mean = np.zeros((n_rows - 1, n_features))
    median = np.zeros((n_rows - 1, n_features))
    std = np.zeros((n_rows - 1, n_features))
    # rms = np.zeros((n_rows - 1, n_features))
    var = np.zeros((n_rows - 1, n_features))
    minmax = np.zeros((n_rows - 1, n_features))
    frequency = np.zeros((n_rows - 1, n_features))

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

    features = np.concatenate(
        (mean, median, std, var, minmax, frequency),
        axis=1
    )
    # cfp = np.nan_to_num(cfp)

    # features = cfp.reshape(n_rows - 1, 22*n_features)
    
    # print(f"Mean shape: {mean.shape}")
    # print(f"cfp shape: {cfp.shape}")
    # print(f"Features shape: {features.shape}")

    fingerprint_timestamps = timestamps[::step]

    return features, fingerprint_timestamps

if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])
