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
import pycatch22

# import tsfresh
import yaml
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        inference_df (DataFrame): A data frame to run inference.

    """

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    dataset = params["featurize"]["dataset"]
    columns = params["featurize"]["columns"]
    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    timestamp_column = params["featurize"]["timestamp_column"]
    convert_timestamp_to_datetime = params["featurize"]["convert_timestamp_to_datetime"]

    # If the timestamp column does not have a name, we use the default name
    # given to unnamed columns by Pandas.
    if timestamp_column == "":
        timestamp_column = "Unnamed: 0"

    # If no name of data set is given, all csv-files present in 'assets/data/raw'
    # will be used.
    if dataset != None:
        dir_path += "/" + dataset

    if inference:
        featurized_df = _featurize(
            inference_df, columns, window_size, overlap, timestamp_column
        )

        return featurized_df

    else:
        filepaths = find_files(dir_path, file_extension=".csv")

        dfs = []
        featurized_dfs = []

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)
        SCALER_PATH.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)

        for filepath in filepaths:

            # Read csv. If no timestamp column name is given in the parameters,
            # the timestamp column will be assumed to be the first one.
            if timestamp_column == None:
                df = pd.read_csv(filepath, index_col=0)
            else:
                df = pd.read_csv(filepath)
                df = df.set_index(timestamp_column)

            # This needs to be set as a configuration parameter to avoid having
            # indeces being interpreted as UNIX timestamps.
            if convert_timestamp_to_datetime:
                # Attempt to convert timestamps to datetime
                try:
                    df.index = pd.to_datetime(df.index)
                    # Convert to UNIX time
                    # df.index = df.index.astype(np.int64) // 1e-9
                except:
                    pass

            featurized_df = _featurize(
                df, columns, window_size, overlap, timestamp_column
            )

            dfs.append(df)
            featurized_dfs.append(featurized_df)

        combined_df = pd.concat(dfs)
        combined_featurized_df = pd.concat(featurized_dfs)
        fp_timestamps = combined_featurized_df.index

        # FIXME: A problem arises when the input consists of multiple files
        # with no timestamps, i. e. the indeces are overlapping.
        # if not convert_timestamp_to_datetime:
        #     combined_df.reset_index(drop=True, inplace=True)
        # combined_featurized_df.reset_index(drop=True, inplace=True)
        # combined_featurized_df.index *= window_size

        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # Save the timestamps for each feature_vector, in order to use it for
        # plotting later.
        np.save(FEATURE_VECTOR_TIMESTAMPS_PATH, fp_timestamps)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined_featurized_df.to_numpy())
        joblib.dump(scaler, INPUT_SCALER_PATH)

        combined_df.to_csv(ORIGINAL_TIME_SERIES_PATH)
        np.save(FEATURE_VECTORS_PATH, scaled)


def _featurize(df, columns, window_size, overlap, timestamp_column):
    """Process individual dataframes.

    Args:
        df (DataFrame): Dataframe to create features from.
        columns (list): Which features/columns to create features from.
        window_size (int): Size of window/subsequences.
        overlap (int): Overlap between windows.
        timestamp_column (str): Name of timestamp column.

    Returns:
        df (DataFrame): Dataframe with features created from original
            dataframe.

    """

    # If no features are specified, use all columns as features
    if type(columns) is str:
        columns = [columns]

    # Check if wanted features from params.yaml exists in the data
    for column in columns:
        if column not in df.columns:
            print(f"Column {column} not found!")

    for col in df.columns:
        if col not in columns:
            del df[col]

        # Remove feature if it is non-numeric
        elif not is_numeric_dtype(df[col]):
            del df[col]

    features, feature_vector_timestamps = create_feature_vectors(
        df, df.index, window_size, overlap
    )

    df = pd.DataFrame(features, index=feature_vector_timestamps)

    return df


def create_feature_vectors(df, timestamps, window_size, overlap, mode="standard"):
    """Create feature_vectors of time series data.

    The feature vector is based on statistical properties.

    Args:
        df (DataFrame): The data frame to create feature_vectors from.
        timestamps (Series): The timestamps corresponding to the time series in
            df.
        window_size (int): Number of time steps to include when calculating
            a feature_vector.
        overlap (int): How much overlap between windows of the time series
            data.

    Returns:
        feature_vectors (Numpy array): An array of feature_vectors for the time
            series data. The array has size n*m, where n is the number of
            feature_vectors (number of subsequences), and m is the number of
            features in the feature vector.

    """

    n_input_columns = df.shape[1]
    n_rows_raw = df.shape[0]
    n_rows = n_rows_raw // window_size
    step = window_size - overlap
    feature_vector_timestamps = []

    if mode == "catch22":
        n_features = 24
        features = np.zeros((n_rows, n_features, n_input_columns))

        # Loop through all observations and calculate features within window
        for i in range(n_rows):
            start = i * step
            stop = start + window_size

            window = np.array(df.iloc[start:stop, :])
            feature_vector_timestamps.append(timestamps[stop - (step // 2)])

            for j in range(n_input_columns):
                features[i, :, j] = pycatch22.catch22_all(window, catch24=True)[
                    "values"
                ]

    # elif mode == "tsfresh":
    #     features = []

    #     # Loop through all observations and calculate features within window
    #     for i in range(n_rows):
    #         start = i * step
    #         stop = start + window_size

    #         window = df.iloc[start:stop, :].copy()
    #         # tsfresh requires an ID column. Calculates features on data points
    #         # with the same ID, which is why we put the same ID on all rows of
    #         # the window.
    #         window["id"] = 0

    #         feature_vector_timestamps.append(timestamps[stop - (step // 2)])

    #         for j in range(n_input_columns):
    #             # features[i, :, j] = tsfresh.extract_features(window, column_id="id")
    #             extracted_features = tsfresh.extract_features(window,
    #                     column_id="id", disable_progressbar=True)
    #             extracted_features = np.array(extracted_features).flatten()
    #             # print(extracted_features)
    #             # print(extracted_features.shape)
    #             features.append(np.array(extracted_features))

    #     print("======= Extracted features ==========")
    #     extracted_features = tsfresh.extract_features(window,
    #             column_id="id", disable_progressbar=True)
    #     print(extracted_features.columns)
    #     extracted_features.to_csv("yo.csv")
    #     return 0
    #     n_features = len(features[-1])
    #     features = np.array(features)
    #     print(features)
    #     print(features.shape)
    #     print("-----------------------------")
    #     # print(extracted_features.info())
    else:
        n_features = 7

        # Initialize descriptive feature matrices
        mean = np.zeros((n_rows, n_input_columns))
        median = np.zeros((n_rows, n_input_columns))
        std = np.zeros((n_rows, n_input_columns))
        # rms = np.zeros((n_rows, n_input_columns))
        var = np.zeros((n_rows, n_input_columns))
        minmax = np.zeros((n_rows, n_input_columns))
        frequency = np.zeros((n_rows, n_input_columns))
        gradient = np.zeros((n_rows, n_input_columns))

        # Loop through all observations and calculate features within window
        for i in range(n_rows):
            start = i * step
            stop = start + window_size

            window = np.array(df.iloc[start:stop, :])
            feature_vector_timestamps.append(timestamps[stop - (step // 2)])

            mean[i, :] = np.mean(window, axis=0)
            median[i, :] = np.median(window, axis=0)
            std[i, :] = np.std(window, axis=0)
            # rms[i, :] = np.sqrt(np.mean(np.square(window, axis=0)))
            var[i, :] = np.var(window, axis=0)
            minmax[i, :] = np.max((window), axis=0) - np.min((window), axis=0)
            frequency[i, :] = np.linalg.norm(np.fft.rfft(window, axis=0), axis=0, ord=2)
            gradient[i, :] = np.mean(np.gradient(window, axis=0))

        features = np.concatenate(
            (mean, median, std, var, minmax, frequency, gradient), axis=1
        )

    features = np.nan_to_num(features)
    features = features.reshape(n_rows, n_features * n_input_columns)

    # feature_vector_timestamps = timestamps[::step]
    feature_vector_timestamps = pd.Index(
        feature_vector_timestamps, dtype=object, name="Date"
    )

    return features, feature_vector_timestamps


if __name__ == "__main__":

    np.random.seed(2020)

    featurize(sys.argv[1])
