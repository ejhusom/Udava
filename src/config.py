#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import *
    >>> file = DATA_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2021-11-29 Monday 10:55:44

"""
from pathlib import Path

import matplotlib.colors as mcolors

PARAMS_FILE_PATH = Path("./params.yaml")
"""Path to params file."""

ASSETS_PATH = Path("./assets")
"""Path to all assets of project."""

PROFILE_PATH = ASSETS_PATH / "profile"
"""Path to profiling reports."""

PROFILE_JSON_PATH = PROFILE_PATH / "profile.json"
"""Path to profiling report in JSON format."""

FEATURES_PATH = ASSETS_PATH / "features"
"""Path to files containing input and output features."""

INPUT_FEATURES_PATH = FEATURES_PATH / "input_columns.csv"
"""Path to file containing input features."""

OUTPUT_FEATURES_PATH = FEATURES_PATH / "output_columns.csv"
"""Path to file containing output features."""

DATA_PATH = ASSETS_PATH / "data"
"""Path to data."""

DATA_PATH_RAW = DATA_PATH / "raw"
"""Path to raw data."""

DATA_FEATURIZED_PATH = DATA_PATH / "featurized"
"""Path to data that is has added features."""

FEATURE_VECTORS_PATH = DATA_FEATURIZED_PATH / "featurized.npy"
"""Path to feature vectors of data set."""

DATA_SCALED_PATH = DATA_PATH / "scaled"
"""Path to scaled data."""

ANNOTATIONS_PATH = DATA_PATH / "annotations"
"""Path to annotations and annotations data."""

MODELS_PATH = ASSETS_PATH / "models"
"""Path to models."""

MODELS_FILE_PATH = MODELS_PATH / "model.pkl"
"""Path to model file."""

API_MODELS_PATH = ASSETS_PATH / "models_api.json"

METRICS_PATH = ASSETS_PATH / "metrics"
"""Path to folder containing metrics file."""

METRICS_FILE_PATH = METRICS_PATH / "metrics.json"
"""Path to file containing metrics."""

OUTPUT_PATH = ASSETS_PATH / "output"
"""Path to all output related to the cluster model."""

PREDEFINED_CENTROIDS_PATH = OUTPUT_PATH / "predefined_centroids.json"
"""Path to file containing predefined centroids."""

LABELS_PATH = OUTPUT_PATH / "labels.csv"
"""Path to file containing cluster labels."""

CLUSTER_CENTERS_PATH = OUTPUT_PATH / "cluster_centers.csv"
"""Path to file containing the cluster centers/centroids of the model."""

FEATURE_VECTOR_TIMESTAMPS_PATH = OUTPUT_PATH / "feature_vector_timestamps.npy"
"""Path to file containing the timestamps of the feature vectors."""

ORIGINAL_TIME_SERIES_PATH = OUTPUT_PATH / "original_data.csv"
"""Path to file containing the original input itme series combined into one file."""

PREDICTIONS_PATH = ASSETS_PATH / "predictions"
"""Path to folder containing predictions file."""

PREDICTIONS_FILE_PATH = PREDICTIONS_PATH / "labels.csv"
"""Path to file containing predictions."""

PLOTS_PATH = ASSETS_PATH / "plots"
"""Path to folder plots."""

SCALER_PATH = ASSETS_PATH / "scalers"
"""Path to folder containing scalers."""

INPUT_SCALER_PATH = SCALER_PATH / "input_scaler.z"
"""Path to input scaler."""

OUTPUT_SCALER_PATH = SCALER_PATH / "output_scaler.z"
"""Path to output scaler."""

# FEATURE_NAMES = ["mean", "median", "std", "var", "minmax", "frequency", "gradient"]
FEATURE_NAMES = ["mean", "median", "std", "minmax", "frequency", "gradient"]
# FEATURE_NAMES = ["mean", "median", "std", "frequency", "gradient"]

COLORS = [
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

COLORS += list(mcolors.CSS4_COLORS.keys())
