#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for handling cluster model during inference using the Udava API.

Author:
    Erik Johannes Husom

Created:
    2021-12-13 Monday 15:58:42 

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
from featurize import *
from cluster import *
from evaluate import *
from preprocess_utils import find_files, move_column


class ClusterModel:

    def __init__(
        self,
        params_file=PARAMS_FILE_PATH,
        input_scaler_file=INPUT_SCALER_PATH,
        model_file=MODELS_FILE_PATH,
        verbose=True,
    ):

        if type(params_file) == dict:
            yaml.dump(params_file, open("params.yaml", "w"),
                    allow_unicode=True)
            self.params_file = "params.yaml"
        else:
            self.params_file = params_file

        self.input_scaler_file = input_scaler_file
        self.model_file = model_file
        self.verbose = verbose

        self.assets_files = [
            self.params_file,
            self.input_scaler_file,
            self.model_file,
        ]

        self._check_assets_existence()

    def _check_assets_existence(self):
        """Check if the needed assets exists."""

        check_ok = True

        for path in self.assets_files:
            if not os.path.exists(path):
                print(f"File {path} not found.")
                check_ok = False

        assert check_ok, "Assets missing."

    def run_cluster_model(self, inference_df):
        """Run cluster model.

        Args:

        """

        params = yaml.safe_load(open("params.yaml"))
        learning_method = params["cluster"]["learning_method"]
        max_iter = params["cluster"]["max_iter"]
        n_clusters = params["cluster"]["n_clusters"]
        columns = params["featurize"]["columns"]
        dataset = params["featurize"]["dataset"]
        timestamp_column = params["featurize"]["timestamp_column"]
        window_size = params["featurize"]["window_size"]
        overlap = params["featurize"]["overlap"]

        featurized_df = featurize(inference=True, inference_df=inference_df)
        fingerprint_timestamps = featurized_df.index

        fingerprints = featurized_df.to_numpy()
        print(fingerprints)
        input_scaler = joblib.load(INPUT_SCALER_PATH)
        fingerprints = input_scaler.transform(fingerprints)

        model = joblib.load(MODELS_FILE_PATH)
        labels = model.predict(fingerprints)
        distance_to_centers, sum_distance_to_centers = calculate_distances(fingerprints, model)

        plt.figure()
        plt.plot(labels)
        plt.show()

        visualize_clusters(labels, fingerprints, model)
        plot_labels_over_time(fingerprint_timestamps, labels, fingerprints,
                inference_df, model)
        plot_cluster_center_distance(fingerprint_timestamps, fingerprints, model)

        return fingerprint_timestamps, labels, sum_distance_to_centers
