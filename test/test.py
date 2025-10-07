#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test UDAVA.

Author:
    Erik Johannes Husom

Created:
    2022-06-09 torsdag 13:44:41 

"""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append("src/")
import train
import cluster_utils
from src.udava import Udava, validate_params


class TestUDAVA(unittest.TestCase):
    """Various tests for UDAVA pipeline."""

    def test_find_segments(self):
        """Test whether find_segments() returns expected results."""

        labels = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2])
        segments = cluster_utils.find_segments(labels)

        expected_segments = np.array(
            [[0, 0, 2, 0, 1], [1, 1, 3, 2, 4], [2, 0, 4, 5, 8], [3, 2, 3, 9, 11]]
        )

        print(expected_segments)
        print(segments)

        np.testing.assert_array_equal(segments, expected_segments)

    def test_find_segments_single_label(self):
        """Test find_segments() with a single label."""

        labels = np.array([1, 1, 1, 1, 1])
        segments = cluster_utils.find_segments(labels)

        expected_segments = np.array([[0, 1, 5, 0, 4]])

        np.testing.assert_array_equal(segments, expected_segments)

    def test_find_segments_alternating_labels(self):
        """Test find_segments() with alternating labels."""

        labels = np.array([0, 1, 0, 1, 0])
        segments = cluster_utils.find_segments(labels)

        expected_segments = np.array(
            [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [2, 0, 1, 2, 2], [3, 1, 1, 3, 3], [4, 0, 1, 4, 4]]
        )

        np.testing.assert_array_equal(segments, expected_segments)

    def test_find_segments_empty(self):
        """Test find_segments() with an empty array."""

        labels = np.array([])
        segments = cluster_utils.find_segments(labels)

        expected_segments = np.array([])

        np.testing.assert_array_equal(segments, expected_segments)

    def test_create_train_test_set_empty(self):
        """Test create_train_test_set() with an empty DataFrame."""

        df = pd.DataFrame(columns=["col1", "col2"])

        with self.assertRaises(ValueError):
            Udava(df)

    def test_create_fingerprints_invalid_window(self):
        """Test create_fingerprints() with an invalid window size."""

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        udava_instance = Udava(df)
        udava_instance.create_train_test_set(["col1", "col2"])

        with self.assertRaises(ValueError):
            udava_instance.create_fingerprints(window_size=0)

    def test_build_model_invalid_method(self):
        """Test build_model() with an invalid clustering method."""

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        udava_instance = Udava(df)

        with self.assertRaises(ValueError):
            udava_instance.build_model(method="invalid_method")

    def test_udava_with_timestamp_column(self):
        """Test Udava class with a Timestamp column."""

        df = pd.DataFrame({"Timestamp": [1, 2, 3], "col1": [4, 5, 6]})
        udava_instance = Udava(df)

        self.assertIn("Timestamp", udava_instance.df.columns)
        self.assertEqual(list(udava_instance.df["Timestamp"]), [1, 2, 3])

    def test_udava_without_timestamp_column(self):
        """Test Udava class without a Timestamp column."""

        df = pd.DataFrame({"col1": [4, 5, 6]})
        udava_instance = Udava(df)

        self.assertIn("Timestamp", udava_instance.df.columns)
        self.assertEqual(list(udava_instance.df["Timestamp"]), [0, 1, 2])

    def test_validate_params(self):
        """Test the validate_params function with valid and invalid parameters."""

        valid_params = {
            "featurize": {
                "columns": "ValueDiff",
                "convert_timestamp_to_datetime": True,
                "dataset": "meter_1274",
                "overlap": 0,
                "timestamp_column": "ResultTimeStamp",
                "window_size": 30,
            },
            "postprocess": {
                "min_segment_length": 1,
            },
            "train": {
                "annotations_dir": None,
                "fix_predefined_centroids": False,
                "learning_method": "minibatchkmeans",
                "max_iter": 100,
                "n_clusters": 7,
                "use_predefined_centroids": False,
            },
        }

        # Should not raise any errors
        validate_params(valid_params)

        invalid_params = valid_params.copy()
        invalid_params["featurize"]["window_size"] = -1

        with self.assertRaises(ValueError):
            validate_params(invalid_params)

        def test_validate_params_missing_fields(self):
            """Test validate_params with missing required fields."""
            params = {
                "featurize": {
                    # Missing 'columns'
                    "convert_timestamp_to_datetime": True,
                    "dataset": "meter_1274",
                    "overlap": 0,
                    "timestamp_column": "ResultTimeStamp",
                    "window_size": 30,
                },
                "postprocess": {
                    "min_segment_length": 1,
                },
                "train": {
                    "annotations_dir": None,
                    "fix_predefined_centroids": False,
                    "learning_method": "minibatchkmeans",
                    "max_iter": 100,
                    "n_clusters": 7,
                    "use_predefined_centroids": False,
                },
            }
            with self.assertRaises(ValueError):
                validate_params(params)

        def test_validate_params_invalid_types(self):
            """Test validate_params with invalid types for parameters."""
            params = {
                "featurize": {
                    "columns": 123,  # Should be str or list
                    "convert_timestamp_to_datetime": "True",  # Should be bool
                    "dataset": 456,  # Should be str
                    "overlap": -1,  # Should be non-negative int
                    "timestamp_column": 789,  # Should be str
                    "window_size": "thirty",  # Should be int
                },
                "postprocess": {
                    "min_segment_length": "one",  # Should be int
                },
                "train": {
                    "annotations_dir": 123,  # Should be str or None
                    "fix_predefined_centroids": "False",  # Should be bool
                    "learning_method": "minibatchkmeans",
                    "max_iter": "hundred",  # Should be int
                    "n_clusters": 0,  # Should be positive int
                    "use_predefined_centroids": "False",  # Should be bool
                },
            }
            with self.assertRaises(ValueError):
                validate_params(params)

        def test_validate_params_edge_cases(self):
            """Test validate_params with edge cases like empty dicts and None values."""
            params = {}
            with self.assertRaises(ValueError):
                validate_params(params)

            params = {
                "featurize": None,
                "postprocess": None,
                "train": None,
            }
            with self.assertRaises(Exception):
                validate_params(params)


if __name__ == "__main__":

    unittest.main()
