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


if __name__ == "__main__":

    unittest.main()
