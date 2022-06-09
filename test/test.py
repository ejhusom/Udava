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
import cluster


class TestUDAVA(unittest.TestCase):
    """Various tests for UDAVA pipeline."""

    def test_find_segments(self):
        """Test whether find_segments() returns expected results."""

        labels = [0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2]
        segments = cluster.find_segments(labels)

        expected_segments = np.array(
            [[0, 0, 2, 0, 1], [1, 1, 3, 2, 4], [2, 0, 4, 5, 8], [3, 2, 3, 9, 11]]
        )

        print(expected_segments)
        print(segments)

        np.testing.assert_array_equal(segments, expected_segments)


if __name__ == "__main__":

    unittest.main()
