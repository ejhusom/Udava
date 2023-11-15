#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
import sys

import numpy as np
import pandas as pd

filepath = sys.argv[1]

df = pd.read_csv(filepath)
del df["timestamp.1"]
df.index = df["timestamp"]
del df["timestamp"]
print(df.info())

pd.options.plotting.backend = "plotly"

fig = df.plot()
fig.show()
