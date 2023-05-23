#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for producing natural language explanation and reasoning about the
model.

Author:
    Erik Johannes Husom

Created:
    2023-04-19 onsdag 13:26:49 

"""
import yaml
import pandas as pd
import numpy as np
import openai

from config import *

def explain():

    # Load parameters
    with open("params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)

    dataset = params["featurize"]["dataset"]
    columns = params["featurize"]["columns"]
    window_size = params["featurize"]["window_size"]
    overlap = params["featurize"]["overlap"]
    timestamp_column = params["featurize"]["timestamp_column"]
    convert_timestamp_to_datetime = params["featurize"]["convert_timestamp_to_datetime"]
    learning_method = params["train"]["learning_method"]
    dataset_description = params["explain"]["dataset_description"]
    
    cluster_names = pd.read_csv(OUTPUT_PATH / "cluster_names.csv", index_col=0)

    # Make a list of the cluster names
    cluster_names = list(cluster_names.cluster_name)
    print(cluster_names)


    # Generate prompt for LLM model, based on the parameters
    user_message = f"""\
The dataset is called {dataset}.
The description of the dataset is as follows: {dataset_description}.
The dataset contains the following columns: {columns}.
The window size is {window_size}.
The overlap is {overlap}.
The learning method is {learning_method}.
The names of the cluster are as follows:
"""

    for cluster_name in cluster_names:
        user_message += "- " + cluster_name + "\n"

    user_message += ""

    print(user_message)

    system_message = f"""\
You are an AI assistant that is trying to help a user understand the output
from a machine learning (ML) pipeline. The ML pipeline analyzes a dataset
consisting of time series data, and uses unsupervised clearning (clustering) to
label the data."""
# clusters the data into different groups.
# The time series is divided into windows, or sub-sequences, and we compute
# statistical features 

    # openai.ChatCompletion.create(
    #   model="gpt-3.5-turbo",
    #   messages=[
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_message},
    #         # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    #         # {"role": "user", "content": "Where was it played?"}
    #     ]
    # )

if __name__ == '__main__':
    explain()
