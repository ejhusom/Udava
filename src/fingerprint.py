"""
Fingerprint creation utilities for UDAVA.
"""
import numpy as np
import pandas as pd
import logging

def create_fingerprints(df: pd.DataFrame, timestamps: pd.Series, window_size: int, overlap: int):
    """
    Create fingerprints of time series data.
    Args:
        df (DataFrame): The data frame to create fingerprints from.
        timestamps (Series): The timestamps corresponding to the time series in df.
        window_size (int): Number of time steps to include when calculating a fingerprint.
        overlap (int): How much overlap between windows of the time series data.
    Returns:
        fingerprints (np.ndarray): Array of fingerprints for the time series data.
        fingerprint_timestamps (np.ndarray): Array of timestamps for each fingerprint.
    """
    n_features = df.shape[1]
    n_rows_raw = df.shape[0]
    step = window_size - overlap
    if step <= 0:
        logging.error("Step size must be positive.")
        raise ValueError("Step size must be positive.")
    fingerprints = []
    fingerprint_timestamps = []
    for start in range(0, n_rows_raw - window_size + 1, step):
        window = df.iloc[start:start+window_size]
        ts = timestamps.iloc[start+window_size//2]
        # Example: mean and std for each feature
        features = np.concatenate([window.mean().values, window.std().values])
        fingerprints.append(features)
        fingerprint_timestamps.append(ts)
    return np.array(fingerprints), np.array(fingerprint_timestamps)
