"""
Data handling utilities for UDAVA.
"""
import pandas as pd
import logging

class DataHandler:
    def __init__(self, df: pd.DataFrame, timestamp_column_name: str = "Timestamp"):
        self.df = df.copy()
        self.timestamp_column_name = timestamp_column_name
        self._validate_df()
        self._ensure_timestamp_column()

    def _validate_df(self):
        if self.df.empty:
            logging.error("Input DataFrame is empty. Please provide a valid DataFrame.")
            raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")

    def _ensure_timestamp_column(self):
        if self.timestamp_column_name not in self.df.columns:
            logging.warning(f"'{self.timestamp_column_name}' column not found. Using row indices as timestamps.")
            self.df[self.timestamp_column_name] = self.df.index

    def get_train_test_split(self, columns, train_start_idx=0, train_stop_idx=None, test_start_idx=0, test_stop_idx=None):
        train_stop_idx = self.df.shape[0] - 1 if train_stop_idx is None else train_stop_idx
        test_stop_idx = self.df.shape[0] - 1 if test_stop_idx is None else test_stop_idx
        train_set = self.df[columns].iloc[train_start_idx:train_stop_idx]
        test_set = self.df[columns].iloc[test_start_idx:test_stop_idx]
        train_timestamps = self.df[self.timestamp_column_name].iloc[train_start_idx:train_stop_idx]
        test_timestamps = self.df[self.timestamp_column_name].iloc[test_start_idx:test_stop_idx]
        return train_set, test_set, train_timestamps, test_timestamps
