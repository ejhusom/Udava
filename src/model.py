"""
Model management utilities for UDAVA.
"""
import logging
from sklearn.cluster import MeanShift, MiniBatchKMeans
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, method="minibatchkmeans", n_clusters=2, max_iter=100):
        valid_methods = ["meanshift", "minibatchkmeans"]
        if method not in valid_methods:
            logging.error(f"Invalid method '{method}'. Supported methods are: {valid_methods}")
            raise ValueError(f"Invalid method '{method}'. Supported methods are: {valid_methods}")
        if method == "meanshift":
            self.model = MeanShift()
        else:
            self.model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter)
        logging.info(f"Model built using method: {method}, n_clusters: {n_clusters}, max_iter: {max_iter}")
        return self.model

    def load_model(self, filepath):
        try:
            self.model = joblib.load(filepath)
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Failed to load model from {filepath}: {e}")
            raise

    def fit_predict(self, train_fingerprints, test_fingerprints):
        if self.model is None:
            logging.info("No model found, building a new one.")
            self.build_model()
        try:
            train_labels = self.model.fit_predict(train_fingerprints)
            test_labels = self.model.predict(test_fingerprints)
            clusters = np.unique(train_labels)
            train_dist = self.model.transform(train_fingerprints)
            test_dist = self.model.transform(test_fingerprints)
            train_dist_sum = train_dist.sum(axis=1)
            test_dist_sum = test_dist.sum(axis=1)
            logging.info("Model fit and predictions completed.")
            return train_labels, test_labels, clusters, train_dist, test_dist, train_dist_sum, test_dist_sum
        except Exception as e:
            logging.error(f"Error during fit_predict: {e}")
            raise

    def scale_fingerprints(self, train_fingerprints, test_fingerprints):
        train_fingerprints = self.scaler.fit_transform(train_fingerprints)
        test_fingerprints = self.scaler.transform(test_fingerprints)
        return train_fingerprints, test_fingerprints
