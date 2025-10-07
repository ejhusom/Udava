"""
Visualization utilities for UDAVA.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

def visualize_clusters(train_fingerprints, train_labels, dim1=0, dim2=1, width=10, height=10):
    plt.figure(figsize=(width, height))
    scatter = plt.scatter(train_fingerprints[:, dim1], train_fingerprints[:, dim2], c=train_labels, cmap='viridis')
    plt.xlabel(f'Feature {dim1}')
    plt.ylabel(f'Feature {dim2}')
    plt.title('Cluster Visualization')
    plt.colorbar(scatter)
    plt.show()
    logging.info("Cluster visualization displayed.")

def plot_labels_over_time(labels, timestamps):
    plt.figure(figsize=(15, 5))
    plt.plot(timestamps, labels, label='Labels')
    plt.xlabel('Timestamp')
    plt.ylabel('Cluster Label')
    plt.title('Labels Over Time')
    plt.legend()
    plt.show()
    logging.info("Labels over time plot displayed.")
