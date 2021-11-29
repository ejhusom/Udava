#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use Udava for inference.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 14:40:34 

"""

from udava import Udava

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_file', help='data file', required=True)
    parser.add_argument('-t', '--timestamp_column_name',
            help='file containing predictions', default="Timestamp")
    parser.add_argument('-n', '--column', help='Which column to use',
            default="OP390_NC_SP_Torque")
    parser.add_argument('-w', '--window_size', help='window size', default=100)
    parser.add_argument('-o', '--overlap', help='overlap', default=0)
    parser.add_argument('-c', '--n_clusters', help='Number of clusters', default=4)

    args = parser.parse_args()

    df = pd.read_csv(args.data_file)

    analysis = Udava(df, timestamp_column_name=args.timestamp_column_name)
    analysis.create_train_test_set(columns=[args.column])
    analysis.create_fingerprints(window_size=args.window_size,
            overlap=args.overlap)
    analysis.load_model("model.pkl")
    analysis.predict()
    analysis.visualize_clusters()
    analysis.plot_labels_over_time()
    analysis.plot_cluster_center_distance()
