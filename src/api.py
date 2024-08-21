#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API for Udava.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 14:48:42 

"""
import datetime
import json
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import flask
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import requests
import yaml
from flask_restful import Api, Resource, reqparse
from plotly.subplots import make_subplots

from clustermodel import ClusterModel
from cluster_utils import create_event_log
from config import API_MODELS_PATH, DATA_PATH_RAW, METRICS_FILE_PATH, LABELS_PATH, PLOTS_PATH, OUTPUT_PATH
from postprocess import event_log_score
from udava import Udava

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True


@app.route("/")
def home():
    """Home page."""

    return flask.render_template("index.html")


@app.route("/create_model_form")
def create_model_form():
    """Create model form."""

    models = get_models()

    return flask.render_template(
        "create_model_form.html", length=len(models), models=models
    )


@app.route("/inference")
def inference():
    """Inference page."""

    models = get_models()

    return flask.render_template("inference.html", models=models)


@app.route("/inference_result")
def inference_result():
    """Inference result page."""

    models = get_models()

    return flask.render_template("inference.html", models=models, show_prediction=True)


@app.route("/result")
def result(plot_div):
    """Result page."""

    return flask.render_template("result.html", plot=flask.Markup(plot_div))


@app.route("/prediction")
def prediction():
    """Prediction page."""

    return flask.render_template("prediction.html")


def get_models():
    """Get models.

    This function reads the API_MODELS_PATH file and returns a dictionary
    containing all models.

    Returns:
        models (dict): Dictionary containing all models.

    """

    try:
        models = json.load(open(API_MODELS_PATH))
    except:
        models = {}

    return models


class CreateModel(Resource):
    """Create model."""

    def get(self):
        """Get models.

        This function reads the API_MODELS_PATH file and returns a dictionary
        containing all models.

        Returns:
            models (dict): Dictionary containing all models.

        """

        try:
            models = json.load(open(API_MODELS_PATH))
            return models, 200
        except:
            return {"message": "No models exist."}, 401

    def post(self):
        """Create model.

        This function creates a model based on the parameters provided in the
        request body.

        Returns:
            model_id (str): The ID of the created model.

        """

        # Read parameters from request body (JSON).
        try:
            params_file = flask.request.files["parameter_file"]
            params = yaml.safe_load(params_file)

            annotations_file = flask.request.files["annotations_file"]
            annotated_data = flask.request.files["annotated_data_file"]

            # flask.session["params"] = params
            # flask.redirect("create_model_form")
        except:
            print("Reading parameters from HTML form")
            params = yaml.safe_load(open("params_default.yaml"))
            params["featurize"]["dataset"] = flask.request.form["dataset"]
            params["featurize"]["columns"] = flask.request.form["target"]
            params["featurize"]["timestamp_column"] = flask.request.form[
                "timestamp_column"
            ]
            params["featurize"]["window_size"] = int(flask.request.form["window_size"])
            params["featurize"]["overlap"] = int(flask.request.form["overlap"])
            params["train"]["learning_method"] = flask.request.form["learning_method"]
            params["train"]["n_clusters"] = int(flask.request.form["n_clusters"])
            params["train"]["max_iter"] = int(flask.request.form["max_iter"])
            params["train"]["annotations_dir"] = flask.request.form["annotations_dir"]
            params["train"]["min_segment_length"] = int(
                flask.request.form["min_segment_length"]
            )

            params["featurize"]["convert_timestamp_to_datetime"] = True
            params["train"]["use_predefined_centroids"] = False
            params["train"]["fix_predefined_centroids"] = False

            if flask.request.form.get("use_predefined_centroids"):
                params["train"]["use_predefined_centroids"] = True
                if flask.request.form.get("fix_predefined_centroids"):
                    params["train"]["fix_predefined_centroids"] = True

        # TODO: Currently override the use of annotations, since it is not
        # supported fully in the API.
        # params["train"]["use_predefined_centroids"] = False
        # params["train"]["fix_predefined_centroids"] = False

        # The ID of the model is given an UUID.
        model_id = str(uuid.uuid4())
        # dataset_id = params["featurize"]["dataset"] + "-" + generate_timestamp(formatting="%Y%m%d%H%M%S")
        dataset_id = params["featurize"]["dataset"] #+ "-" + generate_timestamp(formatting="%Y%m%d%H%M%S")
        params["featurize"]["dataset"] = dataset_id

        # Create directory to host data
        data_path = DATA_PATH_RAW / dataset_id
        data_path.mkdir(parents=True, exist_ok=True)

        # Save data file
        data_file = flask.request.files["data_file"]
        data_file.save(os.path.join(data_path, data_file.filename))

        # Save params to be used by DVC when creating virtual sensor.
        yaml.dump(params, open("params.yaml", "w"), allow_unicode=True)

        # Run DVC to create virtual sensor.
        subprocess.run(["dvc", "repro"], check=True)

        # Reread params-file, in case it is changed during pipeline execution
        # (e.g., the number of clusters).
        with open("params.yaml", "r") as params_file:
            params = yaml.safe_load(params_file)

        # Create dict containing all metadata about model
        model_metadata = {}
        model_metadata["id"] = model_id
        model_metadata["params"] = params

        metrics = json.load(open(METRICS_FILE_PATH))
        model_metadata["metrics"] = metrics

        # Read cluster characteristics
        cluster_characteristics = pd.read_csv("assets/output/cluster_names.csv")
        # Save cluster characteristics
        model_metadata["cluster_characteristics"] = [
            c for c in cluster_characteristics.iloc[:, 1]
        ]

        # Try to load existing models. If no models exists, create an empty
        # dict.
        try:
            models = json.load(open(API_MODELS_PATH))
        except:
            models = {}

        models[model_id] = model_metadata

        json.dump(models, open(API_MODELS_PATH, "w+"))

        return flask.redirect("create_model_form")


class InferDemo(Resource):
    """Infer demo."""

    def get(self):
        return 200

    def post(self):
        """Infer demo.

        This function runs inference on a demo dataset.

        Returns:
            200

        """

        # model_id = flask.request.form["id"]
        csv_file = flask.request.files["file"]
        inference_df = pd.read_csv(csv_file)
        print("File is read.")

        # Running actual inference
        analysis = Udava(inference_df)
        analysis.create_train_test_set(["OP390_NC_SP_Torque"])
        analysis.create_fingerprints()
        print("Creating features done.")

        analysis.load_model("model.pkl")
        print("Loading model done.")
        analysis.predict()
        print("Prediction done.")
        analysis.plot_labels_over_time()
        analysis.plot_cluster_center_distance()
        print("Plotting done.")

        return flask.redirect("prediction")


class InferGUI(Resource):
    """Infer GUI."""

    def get(self):
        return 200

    def post(self):
        """Infer GUI.

        This function runs inference on a dataset uploaded by the user.

        Returns:
            200

        """

        model_id = flask.request.form["id"]
        csv_file = flask.request.files["file"]
        inference_df = pd.read_csv(csv_file, index_col=0)
        print("File is read.")

        models = get_models()
        model = models[model_id]
        params = model["params"]

        cm = ClusterModel(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro", "train"], check=True)

        if flask.request.form.get("plot"):
            plot_results=True
        else:
            plot_results=False
                
        print("Running cluster model...")
        fig_div, timestamps, labels, distance_metric = cm.run_cluster_model(
            inference_df=inference_df, plot_results=plot_results
        )

        # Evaluate event log score
        print("Creating event log...")
        event_log = create_event_log(labels, identifier=params["featurize"]["dataset"], feature_vector_timestamps=timestamps)
        event_log.to_csv(OUTPUT_PATH / "event_log.csv")

        try:
            with open("assets/data/expectations/" + params["featurize"]["dataset"] + "/expectations.json", "r") as f:
                # expectations = json.load(f) 
                expectations = eval(f.read())
        except:
            expectations = None
            print("No expectations found.")

        if expectations != None:
            event_log_score(event_log, expectations)

        # Plot results
        if flask.request.form.get("plot"):
            if flask.request.form.get("plot_in_new_window"):
                return flask.redirect("prediction")
            else:
                return flask.redirect("inference_result")
        else:
            timestamps = np.array(timestamps, dtype=np.int32).reshape(-1, 1)
            labels = labels.reshape(-1, 1)
            distance_metric = distance_metric.reshape(-1, 1)
            output_data = np.concatenate([timestamps, labels, distance_metric], axis=1)

            output = {}
            output["param"] = {"modeluid": model_id}
            output["scalar"] = {
                "headers": ["date", "cluster", "metric"],
                "data": output_data.tolist(),
            }

            return output


class Infer(Resource):
    """Infer."""

    def get(self):
        return 200

    def post(self):
        """Infer.

        This function runs inference on a dataset uploaded by the user through
        the API.

        Returns:
            200

        """

        # If file is JSON
        if flask.request.is_json:
            input_json = flask.request.get_json()
            model_id = str(input_json["param"]["modeluid"])

            inference_df = pd.DataFrame(
                input_json["scalar"]["data"],
                columns=input_json["scalar"]["headers"],
            )
        # Else if file is csv
        else:
            model_id = flask.request.form["id"]
            csv_file = flask.request.files["file"]
            inference_df = pd.read_csv(csv_file, index_col=0)

        models = get_models()
        model = models[model_id]
        params = model["params"]

        timestamp_column_name = params["featurize"]["timestamp_column"]
        inference_df.set_index(timestamp_column_name, inplace=True)


        cm = ClusterModel(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro", "train"], check=True)

        fig, timestamps, labels, distance_metric = cm.run_cluster_model(
            inference_df=inference_df, plot_results=True, return_fig=True, png_only=True
        )
        timestamps = np.array(timestamps).reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        distance_metric = distance_metric.reshape(-1, 1)
        output_data = np.concatenate([timestamps, labels, distance_metric], axis=1)
        output_data = output_data.tolist()

        # fig.write_image(str(PLOTS_PATH / "labels_over_time.png"), height=500, width=860)

        # Evaluate event log score
        print("Creating event log...")
        event_log = create_event_log(labels, identifier=params["featurize"]["dataset"], feature_vector_timestamps=timestamps)
        event_log.to_csv(OUTPUT_PATH / "event_log.csv")

        try:
            with open("assets/data/expectations/" + params["featurize"]["dataset"] + "/expectations.json", "r") as f:
                expectations = eval(f.read())
        except:
            expectations = None
            print("No expectations found.")

        output = {}
        output["max_deviation_metric"] = {"value": distance_metric.max()}

        if expectations != None:
            score, _ = event_log_score(event_log, expectations)
            output["event_log_score"] = {"value": score}

        # Make sure that the values in the list output_data are serializable.
        for i in range(len(output_data)):
            for j in range(len(output_data[i])):
                if isinstance(output_data[i][j], np.int64):
                    output_data[i][j] = int(output_data[i][j])
                elif isinstance(output_data[i][j], np.float64):
                    output_data[i][j] = float(output_data[i][j])
        
        output = {}
        output["param"] = {"modeluid": model_id}
        output["scalar"] = {
            "headers": ["date", "cluster", "metric"],
            "data": output_data,
        }

        return output

def generate_timestamp(formatting="%Y-%m-%d %H:%M:%S"):

    if formatting is None:
        return datetime.datetime.now()

    return datetime.datetime.now().strftime(formatting)


if __name__ == "__main__":

    api.add_resource(CreateModel, "/create_model")
    # api.add_resource(InferDemo, "/infer_demo")
    api.add_resource(InferGUI, "/infer_gui")
    api.add_resource(Infer, "/infer")
    app.run(host="0.0.0.0")
