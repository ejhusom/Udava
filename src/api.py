#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REST API for Udava.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 14:48:42 

"""
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
from udava import Udava

app = flask.Flask(__name__)
api = Api(app)
app.config["DEBUG"] = True


@app.route("/")
def home():
    return flask.render_template("index.html")


@app.route("/create_model_form")
def create_model_form():

    models = get_models()

    # params = flask.session.get("params", None)

    return flask.render_template(
        "create_model_form.html",
        length=len(models),
        models=models
        # params=params
    )


@app.route("/inference")
def inference():

    models = get_models()

    return flask.render_template("inference.html", models=models)


@app.route("/inference_result")
def inference_result():

    models = get_models()

    return flask.render_template("inference.html", models=models, show_prediction=True)


@app.route("/result")
def result(plot_div):
    return flask.render_template("result.html", plot=flask.Markup(plot_div))


@app.route("/prediction")
def prediction():
    return flask.render_template("prediction.html")


def get_models():

    try:
        models = json.load(open("models.json"))
    except:
        models = {}

    return models


class CreateModel(Resource):
    def get(self):

        try:
            models = json.load(open("models.json"))
            return models, 200
        except:
            return {"message": "No models exist."}, 401

    def post(self):

        try:
            # Read params file
            params_file = flask.request.files["file"]
            params = yaml.safe_load(params_file)
            # flask.session["params"] = params
            # flask.redirect("create_model_form")
        except:
            params = yaml.safe_load(open("params_default.yaml"))
            params["featurize"]["dataset"] = flask.request.form["dataset"]
            params["featurize"]["columns"] = flask.request.form["target"]
            params["featurize"]["overlap"] = int(flask.request.form["overlap"])
            params["featurize"]["timestamp_column"] = flask.request.form[
                "timestamp_column"
            ]
            params["featurize"]["window_size"] = int(flask.request.form["window_size"])
            params["cluster"]["learning_method"] = flask.request.form["learning_method"]
            params["cluster"]["n_clusters"] = int(flask.request.form["n_clusters"])
            params["cluster"]["max_iter"] = int(flask.request.form["max_iter"])

        # Create dict containing all metadata about models
        model_metadata = {}
        # The ID of the model is given an UUID.
        model_id = str(uuid.uuid4())
        model_metadata["id"] = model_id
        model_metadata["params"] = params

        # Save params to be used by DVC when creating virtual sensor.
        yaml.dump(params, open("params.yaml", "w"), allow_unicode=True)

        # Run DVC to create virtual sensor.
        subprocess.run(["dvc", "repro", "cluster"], check=True)

        # TODO: Compute metrics
        # metrics = json.load(open(METRICS_FILE_PATH))
        # model_metadata["metrics"] = metrics

        # Read cluster characteristics
        cluster_characteristics = pd.read_csv("assets/output/cluster_names.csv")
        # Save cluster characteristics
        model_metadata["cluster_characteristics"] = [
            c for c in cluster_characteristics.iloc[:, 1]
        ]

        try:
            models = json.load(open("models.json"))
        except:
            models = {}

        models[model_id] = model_metadata
        print(models)

        json.dump(models, open("models.json", "w+"))

        return flask.redirect("create_model_form")


class InferDemo(Resource):
    def get(self):
        return 200

    def post(self):

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
    def get(self):
        return 200

    def post(self):

        model_id = flask.request.form["id"]
        csv_file = flask.request.files["file"]
        inference_df = pd.read_csv(csv_file, index_col=0)
        print("File is read.")

        models = get_models()
        model = models[model_id]
        params = model["params"]

        cm = ClusterModel(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro", "cluster"], check=True)

        if flask.request.form.get("plot"):
            fig_div = cm.run_cluster_model(inference_df=inference_df, plot_results=True)

            if flask.request.form.get("plot_in_new_window"):
                return flask.redirect("prediction")
            else:
                return flask.redirect("inference_result")
        else:
            timestamps, labels, distance_metric = cm.run_cluster_model(
                inference_df=inference_df
            )
            timestamps = np.array(timestamps, dtype=np.int32).reshape(-1, 1)
            labels = labels.reshape(-1, 1)
            distance_metric = distance_metric.reshape(-1, 1)
            output_data = np.concatenate([timestamps, labels, distance_metric], axis=1)

            output = {}
            output["param"] = {"modeluid": model_id}
            output["scalar"] = {
                "headers": ["date", "cluster", "metric"],
                "data": output_data,
            }

            return output


class Infer(Resource):
    def get(self):
        return 200

    def post(self):

        input_json = flask.request.get_json()
        print(input_json)
        model_id = str(input_json["param"]["modeluid"])

        inference_df = pd.DataFrame(
            input_json["scalar"]["data"],
            columns=input_json["scalar"]["headers"],
        )
        inference_df.set_index("date", inplace=True)

        models = get_models()
        model = models[model_id]
        params = model["params"]

        cm = ClusterModel(params_file=params)

        # Run DVC to fetch correct assets.
        subprocess.run(["dvc", "repro", "cluster"], check=True)

        timestamps, labels, distance_metric = cm.run_cluster_model(
            inference_df=inference_df
        )
        timestamps = np.array(timestamps).reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        distance_metric = distance_metric.reshape(-1, 1)
        output_data = np.concatenate([timestamps, labels, distance_metric], axis=1)
        output_data = output_data.tolist()

        output = {}
        output["param"] = {"modeluid": model_id}
        output["scalar"] = {
            "headers": ["date", "cluster", "metric"],
            "data": output_data,
        }

        return output


if __name__ == "__main__":

    api.add_resource(CreateModel, "/create_model")
    # api.add_resource(InferDemo, "/infer_demo")
    api.add_resource(InferGUI, "/infer_gui")
    api.add_resource(Infer, "/infer")
    app.run()
