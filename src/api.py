#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API for Udava.

Author:
    Erik Johannes Husom

Created:
    2021-11-29 Monday 14:48:42 

"""
import os
import json
import time
import subprocess
import urllib.request
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

    return flask.render_template(
            "create_model_form.html",
            length=len(models),
            models=models
    )

@app.route("/inference")
def inference():

    models = get_models()

    return flask.render_template(
            "inference.html",
            models=models
    )

@app.route("/result")
def result(plot_div):
    # plot_div = session["plot_div"]
    return flask.render_template("result.html",
            plot=flask.Markup(plot_div))

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
            virtual_sensors = json.load(open("models.json"))
            return virtual_sensors, 200
        except:
            return {"message": "No models exist."}, 401

    def post(self):

        try:
            # Read params file
            params_file = flask.request.files["file"]
            params = yaml.safe_load(params_file)
        except:
            params = yaml.safe_load(open("params_default.yaml"))
            params["featurize"]["dataset"] = flask.request.form["dataset"]
            params["featurize"]["columns"]= flask.request.form["column"]
            params["cluster"]["learning_method"]= flask.request.form["learning_method"]
            print(params)

        # Create dict containing all metadata about virtual sensor
        model_metadata = {}
        # The ID of the virtual sensor is set to the current Unix time for
        # uniqueness.
        model_metadata["id"] = int(time.time())
        model_metadata["params"] = params

        # Save params to be used by DVC when creating virtual sensor.
        yaml.dump(params, open("params.yaml", "w"), allow_unicode=True)

        # Run DVC to create virtual sensor.
        subprocess.run(["dvc", "repro"])

        metrics = json.load(open(METRICS_FILE_PATH))
        virtual_sensor_metadata["metrics"] = metrics

        try:
            models = json.load(open("models.json"))
        except:
            models = {}

        models[model_id] = model_metadata
        print(models)

        json.dump(models, open("models.json", "w+"))

        return flask.redirect("create_model_form")


class Infer(Resource):
    def get(self):
        return 200

    def post(self):

        # virtual_sensor_id = flask.request.form["id"]
        csv_file = flask.request.files["file"]
        inference_df = pd.read_csv(csv_file)
        print("File is read.")

        analysis = Udava(inference_df)
        analysis.create_train_test_set(["OP390_NC_SP_Torque"])
        analysis.create_fingerprints()
        print("Fingerprints have been created.")
        analysis.load_model("model.pkl")
        print("Loading model done.")
        analysis.predict()
        print("Done with prediction.")
        analysis.visualize_clusters()
        analysis.plot_labels_over_time()
        analysis.plot_cluster_center_distance()
        print("Done with plotting.")

        return flask.redirect("prediction")


if __name__ == "__main__":

    api.add_resource(CreateModel, "/create_model")
    api.add_resource(Infer, "/infer")
    app.run()
