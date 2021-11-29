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

@app.route("/inference")
def inference():

    virtual_sensors = get_virtual_sensors()

    return flask.render_template(
            "inference.html",
            virtual_sensors=virtual_sensors
    )

@app.route("/result")
def result(plot_div):
    # plot_div = session["plot_div"]
    return flask.render_template("result.html",
            plot=flask.Markup(plot_div))

@app.route("/prediction")
def prediction():
    return flask.render_template("prediction.html")

def get_virtual_sensors():

    try:
        virtual_sensors = json.load(open("virtual_sensors.json"))
    except:
        virtual_sensors = {}

    return virtual_sensors

class CreateVirtualSensor(Resource):
    def get(self):

        try:
            virtual_sensors = json.load(open("virtual_sensors.json"))
            return virtual_sensors, 200
        except:
            return {"message": "No virtual sensors exist."}, 401

    def post(self):


        try:
            # Read params file
            params_file = flask.request.files["file"]
            params = yaml.safe_load(params_file)
        except:
            params = yaml.safe_load(open("params_default.yaml"))
            params["profile"]["dataset"] = flask.request.form["dataset"]
            params["clean"]["target"]= flask.request.form["target"]
            params["train"]["learning_method"]= flask.request.form["learning_method"]
            params["split"]["train_split"] = float(flask.request.form["train_split"]) / 10
            print(params)

        # Create dict containing all metadata about virtual sensor
        virtual_sensor_metadata = {}
        # The ID of the virtual sensor is set to the current Unix time for
        # uniqueness.
        virtual_sensor_id = int(time.time())
        virtual_sensor_metadata["id"] = virtual_sensor_id
        virtual_sensor_metadata["params"] = params

        # Save params to be used by DVC when creating virtual sensor.
        yaml.dump(params, open("params.yaml", "w"), allow_unicode=True)

        # Run DVC to create virtual sensor.
        subprocess.run(["dvc", "repro"])

        metrics = json.load(open(METRICS_FILE_PATH))
        virtual_sensor_metadata["metrics"] = metrics

        try:
            virtual_sensors = json.load(open("virtual_sensors.json"))
        except:
            virtual_sensors = {}

        virtual_sensors[virtual_sensor_id] = virtual_sensor_metadata
        print(virtual_sensors)

        json.dump(virtual_sensors, open("virtual_sensors.json", "w+"))

        return flask.redirect("virtual_sensors")


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

    api.add_resource(CreateVirtualSensor, "/create_virtual_sensor")
    api.add_resource(Infer, "/infer")
    app.run()
