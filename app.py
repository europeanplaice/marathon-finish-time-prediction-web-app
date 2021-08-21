from flask import Flask
from flask import render_template, request
import argparse

from finish_time_predictor import Decoder, Encoder, FinishTimePredictor
from finish_time_predictor import MILESTONE
from utils import (makedataset, preprocess_rawdata)
from main import main

app = Flask(__name__)


@app.route('/')
def hello_world(name="Tomo"):
    return render_template("hello.html", prediction=False)


@app.route('/submit', methods=['POST'])
def process():
    results = main(
        ["--do_predict", "--elapsed_time", request.form['record']])
    return render_template("hello.html", results=results, prediction=True)



