from flask import Flask
from flask import render_template, request
import argparse

from finish_time_predictor import Decoder, Encoder, FinishTimePredictor
from utils import (makedataset, preprocess_rawdata)
from main import main

app = Flask(__name__)


@app.route('/')
def hello_world(name="Tomo"):
    return render_template("hello.html", name=name)


@app.route('/submit', methods=['POST'])
def process():
    request = Request
    main(["--do_predict", "--elapsed_time", data])


