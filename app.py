from flask import Flask
from flask import render_template, request, redirect
import argparse

from finish_time_predictor import Decoder, Encoder, FinishTimePredictor
from finish_time_predictor import MILESTONE
from utils import (makedataset, preprocess_rawdata)
from main import main

app = Flask(__name__)


@app.before_request
def before_request():
    if not request.is_secure:
        url = request.url.replace('http://', 'https://', 1)
        code = 301
        return redirect(url, code=code)


@app.route('/')
def to_english():
    return redirect("/en")


@app.route('/en')
def hello_world():
    return render_template("hello.html", prediction=False)


@app.route('/ja')
def hello_world_ja():
    return render_template("hello_ja.html", prediction=False)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.route('/<string:text>/submit', methods=['POST', 'GET'])
def process(text):
    if request.method == 'POST':
        if text == "ja":
            html_path = "hello_ja.html"
        else:
            html_path = "hello.html"
        results = main(
            ["--do_predict", "--elapsed_time", request.form['record']])
        return render_template(
            html_path, results=results,
            prediction=True, input_value=request.form['record'])
    else:
        return redirect("/" + text)
