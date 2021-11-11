from flask import Flask
from flask import render_template, request, redirect

from main import main
import os

app = Flask(__name__)


# @app.before_request
# def before_request():
#     if not request.is_secure and os.environ.get('FLASK_ENV') != 'development':
#         url = request.url.replace('http://', 'https://', 1)
#         code = 301
#         return redirect(url, code=code)


@app.route('/explain')
def how_it_predicts():
    return render_template("explain.html", prediction=False)


@app.route('/')
def to_english():
    return render_template("index.html", prediction=False)


@app.route('/en')
def index():
    return render_template("index.html", prediction=False)


@app.route('/ja')
def index_ja():
    return render_template("index_ja.html", prediction=False)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.route('/<string:text>/submit', methods=['POST', 'GET'])
def process(text):
    if request.method == 'POST':
        if text == "ja":
            html_path = "index_ja.html"
        else:
            html_path = "index.html"
        results = main(
            ["--do_predict", "--elapsed_time", request.form['record']])
        return render_template(
            html_path, results=results,
            prediction=True, input_value=request.form['record'])
    else:
        return redirect("/" + text)
