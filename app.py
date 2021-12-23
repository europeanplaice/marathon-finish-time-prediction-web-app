from flask import Flask
from flask import render_template, request, redirect, send_from_directory

from main import main
import os

app = Flask(__name__)


@app.before_request
def before_request():
    if not request.is_secure and os.environ.get('FLASK_ENV') != 'development':
        url = request.url.replace('http://', 'https://', 1)
        code = 301
        return redirect(url, code=code)


@app.route('/sitemap.xml')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route('/explaination')
def how_it_predicts():
    return render_template("explaination.html", prediction=False)


@app.route('/')
def to_english():
    return render_template("index.html", prediction=False)


@app.route('/zh')
def index_zh():
    return render_template("index_zh.html", prediction=False)


@app.route('/ja')
def index_ja():
    return render_template("index_ja.html", prediction=False)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.route('/submit', methods=['POST', 'GET'])
@app.route('/<string:text>/submit', methods=['POST', 'GET'])
def process(text=None):
    if request.method == 'GET':
        if text == "ja":
            html_path = "index_ja.html"
        elif text == "zh":
            html_path = "index_zh.html"
        else:
            html_path = "index.html"
        results = main(
            ["--do_predict", "--elapsed_time", request.args.get('q')])
        return render_template(
            html_path, results=results,
            prediction=True, input_value=request.args.get('q'))
    else:
        return redirect("/" + text)
