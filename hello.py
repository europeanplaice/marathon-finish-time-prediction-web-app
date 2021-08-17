from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/')
def hello_world(name="Tomo"):
    return render_template("hello.html", name=name)
