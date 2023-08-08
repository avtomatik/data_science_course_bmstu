import pickle

import flask
from flask import render_template

app = flask.Flask(__name__, template_folder="templates")

@app.route("/", methods=["POST", "GET"])

@app.route("/index", methods=["POST", "GET"])
def main():
    if request.method == "GET":
        return render_template("main.html")

    if request.method == "POST":
        with open("lr_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        experience = float(request.form["exprerience"])
        y_pred = loaded_model.predict([[experience]])

        return render_template("main.html", result=y_pred)


if __name__ == "__main__":
    app.run()
