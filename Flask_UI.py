# =======================  Importation =========================================

from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify
import requests
# from flask_restx import Api, Resource
# from flasgger import Swagger

from functools import wraps

import sqlite3

from werkzeug.exceptions import abort
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
# import joblib
import json

# from SVR import PIPELINE
# Pipe = PIPELINE()

# =================================================================================

app = Flask(__name__)
# api = Api(app, doc='/docs')  # Swagger disponible sur /api/docs
# swagger = Swagger(app)  # doc accessible par défaut sur /apidocs

app.secret_key = "super_secret_key_change_this"

# @app.after_request
# def add_no_cache_headers(response):
#     response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
#     response.headers["Pragma"] = "no-cache"
#     response.headers["Expires"] = "0"
#     return response

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# @app.route('/')
# def redirect_to_index():
#     return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

# @api.route('/api/hello', endpoint='api_hello')
# class Hello(Resource):
#     def get(self):
#         return "Bonjour"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            return redirect(url_for("dashboard"))

        return "Login invalide"

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/Sign_Up", methods=["GET", "POST"])
def Sign_Up():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, password))
            conn.commit()
        except:
            return "Utilisateur déjà existant"

        conn.close()
        return redirect(url_for("login"))

    return render_template("Sign_Up.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("DATAVIZ.html")


@app.route("/estimate", methods=['GET', 'POST'])
def estimate():

    if request.method == "POST":
        try:
            data = {
                "Location": request.form["Location"],
                "Year": int(request.form["Year"]),
                "Kilometers_Driven": int(request.form["Kilometers_Driven"]),
                "Mileage": request.form["Mileage"],
                "Engine": request.form["Engine"],
                "Power": request.form["Power"],
                "Seats": int(request.form["Seats"]),
                "Marque": request.form["Marque"],
                "Modele": request.form["Modele"],
                "Fuel_Type": request.form["Fuel_Type"],
                "Transmission": request.form["Transmission"],
                "Owner_Type": request.form["Owner_Type"],
            }

            df_input = pd.DataFrame([data])
            tables = [df_input.to_html(classes="data")]

            # 🔥 Appel API sécurisé
            try:
                response = requests.post(
                    "http://127.0.0.1:5002/api/estimate",
                    json=data,
                    timeout=5
                )

                if response.status_code == 200:
                    prediction = response.json().get("price")
                else:
                    prediction = "Erreur API"

            except requests.exceptions.RequestException:
                prediction = "API indisponible"

            return render_template(
                "estimate.html",
                tables=tables,
                prediction=prediction
            )

        except ValueError:
            return render_template(
                "estimate.html",
                error="Erreur : Vérifiez les champs numériques."
            )

    return render_template("estimate.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)

