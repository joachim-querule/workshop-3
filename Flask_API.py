from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify

import joblib

import pandas as pd

from flasgger import Swagger

from functools import wraps

from werkzeug.exceptions import abort
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

model_dict = joblib.load('Projet-Voiture-D-Occasion-main\mon_ia_voitures_v3.pkl')

modele = model_dict["modele"]
colonnes = model_dict["colonnes"]

# df_input = pd.DataFrame([input_data])

# prediction = modele.predict(df_input)


# print(type(model_charge))
# print(model_charge.keys() if isinstance(model_charge, dict) else model_charge)

# Ajouter le modèle choisie pour faire la prédiction

# api = Api(app, doc='/docs')  # Swagger disponible sur /api/docs

swagger = Swagger(app)  # doc accessible par défaut sur /apidocs

@app.route('/')
def redirect_to_docs():
    return redirect(url_for('flasgger.apidocs'))


@app.route('/api/hello', methods=['GET'])
def hello():
    """
    Un endpoint pour dire bonjour
    ---
    responses:
      200:
        description: Retourne un message
        examples:
          application/json: {"message": "Bonjour"}
    """
    return jsonify({"message": "Bonjour"})

# Version API / JSON
@app.route("/api/estimate", methods=['POST'])
def estimate_api():

    input_data = request.get_json()
    print("Données reçues :", input_data)

    df_input = pd.DataFrame([input_data])
    df_input = df_input[colonnes]

    prediction = modele.predict(df_input)
    price = int(prediction[0])

    # 🔥 IMPORTANT : on renvoie du JSON
    return jsonify({
        "price": price
    })
if __name__ == "__main__":
    app.run(port=5002, debug=True, use_reloader=False)
