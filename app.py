# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

from Chatbot import generar_explicacion

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

MODEL_PATH = "fertilizer_prediction_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

model = joblib.load(MODEL_PATH)
le_dict = joblib.load(ENCODERS_PATH)

# -----------------------------
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # 1. Variables categóricas
    district = data["District_Name"]
    soil = data["Soil_color"]
    crop = data["Crop"]

    # 2. Variables numéricas
    nitrogen = float(data["Nitrogen"])
    phosphorus = float(data["Phosphorus"])
    potassium = float(data["Potassium"])
    ph = float(data["pH"])
    rainfall = float(data["Rainfall"])
    temperature = float(data["Temperature"])

    # 3. Transformación categórica
    district_num = le_dict["District_Name"].transform([district])[0]
    soil_num = le_dict["Soil_color"].transform([soil])[0]
    crop_num = le_dict["Crop"].transform([crop])[0]

    # 4. Vector ordenado
    features = np.array([[ 
        district_num,
        soil_num,
        nitrogen,
        phosphorus,
        potassium,
        ph,
        rainfall,
        temperature,
        crop_num
    ]])

    # 5. Predicción
    pred = model.predict(features)[0]

    # 6. Pasamos la predicción al chatbot
    explicacion = generar_explicacion(
        pred,
        district,
        soil,
        crop,
        nitrogen,
        phosphorus,
        potassium,
        ph,
        rainfall,
        temperature
    )

    # 7. Respuesta
    return jsonify({
        "fertilizer": str(pred),
        "explicacion": explicacion
    })

# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
