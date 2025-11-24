# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS      # <<< CAMBIO NUEVO
import joblib
import numpy as np
import os

# IMPORTAR EL CHATBOT
from Chatbot import generar_explicacion

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)   # <<< ACTIVAR CORS

# -----------------------------
# Cargar modelo y label encoders
# -----------------------------
MODEL_PATH = "fertilizer_prediction_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

model = joblib.load(MODEL_PATH)
le_dict = joblib.load(ENCODERS_PATH)

# -----------------------------
# Servir el index.html
# -----------------------------
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

# -----------------------------
# Endpoint de predicción
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # 1. Leer variables categóricas
    try:
        district = data["District_Name"]
        soil = data["Soil_color"]
        crop = data["Crop"]
    except:
        return jsonify({"error": "Faltan valores categóricos"}), 400

    # 2. Leer variables numéricas
    try:
        nitrogen = float(data["Nitrogen"])
        phosphorus = float(data["Phosphorus"])
        potassium = float(data["Potassium"])
        ph = float(data["pH"])
        rainfall = float(data["Rainfall"])
        temperature = float(data["Temperature"])
    except Exception as e:
        return jsonify({"error": f"Datos inválidos: {e}"}), 400

    # 3. Convertir categóricas usando tus label encoders
    try:
        district_num = le_dict["District_Name"].transform([district])[0]
        soil_num = le_dict["Soil_color"].transform([soil])[0]
        crop_num = le_dict["Crop"].transform([crop])[0]
    except Exception as e:
        return jsonify({"error": f"Valor categórico no reconocido: {e}"}), 400

    # 4. Crear vector EXACTO de 9 features como fue entrenado el modelo
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

    # 6. Explicación del chatbot
    explicacion = generar_explicacion()

    # 7. Devolver respuesta
    return jsonify({
        "fertilizer": str(pred),
        "explicacion": explicacion
    })

# -----------------------------
# Servir JSON de resultados
# -----------------------------
@app.route("/resultados.json")
def serve_resultados():
    return send_from_directory(".", "resultados.json")

# -----------------------------
# Ejecutar app Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

