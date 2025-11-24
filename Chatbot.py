# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from ollama import Client

# -----------------------------
# 0. Configuración de la API de Ollama
# -----------------------------
client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer 3ba702137e6243918677d18184842d9d.ENRd4tpbcqW-qdb88_rv3dwv"}
)

OLLAMA_MODEL = "gpt-oss:120b"


# -----------------------------
# 1. Función para enviar prompt al LLM vía Cloud
# -----------------------------
def enviar_a_llm_api(prompt):
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]
        respuesta = ""
        for part in client.chat(OLLAMA_MODEL, messages=messages, stream=True):
            respuesta += part["message"]["content"]
        return respuesta
    except Exception as e:
        return f"Error al comunicarse con Ollama Cloud: {e}"


# -----------------------------
# 2. Leer resultados JSON
# -----------------------------
def leer_resultados_json(file_path="resultados.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error al leer el JSON: {e}")
        return None


# ---------------------------------------------------
# 3. FUNCIÓN QUE USARÁ FLASK → generar_explicacion()
# ---------------------------------------------------
def generar_explicacion():
    """
    ✔ Lee el JSON
    ✔ Envía prompt a Ollama
    ✔ Devuelve texto (NO imprime)
    ✔ Puede ser llamado desde app.py
    """
    data = leer_resultados_json()
    if not data:
        return "No se pudo cargar resultados.json"

    # Prompt simple para agricultores
    prompt = f"""
Explica estos resultados de IA de forma MUY simple, como si hablaras con un agricultor.

Quiero SOLO este formato:

¿Qué significa?
[respuesta breve]

¿Cuándo puede fallar?
[respuesta breve]

Datos de la IA:
Accuracy: {data['accuracy']}
Reporte por clase:
{json.dumps(data['classification_report'], indent=4)}
"""

    respuesta = enviar_a_llm_api(prompt)
    return respuesta


# -----------------------------
# 4. Ejecutar chatbot manualmente (solo consola)
# -----------------------------
def chatbot():
    print(generar_explicacion())


if __name__ == "__main__":
    chatbot()
