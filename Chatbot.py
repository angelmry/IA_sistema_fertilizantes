# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from ollama import Client

# -----------------------------
# Configuración de la API
# -----------------------------
client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer 3ba702137e6243918677d18184842d9d.ENRd4tpbcqW-qdb88_rv3dwv"}
)

OLLAMA_MODEL = "gpt-oss:120b"

# -----------------------------
# 1. Enviar prompt al LLM
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

# ---------------------------------------------------
# 2. FUNCIÓN PRINCIPAL → generar_explicacion()
# ---------------------------------------------------
def generar_explicacion(
    fertilizer, district, soil, crop,
    nitrogen, phosphorus, potassium,
    ph, rainfall, temperature
):
    """
    Genera una explicación breve basada en los datos reales ingresados por el usuario.
    """

    prompt = f"""
    Eres un asesor agrícola. Usa estos datos internos para tu análisis pero NO los menciones en tu respuesta:
    Distrito={district}, Suelo={soil}, Cultivo={crop},
    N={nitrogen}, P={phosphorus}, K={potassium},
    pH={ph}, Lluvia={rainfall}, Temperatura={temperature}

    RESPONDE SOLO EN ESTE FORMATO, RESPETANDO LOS SALTOS DE LÍNEA:

    ¿Qué significa?
    - respuesta breve

    ¿Por qué se recomienda este fertilizante?
    - respuesta breve

    ¿Cómo aplicarlo correctamente?
    - respuesta breve

    Advertencia (si aplica):
    - respuesta breve o "Ninguna"
    """

    return enviar_a_llm_api(prompt)


# -----------------------------
# Modo consola
# -----------------------------
if __name__ == "__main__":
    print(generar_explicacion("Urea"))
