from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from werkzeug.utils import secure_filename

# Cargar el modelo de clasificación de rostros

# Configuración de Flask
servidor = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
servidor.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Rutas de páginas HTML
@servidor.route("/")
def landing_page():
    return render_template("landing_page.html")

@servidor.route("/index.html")
def index():
    return render_template("Index.html")

@servidor.route("/pregunta")
def pregunta():
    return render_template("pregunta.html")

@servidor.route("/traducir")
def traducir_page():
    return render_template("traducir.html")

# Procesamiento de la imagen y predicción
@servidor.route("/resumen", methods=["POST"])


# Generar pregunta (redirige a la página de pregunta)
@servidor.route("/generar_pregunta", methods=["POST"])
def generar_pregunta():
    datos = request.json
    tema = datos.get("tema", "general")
    
    pregunta_generada = "¿Cuál es la capital de Francia?" if tema == "geografía" else "¿Cuál es la raíz cuadrada de 16?"
    
    return render_template("pregunta.html", pregunta=pregunta_generada)

# Traducción de texto (redirige a la página de traducción)
@servidor.route("/traducir", methods=["POST"])
def traducir():
    datos = request.json
    texto = datos.get("texto", "")
    idioma = datos.get("idioma", "en")
    
    traducciones = {
        "en": "Hello, how are you?",
        "fr": "Bonjour, comment ça va?",
        "it": "Ciao, come stai?"
    }
    
    traduccion = traducciones.get(idioma, "Traducción no disponible")
    
    return render_template("traducir.html", texto_original=texto, traduccion=traduccion)

if __name__ == "__main__":
    servidor.run(port=4000, debug=True)
