from flask import Flask, render_template, request
from transformers import pipeline
import requests
import io
import os

# Configuración de Flask
servidor = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
servidor.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ruta para generación de imágenes (nueva)
@servidor.route('/generar_imagen', methods=['POST'])
def generar_imagen():
    prompt = request.form.get('prompt', '')
    if not prompt:
        return "Prompt requerido", 400
    
    # Usando API gratuita de Stable Diffusion
    response = requests.post(
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        headers={"Authorization": "Bearer TU_API_KEY"},
        json={"inputs": prompt}
    )
    
    if response.status_code == 200:
        return response.content, 200, {'Content-Type': 'image/png'}
    else:
        return f"Error: {response.text}", 500

def generar_preguntas(texto, num_preguntas=10):
    input_text = f"generate questions: {texto}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generar múltiples preguntas (ajusta los parámetros según necesites)
    outputs = modelo.generate(
        input_ids,
        max_length=100,
        num_return_sequences=num_preguntas,
        num_beams=10,
        early_stopping=True
    )
    
    preguntas = [tokenizer.decode(output, skip_special_tokens=True) 
                for output in outputs]
    return preguntas

# Rutas de páginas HTML
@servidor.route("/")
def landing_page():
    return render_template("landing_page.html")

@servidor.route("/index.html")
def index():
    return render_template("index.html")

@servidor.route("/hacer_pregunta")
def pregunta():
    return render_template("hacer_pregunta.html")

@servidor.route("/procesar_pregunta", methods=["POST"])
def procesar_pregunta():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()
        contexto = data.get("contexto", "").strip()
        
        if not pregunta:
            return jsonify({"error": "Se requiere una pregunta"}), 400

        # Mejor formato para respuestas (no solo preguntas)
        if contexto:
            input_text = f"respond to '{pregunta}' using this context: {contexto}"
        else:
            input_text = f"answer this question: {pregunta}"
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = modelo.generate(
            input_ids,
            max_length=200,
            num_beams=5,
            early_stopping=True,
            temperature=0.7  # Añadido para más variedad
        )
        
        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpiar respuesta (opcional)
        respuesta = respuesta.replace(pregunta, "").strip()
        return jsonify({"respuesta": respuesta})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@servidor.route("/traducir")
def traducir_page():
    return render_template("traducir.html")

# Procesamiento de la imagen y predicción
@servidor.route("/resumen", methods=["GET"])
def resumen():
    return render_template("resumir.html")  

@servidor.route("/procesar_resumen", methods=["POST"])
def procesar_resumen():
    # Tu lógica de procesamiento aquí
    pass

# Generar pregunta (redirige a la página de pregunta)
# Versión corregida con ambos métodos
@servidor.route("/generar_pregunta", methods=["GET", "POST"])
def generar_pregunta():
    if request.method == "POST":
        # Lógica para generar pregunta desde un formulario
        datos = request.form
        tema = datos.get("tema", "general")
        pregunta_generada = generar_pregunta_ia(tema)  # Tu función de generación
        return render_template("generar_preguntas.html", pregunta=pregunta_generada)
    else:
        # Lógica para GET (mostrar formulario)
        return render_template("generar_preguntas.html")  # Crea este template
    
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
