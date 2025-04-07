from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# Configuración inicial
servidor = Flask(__name__)
CORS(servidor)  # Habilita CORS

# Cargar modelo y tokenizer
MODEL_NAME = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Configurar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)


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

@servidor.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text input required'}), 400
        
        # Prefijo optimizado para generación de preguntas en inglés
        input_text = f"generate questions: {text}"
        
        # Tokenizar y enviar al dispositivo adecuado
        inputs = tokenizer.encode(input_text, return_tensors='pt', 
                                max_length=512, truncation=True).to(device)
        
        # Generar con parámetros optimizados para preguntas en inglés
        outputs = model.generate(
            inputs,
            max_length=64,  # Más corto para preguntas concisas
            num_return_sequences=10,
            num_beams=15,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.8,  # Más variedad
            do_sample=True    # Para más diversidad
        )
        
        # Decodificar y limpiar resultados
        questions = []
        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            # Limpieza y formato de preguntas
            question = question.replace("generate questions:", "").strip()
            if not question.endswith("?"):
                question += "?"
            question = question.capitalize()
            
            # Filtrar preguntas mal formadas
            if len(question.split()) > 4 and "?" in question:  # Mínimo 5 palabras
                questions.append(question)
        
        # Eliminar duplicados manteniendo el orden
        unique_questions = []
        [unique_questions.append(q) for q in questions if q not in unique_questions]
        
        return jsonify({'questions': unique_questions[:10]})  # Asegurar máximo 10 preguntas
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Error generating questions'}), 500



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
            return jsonify({"error": "Question is required"}), 400

        # Improved prompt engineering
        if contexto:
            input_text = f"Question: {pregunta}\nContext: {contexto}\nProvide a detailed, structured answer:"
        else:
            input_text = f"Answer this question in detail: {pregunta}"
        
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Improved generation parameters
        outputs = model.generate(
            inputs,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing to clean the output
        respuesta = respuesta.replace(pregunta, "").strip()
        respuesta = respuesta.split("Answer:")[-1].strip()  # Remove any remaining prompt fragments
        return jsonify({"respuesta": respuesta})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint específico para preguntas con GPT-2
@servidor.route("/procesar_pregunta_gpt2", methods=["POST"])
def procesar_pregunta_gpt2():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()
        contexto = data.get("contexto", "").strip()
        
        if not pregunta:
            return jsonify({"error": "Se requiere una pregunta"}), 400

        # Construir el prompt para GPT-2
        prompt = f"Pregunta: {pregunta}\n"
        if contexto:
            prompt += f"Contexto: {contexto}\n"
        prompt += "Respuesta detallada:"
        
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        outputs = gpt2_model.generate(
            inputs,
            max_length=600,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
        respuesta = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpiar la respuesta eliminando el prompt
        respuesta = respuesta.replace(prompt, "").strip()
        return jsonify({"respuesta": respuesta})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@servidor.route("/traducir")
def traducir_page():
    return render_template("traducir.html")

@servidor.route("/traducir", methods=["POST"])
def traducir():
    try:
        data = request.get_json()
        texto = data.get("texto", "").strip()
        
        if not texto:
            return jsonify({"error": "Se requiere texto para traducir"}), 400
            
        if len(texto) > 5000:
            return jsonify({"error": "El texto no puede exceder los 5000 caracteres"}), 400
            
        # Prefijo fijo para traducción inglés a francés
        prefix = "translate English to French: "
        
        # Tokenizar y traducir
        input_ids = tokenizer.encode(prefix + texto, return_tensors="pt", 
                                   max_length=512, truncation=True).to(device)
        
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True,
            temperature=0.7
        )
        
        traduccion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "traduccion": traduccion,
            "caracteres": len(traduccion)
        })
        
    except Exception as e:
        print(f"Error en traducción: {str(e)}")
        return jsonify({"error": "Error al traducir el texto"}), 500

# Procesamiento de la imagen y predicción
@servidor.route("/resumen", methods=["GET"])
def resumen():
    return render_template("resumir.html")  

@servidor.route("/procesar_resumen", methods=["POST"])
def procesar_resumen():
    try:
        data = request.get_json()  # Obtenemos el texto del cuerpo de la solicitud
        texto = data.get('texto', '').strip()  # El texto a resumir

        if not texto:
            return jsonify({"error": "Se requiere texto para resumir"}), 400

        # Preparamos el input para el modelo T5
        input_text = f"summarize: {texto}"

        # Tokenizamos el texto de entrada
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

        # Generamos el resumen
        outputs = model.generate(
            inputs,
            max_length=150,  # Limitar el largo del resumen
            num_beams=5,     # Usamos búsqueda en haz para mejorar la calidad del resumen
            early_stopping=True,
            temperature=0.7  # Se puede ajustar para controlar la creatividad del resumen
        )

        # Decodificamos el resumen generado
        resumen = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"resumen": resumen})  # Devolvemos el resumen generado en formato JSON

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Error al generar el resumen"}), 500



# Generar pregunta (redirige a la página de pregunta)
# Versión corregida con ambos métodos
@servidor.route("/generar_pregunta", methods=["GET", "POST"])
def generar_pregunta():
    if request.method == "POST":
        try:
            tema = request.form.get("tema", "general")  # Usamos request.form si viene del formulario HTML
            pregunta_generada = generar_pregunta_ia(tema)  # Tu función personalizada
            return render_template("generar_preguntas.html", pregunta=pregunta_generada)
        except Exception as e:
            return render_template("generar_preguntas.html", error=str(e))
    else:
        # GET: simplemente mostrar el formulario vacío
        return render_template("generar_preguntas.html")
    

def generar_pregunta_ia(tema: str) -> str:
    # Preparamos el input para el modelo
    input_text = f"Genera una pregunta sobre el tema: {tema}"

    # Tokenizamos el texto
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Generamos la pregunta
    output = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)

    # Decodificamos la respuesta generada
    pregunta_generada = tokenizer.decode(output[0], skip_special_tokens=True)

    return pregunta_generada

# Traducción de texto (redirige a la página de traducción)

if __name__ == "__main__":
    servidor.run(port=4000, debug=True)
