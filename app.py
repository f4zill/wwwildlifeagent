from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

app = Flask(__name__, static_folder=".", template_folder=".")

# Path Configurations
MODEL_PATH = "animal_classifier.h5"
CLASS_MAP_PATH = "class_indices.json"
SPECIES_DATA_PATH = "data/species.json"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
model = load_model(MODEL_PATH)

# Load Class Indices
with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)
labels = {v: k for k, v in class_map.items()}  # reverse key/value

# Load Species Metadata
with open(SPECIES_DATA_PATH) as s:
    species_data = json.load(s)


# ========== PAGE ROUTES ==========
@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload.html")
def upload():
    return send_from_directory(".", "upload.html")


@app.route("/enci.html")
def enci():
    return send_from_directory(".", "enci.html")


@app.route("/map.html")
def map_page():
    return send_from_directory(".", "map.html")


@app.route("/species.html")
def species():
    return send_from_directory(".", "species.html")


# ========== SERVE STATIC FILES (data, geojson, libs, images, etc.) ==========
@app.route("/<path:filename>")
def static_files(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return "404 Not Found"


# ========== IMAGE PREPROCESSING ==========
def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # change if your model uses diff size
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ========== PREDICTION API ==========
@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    img = prepare_image(img_path)
    preds = model.predict(img)

    idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    predicted_name = labels.get(idx, "Unknown").lower().strip()

    # Cleaning function to match names properly
    def normalize(n):
        return (
            n.lower()
             .replace("(domestic)", "")
             .replace("(wild water buffalo)", "")
             .replace("(asian elephant)", "")
             .replace("(black panther – melanistic leopard)", "")
             .replace("(", "")
             .replace(")", "")
             .replace("-", "")
             .strip()
        )

    predicted_clean = normalize(predicted_name)

    # Fuzzy match from species.json
    details = next(
        (a for a in species_data if predicted_clean in normalize(a["name"])),
        None
    )

    return jsonify({
        "name": details["name"] if details else predicted_name,
        "scientific_name": details["scientific_name"] if details else "Not Available",
        "status": details["status"] if details else "Unknown",
        "upload_count": 1,
        "confidence": round(confidence, 3)
    })


# Run Server
if __name__ == "__main__":
    app.run(debug=True)
