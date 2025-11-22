from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

app = Flask(__name__, static_folder=".", template_folder=".")

# ----------------------
# PATHS
# ----------------------
MODEL_PATH = "animal_classifier.h5"
CLASS_MAP_PATH = "class_indices.json"
SPECIES_DATA_PATH = "static/data/species.json"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------
# LOAD MODEL + METADATA
# ----------------------
try:
    model = load_model(MODEL_PATH)
    print("✔ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

# reverse mapping: {0: "elephant", 1: "tiger", ...}
labels = {v: k for k, v in class_map.items()}

with open(SPECIES_DATA_PATH) as f:
    species_data = json.load(f)


# ----------------------
# ROUTES FOR PAGES
# ----------------------
@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload.html")
def upload_page():
    return send_from_directory(".", "upload.html")


@app.route("/enci.html")
def enci_page():
    return send_from_directory(".", "enci.html")


@app.route("/map.html")
def map_page():
    return send_from_directory(".", "map.html")


@app.route("/species.html")
def species_page():
    return send_from_directory(".", "species.html")


# ----------------------
# SERVE STATIC FILES
# ----------------------
@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return "404 Not Found"


# ----------------------
# IMAGE PREPROCESS
# ----------------------
def prepare_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print("Image processing error:", e)
        return None


# ----------------------
# NORMALIZATION FOR NAME MATCHING
# ----------------------
def normalize(name):
    return (
        name.lower()
        .replace("(domestic)", "")
        .replace("(wild water buffalo)", "")
        .replace("(asian elephant)", "")
        .replace("(black panther – melanistic leopard)", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .strip()
    )


# ----------------------
# IDENTIFICATION API
# ----------------------
@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    img = prepare_image(img_path)
    if img is None:
        return jsonify({"error": "Image processing failed"}), 500

    # Prediction
    preds = model.predict(img)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    # Raw predicted name
    predicted_label = labels.get(idx, "Unknown")
    predicted_clean = normalize(predicted_label)

    # Try to find metadata from species.json using fuzzy match
    matched = next(
        (item for item in species_data if predicted_clean in normalize(item["name"])),
        None
    )

    response = {
        "name": matched["name"] if matched else predicted_label,
        "scientific_name": matched["scientific_name"] if matched else "Not Available",
        "status": matched["status"] if matched else "Unknown",
        "confidence": round(confidence, 3),
        "upload_count": 1
    }

    return jsonify(response)


# ----------------------
# RUN APP
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
