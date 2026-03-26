from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

import os

from flask import render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Suppress certain Keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Suppress certain Keras warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

def custom_load_model(path):
    return load_model(
        path,
        compile=False,
        safe_mode=False,
        custom_objects={"Dense": CustomDense}
    )

# Load model (simplified)
model = custom_load_model("crop_model.h5")

# Class names (IMPORTANT: same order as training)
class_names = ['early_blight', 'healthy', 'late_blight']


# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # Soil condition logic
        if predicted_class == "early_blight":
            soil_condition = "Possible nitrogen deficiency"
        elif predicted_class == "late_blight":
            soil_condition = "Excess moisture in soil"
        else:
            soil_condition = "Soil is healthy"

        # Pest risk logic
        if predicted_class == "early_blight":
            pest_risk = "Medium"
        elif predicted_class == "late_blight":
            pest_risk = "High"
        else:
            pest_risk = "Low"
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Recommendation logic
    if predicted_class == "early_blight":
        solution = "Use fungicide and remove affected leaves"
    elif predicted_class == "late_blight":
        solution = "Apply pesticide and reduce moisture"
    else:
        solution = "Crop is healthy"

    return jsonify({
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
        "soil_condition": soil_condition,
        "pest_risk": pest_risk,
        "solution": solution
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)