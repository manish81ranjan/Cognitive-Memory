# server.py (Flask Backend with Keras Model Fix)
# --------------------
# Dependencies:
# pip install Flask pillow numpy tensorflow Flask-CORS uvicorn
# --------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import io
import base64
import uvicorn # Used to serve the Flask app efficiently

# --------------------
# Flask App Setup
# --------------------
app = Flask(__name__)
# Crucial for integration: Allows your frontend (running on file:// or a different port) 
# to communicate with this backend.
CORS(app) 

# --------------------
# Model Loading (The Critical Fix)
# --------------------
model = None
try:
    # **FIX:** Added safe_mode=False to resolve the 'Functional' deserialization error.
    # This addresses the "TypeError: Could not deserialize class 'Functional'"
    model = tf.keras.models.load_model(
        "best_demnet_model.keras",
        compile=False,
        safe_mode=False 
    )
    print("DEMNET Model loaded successfully.")
    # 
    
except Exception as e:
    print(f"FATAL ERROR: Could not load the Keras model 'best_demnet_model.keras'.")
    print(f"Reason: {e}")
    print("Please ensure the model file is in the same directory and compatible with your TensorFlow/Keras version.")
    model = None


CLASS_NAMES = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# --------------------
# Image Preprocessing
# --------------------
def preprocess_image(img_bytes):
    """Loads image, converts to RGB, resizes to 224x224, and normalizes."""
    # Ensure all three color channels (RGB) are present for the model input
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB") 
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized) / 255.0 # Normalize to 0-1
    arr = np.expand_dims(arr, axis=0) # Add batch dimension (1, 224, 224, 3)
    return arr, img_resized

# --------------------
# Predict API Endpoint
# --------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model service unavailable"}, 503)
        
    if 'file' not in request.files:
        return jsonify({"error": "No MRI slice file found in request"}, 400)
        
    file = request.files['file']
    img_bytes = file.read()

    try:
        # Preprocess the uploaded image
        tensor, raw_img = preprocess_image(img_bytes)
        
        # Prediction
        preds = model.predict(tensor)[0]

        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx]
        probs = preds.tolist()

        # ---- SIMPLE HEATMAP PLACEHOLDER for Grad-CAM ----
        # The frontend expects a 'gradcam' field. For a full implementation, you'd 
        # compute the real Grad-CAM heatmap here and overlay it.
        # This implementation just provides a simple, solid red overlay.
        heat = Image.new("RGBA", raw_img.size, (255, 0, 0, 90)) 
        blended = Image.alpha_composite(raw_img.convert("RGBA"), heat)

        # Encode the 'Grad-CAM' image to base64 for the frontend
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode()
        # 

        return jsonify({
            "label": label,
            "probs": probs,
            "gradcam": gradcam_b64
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": f"Error during prediction: {e}"}, 500)


# --------------------
# Metrics API Endpoint
# --------------------
@app.route("/metrics", methods=["GET"])
def metrics():
    """Provides mock performance metrics for the home page counters."""
    return jsonify({"acc": 92, "auc": 88, "f1": 85})

# --------------------
# Run Server
# --------------------
if __name__ == "__main__":
    # Runs the Flask application using uvicorn for better performance.
    # Frontend calls are directed to this host and port.
    uvicorn.run(app, host="0.0.0.0", port=8000)