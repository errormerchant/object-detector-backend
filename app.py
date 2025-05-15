from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
from detector import detect_objects

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    results = detect_objects(image)

    return jsonify({"predictions": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)