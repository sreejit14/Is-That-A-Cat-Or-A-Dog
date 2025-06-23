from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import os
import requests
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='../cat-dog-ui/build', static_url_path='/')
CORS(app)

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

MODEL_URL = 'https://huggingface.co/Sreejit14/Is-That-A-Cat-or-A-Dog/resolve/main/cat_dog_classifier.h5'
MODEL_PATH = 'cat_dog_classifier.h5'

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

with open(MODEL_PATH, 'rb') as f:
    print(f.read(200))

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = preprocess_image(file.read())
    prediction = model.predict(img)[0][0]
    label = 'Dog' if prediction >= 0.5 else 'Cat'
    return jsonify({
        'label': label,
        'probability': round(float(prediction if label == 'Dog' else 1 - prediction), 3)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
