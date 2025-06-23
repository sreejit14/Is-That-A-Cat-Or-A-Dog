import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

MODEL_URL = "https://huggingface.co/Sreejit14/Is-That-A-Cat-or-A-Dog/resolve/main/cat_dog_classifier.h5"
MODEL_PATH = "cat_dog_classifier.h5"

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption="Uploaded Image")

    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction >= 0.5 else "Cat"
    confidence = round(prediction if label == "Dog" else 1 - prediction, 3)

    st.success(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")
