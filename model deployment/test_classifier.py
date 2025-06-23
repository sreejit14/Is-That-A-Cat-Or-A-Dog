import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('cat_dog_classifier.h5')

# Load and preprocess the image
img_path = 'dog vs cat\mcbc.jpg'  # Update with your image path
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale

# Predict
prediction = model.predict(img_array)
print("Prediction Probability:", prediction[0][0])

# Convert probability to label
label = "Dog" if prediction[0][0] > 0.5 else "Cat"
print(f"Predicted Label: {label}")