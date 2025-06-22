import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null); // Add this line
  const [result, setResult] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    if (file) {
      setPreview(URL.createObjectURL(file)); // Set preview URL
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append('file', image);

    try {
      const res = await axios.post('http://localhost:5000/predict', formData);
      setResult(res.data);
    } catch (err) {
      alert('Error uploading image');
    }
  };

   return (
    <div className="app-container">
      <h2>Is That A Cat or A Dog </h2>
      <h3>🐱🐶</h3>
      <input type="file" onChange={handleImageUpload} accept="image/*" />
      {preview && (
        <img src={preview} alt="Preview" className="image-preview" />
      )}
      <button onClick={handleSubmit}>Predict</button>
      {result && (
        <div className="result">
          <p><strong>Prediction:</strong> {result.label}</p>
          <p><strong>Probability:</strong> {result.probability * 100}%</p>
        </div>
      )}
    </div>
  );
}

export default App;