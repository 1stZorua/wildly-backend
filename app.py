from flask import Flask, request, jsonify
from keras._tf_keras.keras.models import load_model
from utils import load_class_names, prepare_image
import numpy as np

MODEL_PATH = 'models/dog_classifier_breed.keras'
CLASSES_PATH = 'models/classes.json'

app = Flask(__name__)
model = load_model(MODEL_PATH)
classes = load_class_names(CLASSES_PATH)

@app.route('/')
def home():
    return "Wildly API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    
    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    img = prepare_image(image)
    prediction = model.predict(img)

    predicted_index = np.argmax(prediction)
    confidence = float(prediction[0][predicted_index] * 100) 

    top_3_indices = np.argsort(prediction)[0][-3:][::-1]

    top_3_breeds = [
        {
            'breed': classes[i],
            'confidence': float(round(prediction[0][i] * 100, 2))
        }
        for i in top_3_indices
    ]

    return jsonify({
        'predicted_breed': classes[predicted_index],
        'confidence': round(confidence, 2),
        'top_3': top_3_breeds
    })


if __name__ == '__main__':
    app.run(debug=True)
