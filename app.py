from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2 as cv
import io
import os
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://mateuszkaron.github.io/emotion-scanner/"])  # adres frontu

# Parametry i modele
IMG_SIZE = (48, 48)
MODEL_PATH = "models/emotion_model_v0.04.h5"
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Wczytaj model
model = tf.keras.models.load_model(MODEL_PATH)

# Klasyfikator twarzy
facecasc = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Cascade classifier loaded:", not facecasc.empty())

# API endpoint
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')
        image = np.array(image)

        print("Received image shape:", image.shape)
        print("Image dtype:", image.dtype)
        
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        print("Faces detected:", faces)

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv.resize(roi_gray, IMG_SIZE)
        input_image = np.expand_dims(np.expand_dims(roi_resized, -1), 0)

        prediction = model.predict(input_image, verbose=0)
        print("Model prediction:", prediction)
        max_index = int(np.argmax(prediction))
        emotion = emotion_dict[max_index]
        confidence = float(np.max(prediction)) * 100

        return jsonify({'emotion': emotion, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Emotion Scanner API is working!"
    
# Start serwera
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

