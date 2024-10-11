from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('emotion_recognition_cnn.h5')

# Emotion labels corresponding to your model's outputs
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/analyze-expression', methods=['POST'])
def analyze_expression():
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image']

    # Process the image data
    # Extract base64 image data
    header, encoded = image_data.split(',', 1)
    image = base64.b64decode(encoded)
    np_img = np.frombuffer(image, np.uint8)

    # Decode the image
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (48, 48))  # Resize according to your model input
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map predicted class to emotion
    emotion = emotion_labels[predicted_class]

    return jsonify({'mood': emotion})

if __name__ == '__main__':
    app.run(debug=True)
