from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from typing import Tuple, Dict, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable for the model
model = None

def load_model() -> None:
    """Load the trained model."""
    global model
    try:
        model = keras.models.load_model('mnist_model.h5')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check() -> Tuple[Dict[str, str], int]:
    """Health check endpoint."""
    return {"status": "healthy"}, 200

@app.route('/predict', methods=['POST'])
def predict() -> Tuple[Dict[str, Union[str, int, List[float]]], int]:
    """Prediction endpoint."""
    try:
        data = request.json
        
        # Input validation
        if 'image' not in data:
            return {"error": "No image provided"}, 400
            
        # Prepare input data
        image_data = np.array(data['image'])
        if image_data.shape != (784,):
            return {"error": "Image must be a 1D array of 784 values"}, 400
            
        # Preprocess
        image_data = image_data.reshape(1, 784)
        image_data = image_data.astype("float32") / 255.0
        
        # Make prediction
        prediction = model.predict(image_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        return {
            "prediction": int(predicted_class),
            "probabilities": prediction.tolist()[0]
        }, 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "Internal server error"}, 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Run app
    app.run(host='0.0.0.0', port=5000)