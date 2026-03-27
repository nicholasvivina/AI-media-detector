import tensorflow as tf
import numpy as np
from utils import preprocess_image
import os

def load_model(model_path='model/detector_model.h5'):
    """
    Load the trained model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(image_path, model):
    """
    Predict if an image is real or AI-generated.

    Args:
        image_path (str): Path to the image.
        model (tf.keras.Model): Trained model.

    Returns:
        str: Prediction result ('Real' or 'AI-Generated').
        float: Confidence score.
    """
    # Preprocess image
    processed_image = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = 'AI-Generated' if prediction > 0.5 else 'Real'

    return label, confidence

def main():
    """
    Main function to detect image.
    """
    # Load model
    model = load_model()

    # Example usage - replace with actual image path
    image_path = input("Enter the path to the image: ")
    if not os.path.exists(image_path):
        print("Image file not found.")
        return

    label, confidence = predict_image(image_path, model)
    print(f"Prediction: {label}")
    print(".2f")

if __name__ == "__main__":
    main()