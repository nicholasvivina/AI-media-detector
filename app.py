import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from utils import preprocess_image
import os
import tempfile

def load_model():
    """
    Load the trained model.
    """
    model_path = 'model/detector_model.h5'
    if not os.path.exists(model_path):
        st.error("Model not found. Please train the model first by running train_model.py")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image_file(image_file, model):
    """
    Predict for uploaded image file.

    Args:
        image_file: Uploaded file from Streamlit.
        model: Trained model.

    Returns:
        str: Prediction result.
        float: Confidence.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_file.read())
        temp_path = temp_file.name

    try:
        label, confidence = predict_image(temp_path, model)
    finally:
        os.unlink(temp_path)

    return label, confidence

def predict_image(image_path, model):
    """
    Predict if an image is real or AI-generated.

    Args:
        image_path (str): Path to the image.
        model: Trained model.

    Returns:
        str: Label.
        float: Confidence.
    """
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = 'AI-Generated' if prediction > 0.5 else 'Real'
    return label, confidence

def predict_video_file(video_file, model):
    """
    Predict for uploaded video file by analyzing frames.

    Args:
        video_file: Uploaded file from Streamlit.
        model: Trained model.

    Returns:
        list: List of predictions for frames.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name

    try:
        predictions = analyze_video(temp_path, model)
    finally:
        os.unlink(temp_path)

    return predictions

def analyze_video(video_path, model, frame_skip=30):
    """
    Analyze video by predicting on selected frames.

    Args:
        video_path (str): Path to video.
        model: Trained model.
        frame_skip (int): Skip frames for efficiency.

    Returns:
        list: List of (frame_num, label, confidence) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                pil_image.save(temp_file.name)
                temp_path = temp_file.name

            try:
                label, confidence = predict_image(temp_path, model)
                predictions.append((frame_num, label, confidence))
            finally:
                os.unlink(temp_path)

        frame_num += 1

    cap.release()
    return predictions

def main():
    st.title("AI Generated Image and Video Detection")
    st.write("Upload an image or video to detect if it's AI-generated or real.")

    # Load model
    model = load_model()
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        if file_type == 'image':
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            if st.button('Detect'):
                with st.spinner('Detecting...'):
                    label, confidence = predict_image_file(uploaded_file, model)
                st.success(f"Prediction: {label}")
                st.write(".2f")

        elif file_type == 'video':
            st.video(uploaded_file)
            if st.button('Analyze Video'):
                with st.spinner('Analyzing video frames...'):
                    predictions = predict_video_file(uploaded_file, model)
                st.write("Frame Analysis Results:")
                for frame_num, label, conf in predictions:
                    st.write(f"Frame {frame_num}: {label} ({conf:.2f})")

if __name__ == "__main__":
    main()