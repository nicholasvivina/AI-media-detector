import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import random

# -------------------------------
# Dummy Model (No TensorFlow)
# -------------------------------
def load_model():
    return "dummy_model"


# -------------------------------
# Random Prediction (Image)
# -------------------------------
def predict_image(image_path, model):
    prediction = random.random()
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = 'AI-Generated' if prediction > 0.5 else 'Real'
    return label, confidence


def predict_image_file(image_file, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(image_file.read())
        temp_path = temp_file.name

    try:
        label, confidence = predict_image(temp_path, model)
    finally:
        os.unlink(temp_path)

    return label, confidence


# -------------------------------
# Video Prediction (Random)
# -------------------------------
def analyze_video(video_path, model, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_skip == 0:
            prediction = random.random()
            confidence = prediction if prediction > 0.5 else 1 - prediction
            label = 'AI-Generated' if prediction > 0.5 else 'Real'
            predictions.append((frame_num, label, confidence))

        frame_num += 1

    cap.release()
    return predictions


def predict_video_file(video_file, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name

    try:
        predictions = analyze_video(temp_path, model)
    finally:
        os.unlink(temp_path)

    return predictions


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title("AI Generated Image and Video Detection")
    st.write("Upload an image or video to detect if it's AI-generated or real.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]

        if file_type == 'image':
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            if st.button('Detect'):
                with st.spinner('Detecting...'):
                    label, confidence = predict_image_file(uploaded_file, model)

                st.success(f"Prediction: {label}")
                st.write(f"Confidence: {confidence:.2f}")

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