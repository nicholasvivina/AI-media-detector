import cv2
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

def predict_frame(frame, model):
    """
    Predict if a frame is real or AI-generated.

    Args:
        frame (np.array): Video frame.
        model (tf.keras.Model): Trained model.

    Returns:
        str: Prediction result ('Real' or 'AI-Generated').
        float: Confidence score.
    """
    # Convert frame to PIL Image for preprocessing
    from PIL import Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Save temporarily to use preprocess_image (or modify utils to accept array)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        pil_image.save(temp_file.name)
        temp_path = temp_file.name

    try:
        processed_image = preprocess_image(temp_path)
        prediction = model.predict(processed_image)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = 'AI-Generated' if prediction > 0.5 else 'Real'
    finally:
        os.unlink(temp_path)

    return label, confidence

def detect_video(video_path, model):
    """
    Detect AI-generated content in video frames.

    Args:
        video_path (str): Path to the video file.
        model (tf.keras.Model): Trained model.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict on frame
        label, confidence = predict_frame(frame, model)

        # Display prediction on frame
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('AI Media Detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to detect video.
    """
    # Load model
    model = load_model()

    # Example usage - replace with actual video path
    video_path = input("Enter the path to the video: ")
    if not os.path.exists(video_path):
        print("Video file not found.")
        return

    detect_video(video_path, model)

if __name__ == "__main__":
    main()