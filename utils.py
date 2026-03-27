import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess an image for model prediction.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing (width, height).

    Returns:
        np.array: Preprocessed image array ready for model input.
    """
    # Load image using PIL for better format support
    image = Image.open(image_path)
    # Convert to RGB if necessary
    image = image.convert('RGB')
    # Resize image
    image = image.resize(target_size)
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array