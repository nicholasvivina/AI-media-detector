# AI Generated Image and Video Detection

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to detect whether images and videos are AI-generated or real.

## Features

- Train a CNN model to classify images as real or AI-generated
- Detect AI content in individual images
- Analyze videos frame by frame for AI detection
- Web interface using Streamlit for easy uploading and prediction

## Project Structure

```
ai_media_detector/
    dataset/
        real/          # Real images for training
        fake/          # AI-generated images for training
    model/             # Saved trained model
    train_model.py     # Script to train the CNN model
    detect_image.py    # Script to detect AI in images
    detect_video.py    # Script to detect AI in videos
    utils.py           # Utility functions for preprocessing
    app.py             # Streamlit web application
    requirements.txt   # Python dependencies
    README.md          # This file
```

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Create folders `dataset/real/` and `dataset/fake/`
2. Place real images in `dataset/real/`
3. Place AI-generated images in `dataset/fake/`

Example datasets you can use:
- Real images: Photos from ImageNet, COCO dataset, or personal photos
- AI-generated images: Images from DALL-E, Midjourney, Stable Diffusion outputs

Ensure you have at least 100-200 images per class for training.

## Training the Model

Run the training script:

```bash
python train_model.py
```

This will:
- Load images from the dataset folder
- Train a CNN model for 10 epochs
- Save the trained model as `model/detector_model.h5`

## Running Detection Scripts

### Image Detection

```bash
python detect_image.py
```

Enter the path to an image when prompted.

### Video Detection

```bash
python detect_video.py
```

Enter the path to a video. The script will display frames with predictions overlaid.

## Running the Streamlit App

```bash
streamlit run app.py
```

Open the provided URL in your browser to use the web interface for uploading images or videos.

## Requirements

- Python 3.7+
- TensorFlow 2.13+
- OpenCV
- NumPy
- Pillow
- Streamlit

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with Dropout
- Sigmoid output for binary classification

## Notes

- The model is trained on 128x128 images. Larger images will be resized.
- For videos, frames are analyzed every 30 frames by default for efficiency.
- The model assumes binary classification (real vs AI-generated).

## License

This project is open-source. Feel free to modify and distribute.