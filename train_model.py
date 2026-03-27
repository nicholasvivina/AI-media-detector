import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

def create_cnn_model(input_shape=(128, 128, 3)):
    """
    Create a CNN model for binary classification.

    Args:
        input_shape (tuple): Shape of input images.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    """
    Train the CNN model using ImageDataGenerator.
    """
    # Data directories
    train_dir = 'dataset'
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    # ImageDataGenerator for data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% for validation
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Create model
    model = create_cnn_model()

    # Train model
    fit_kwargs = {
        'x': train_generator,
        'epochs': 10,  # Adjust as needed
    }
    if len(validation_generator) > 0:
        fit_kwargs['validation_data'] = validation_generator
    
    history = model.fit(**fit_kwargs)

    # Save model
    model_path = os.path.join(model_dir, 'detector_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()