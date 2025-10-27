"""
AI-Powered Waste Segregation - Utility Functions
=================================================
Helper functions for data preprocessing, augmentation, and label mapping.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Image settings
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Class mapping for TrashNet dataset
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = 6

def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size
    
    Returns:
        Preprocessed image array
    """
    # Load image
    img = load_img(image_path, target_size=(img_size, img_size))
    
    # Convert to array
    img_array = img_to_array(img)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    return img_array

def load_and_preprocess_data(data_dir, img_size=IMG_SIZE):
    """
    Load and preprocess all images from the dataset directory.
    
    Args:
        data_dir: Root directory containing class subdirectories
        img_size: Target image size
    
    Returns:
        X_train, y_train, X_val, y_val, class_names
    """
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found!")
    
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    print(f"Loading images from {len(class_names)} classes...")
    
    # Load images and labels
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        print(f"  Loading {class_name}...", end=' ')
        count = 0
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)
                
                try:
                    # Preprocess image
                    img_array = preprocess_image(img_path, img_size)
                    images.append(img_array)
                    labels.append(class_idx)
                    count += 1
                except Exception as e:
                    print(f"\nError loading {img_path}: {e}")
                    continue
        
        print(f"{count} images")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # One-hot encode labels
    y_categorical = to_categorical(y, num_classes=len(class_names))
    
    # Split into train and validation sets (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val, class_names

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators with augmentation for training.
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Batch size for generators
    
    Returns:
        train_generator, val_generator
    """
    
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation set (only rescale)
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator

def predict_waste(image_path, model):
    """
    Predict waste category for a single image.
    
    Args:
        image_path: Path to the image file
        model: Trained Keras model
    
    Returns:
        predicted_class: Name of the predicted class
        confidence: Confidence score (0-1)
    """
    
    # Preprocess image
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get class with highest probability
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Map to class name
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    return predicted_class, confidence

def load_trained_model(model_path='models/waste_classifier_best.h5'):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded Keras model
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    model = tf.keras.models.load_model(model_path)
    return model

def get_class_color(class_name):
    """
    Get color for a waste class (for visualization).
    
    Args:
        class_name: Name of the waste class
    
    Returns:
        Color code (RGB tuple)
    """
    
    color_map = {
        'cardboard': (139, 90, 43),     # Brown
        'glass': (79, 79, 79),          # Gray
        'metal': (192, 192, 192),       # Silver
        'paper': (255, 255, 224),       # Beige
        'plastic': (255, 192, 203),     # Pink
        'trash': (128, 128, 128)        # Dark gray
    }
    
    return color_map.get(class_name.lower(), (128, 128, 128))

def get_class_display_name(class_name):
    """
    Get display-friendly name for a waste class.
    
    Args:
        class_name: Name of the waste class
    
    Returns:
        Display-friendly name
    """
    
    display_map = {
        'cardboard': 'Cardboard',
        'glass': 'Glass',
        'metal': 'Metal',
        'paper': 'Paper',
        'plastic': 'Plastic',
        'trash': 'General Waste'
    }
    
    return display_map.get(class_name.lower(), class_name.title())

def download_dataset(dataset_url, output_dir='data'):
    """
    Helper function to download the TrashNet dataset.
    Note: This is a placeholder. In practice, you would implement
    actual download logic or use kaggle API.
    
    Args:
        dataset_url: URL or command to download dataset
        output_dir: Directory to save the dataset
    """
    
    print("To download the TrashNet dataset, you can:")
    print("1. Visit: https://github.com/garythung/trashnet")
    print("2. Download from Kaggle: https://www.kaggle.com/datasets/techsash/waste-classification-data")
    print("3. Run: python download_dataset.py")
    print(f"\nExtract the dataset to: {output_dir}/")

# Additional helper for processing video frames
def preprocess_frame(frame, img_size=IMG_SIZE):
    """
    Preprocess a video frame for prediction.
    
    Args:
        frame: OpenCV frame (BGR format)
        img_size: Target image size
    
    Returns:
        Preprocessed image array ready for model input
    """
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
    
    # Normalize to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # Expand dimensions for batch
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    
    return frame_expanded

