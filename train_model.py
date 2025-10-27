"""
AI-Powered Waste Segregation - Model Training Script
=====================================================
This script handles dataset loading, model training, evaluation, and saving.
Uses MobileNetV2 for transfer learning on trash classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from utils import load_and_preprocess_data, create_data_generators

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_model(num_classes=6, img_size=224, learning_rate=0.0001):
    """
    Build a MobileNetV2-based model for waste classification.
    
    Args:
        num_classes: Number of output classes (default: 6 for TrashNet)
        img_size: Input image size (default: 224x224)
        learning_rate: Initial learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 (excluding top layer)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model layers (optional - can fine-tune later)
    base_model.trainable = True
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

def train_model(data_dir='data', epochs=2, batch_size=32, img_size=224):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
    """
    
    print("=" * 60)
    print("AI WASTE SEGREGATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, class_names = load_and_preprocess_data(
        data_dir, img_size
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Classes: {class_names}")
    
    # Build model
    print("\n[2/5] Building model...")
    model = build_model(num_classes=len(class_names), img_size=img_size)
    model.summary()
    
    # Create callbacks
    print("\n[3/5] Setting up callbacks...")
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint to save best model
    checkpoint = ModelCheckpoint(
        'models/waste_classifier_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Data augmentation and generators
    print("\n[4/5] Setting up data generators...")
    train_generator, val_generator = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Train model
    print("\n[5/5] Starting training...")
    print("=" * 60)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Save final model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/waste_classifier_final.h5')
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = model.evaluate(val_generator, verbose=0)
    print(f"\nValidation Accuracy: {eval_results[1]:.4f}")
    print(f"Validation Loss: {eval_results[0]:.4f}")
    
    # Generate predictions for classification report
    y_pred = model.predict(val_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names
    ))
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved at: models/waste_classifier_best.h5")
    print(f"Final model saved at: models/waste_classifier_final.h5")
    
    return model, history

def plot_training_history(history):
    """Plot training and validation accuracy/loss over epochs."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved to: models/training_history.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to: models/confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    import sys
    
    # Default parameters
    data_dir = 'data'
    epochs = 2
    batch_size = 32
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    
    print(f"Starting training with:")
    print(f"  Data directory: {data_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    train_model(data_dir, epochs, batch_size)

