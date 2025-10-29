"""
AI-Powered Waste Segregation - Model Testing Script
====================================================
Test the trained model on a single image or batch of images.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import load_trained_model, preprocess_image, CLASS_NAMES, get_class_display_name, configure_gpu


def predict_single_image(model, image_path):
    """
    Predict waste category for a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
    
    Returns:
        predicted_class: Predicted waste category
        confidence: Confidence score
        all_probabilities: Dictionary of all class probabilities
    """
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None, None, None
    
    # Preprocess image
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get predictions
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    # Create probability dictionary
    all_probabilities = {
        get_class_display_name(CLASS_NAMES[i]): predictions[0][i]
        for i in range(len(CLASS_NAMES))
    }
    
    return predicted_class, confidence, all_probabilities


def visualize_prediction(image_path, predicted_class, confidence, all_probabilities):
    """
    Visualize prediction with image and bar chart.
    
    Args:
        image_path: Path to the image
        predicted_class: Predicted class name
        confidence: Confidence score
        all_probabilities: Dictionary of all probabilities
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title(f"Input Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display prediction
    ax1.text(
        0.5, -0.05,
        f"Prediction: {get_class_display_name(predicted_class)}\nConfidence: {confidence*100:.1f}%",
        transform=ax1.transAxes,
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Bar chart of probabilities
    classes = list(all_probabilities.keys())
    probs = list(all_probabilities.values())
    colors = ['green' if p == max(probs) else 'gray' for p in probs]
    
    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    for i, v in enumerate(probs):
        ax2.text(v + 0.01, i, f'{v*100:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('test_prediction.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to: test_prediction.png")
    plt.show()


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Test trained waste classification model')
    parser.add_argument('image_path', type=str, help='Path to the image file to test')
    parser.add_argument('--model', type=str, default='models/waste_classifier_best.h5',
                       help='Path to the trained model file')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of prediction')
    
    args = parser.parse_args()
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Please train the model first using: python train_model.py")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from: {args.model}")
    # Configure GPU for inference (silent)
    configure_gpu(announce=False, context_label="Inference")
    model = load_trained_model(args.model)
    
    # Make prediction
    print(f"\nAnalyzing image: {args.image_path}")
    predicted_class, confidence, all_probabilities = predict_single_image(model, args.image_path)
    
    if predicted_class is None:
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted Class: {get_class_display_name(predicted_class)}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Probabilities:")
    for class_name, prob in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob*100:.2f}%")
    print("=" * 60)
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(args.image_path, predicted_class, confidence, all_probabilities)


if __name__ == '__main__':
    main()

