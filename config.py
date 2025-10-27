"""
AI-Powered Waste Segregation - Configuration File
=================================================
Centralized configuration settings for the waste segregation system.
"""

# Model Configuration
MODEL_CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.0001,
    'num_classes': 6,
    'input_shape': (224, 224, 3),
}

# Class Names (matching TrashNet dataset)
CLASS_NAMES = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash'
]

# Display names for UI
DISPLAY_NAMES = {
    'cardboard': 'Cardboard',
    'glass': 'Glass',
    'metal': 'Metal',
    'paper': 'Paper',
    'plastic': 'Plastic',
    'trash': 'General Waste'
}

# Class colors for visualization (RGB)
CLASS_COLORS = {
    'cardboard': (139, 90, 43),      # Brown
    'glass': (79, 79, 79),          # Gray
    'metal': (192, 192, 192),       # Silver
    'paper': (255, 255, 224),       # Beige
    'plastic': (255, 192, 203),     # Pink
    'trash': (128, 128, 128)        # Dark gray
}

# Model Paths
MODEL_PATHS = {
    'best': 'models/waste_classifier_best.h5',
    'final': 'models/waste_classifier_final.h5',
    'training_history': 'models/training_history.png',
    'confusion_matrix': 'models/confusion_matrix.png'
}

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Training Callbacks Configuration
CALLBACKS_CONFIG = {
    'early_stopping_patience': 10,
    'early_stopping_monitor': 'val_loss',
    'checkpoint_monitor': 'val_accuracy',
    'restore_best_weights': True
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'AI Waste Segregation',
    'page_icon': '♻️',
    'layout': 'wide',
    'theme': {
        'primaryColor': '#2E8B57',
        'backgroundColor': '#FFFFFF',
        'secondaryBackgroundColor': '#F0F2F6',
        'textColor': '#262730',
        'font': 'sans serif'
    }
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'min_accuracy': 0.85,
    'max_inference_time_ms': 100,
    'model_size_mb': 15
}

