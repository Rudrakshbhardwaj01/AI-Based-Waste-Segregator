# Project Structure

This document describes the complete structure of the AI Waste Segregation project.

```
AI waste management/                    # Project root
│
├── README.md                           # Main documentation
├── PROJECT_STRUCTURE.md                # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── setup.py                            # Setup script
├── download_dataset.py                 # Dataset download helper
├── config.py                           # Configuration settings
│
├── train_model.py                      # Model training script
├── test_model.py                       # Model testing script
├── app.py                              # Streamlit web application
├── utils.py                            # Utility functions
│
├── data/                               # Dataset directory (not in repo)
│   ├── cardboard/                      # Cardboard images
│   ├── glass/                          # Glass images
│   ├── metal/                          # Metal images
│   ├── paper/                           # Paper images
│   ├── plastic/                        # Plastic images
│   └── trash/                          # General waste images
│
├── models/                             # Trained models (created after training)
│   ├── waste_classifier_best.h5       # Best model (by validation accuracy)
│   ├── waste_classifier_final.h5      # Final model (after all epochs)
│   ├── training_history.png            # Training plots
│   └── confusion_matrix.png            # Confusion matrix
│
└── samples/                            # Sample test images (optional)
    ├── sample1.jpg
    ├── sample2.jpg
    └── ...
```

## File Descriptions

### Core Python Files

- **`app.py`**: Streamlit web application for interactive waste classification
- **`train_model.py`**: Complete training pipeline with data loading, augmentation, and model training
- **`utils.py`**: Helper functions for preprocessing, data loading, and prediction
- **`config.py`**: Centralized configuration settings

### Scripts

- **`setup.py`**: Automated setup script to install dependencies and create directories
- **`download_dataset.py`**: Helper script to download the TrashNet dataset
- **`test_model.py`**: Command-line tool to test model on individual images

### Documentation

- **`README.md`**: Complete project documentation with installation and usage instructions
- **`PROJECT_STRUCTURE.md`**: This file - describes the project structure

### Configuration

- **`requirements.txt`**: Python package dependencies
- **`.gitignore`**: Git ignore rules to exclude large files and directories

## Key Directories

### `data/` (Not in Repository)
Contains the TrashNet dataset with 6 class subdirectories:
- Approximately 2,500 total images
- Required for training
- Download separately (see README.md)

### `models/` (Created After Training)
Contains trained model files and evaluation plots:
- Model checkpoints (.h5 files)
- Training history plots
- Confusion matrix visualization

### `samples/` (Optional)
Optional directory for sample test images to use in the Streamlit app's "Sample Predictions" feature.

## Workflow

1. **Setup**: Run `python setup.py` to install dependencies
2. **Download Data**: Download TrashNet dataset to `data/` directory
3. **Train**: Run `python train_model.py` to train the model
4. **Test**: Run `python test_model.py <image_path>` to test on an image
5. **Deploy**: Run `streamlit run app.py` to launch the web app

## Environment

- Python 3.8+
- Virtual environment recommended
- CUDA GPU optional (for faster training)

