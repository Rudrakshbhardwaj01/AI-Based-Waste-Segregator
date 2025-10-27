# AI-Powered Waste Segregation System

A production-grade, end-to-end computer vision system that automatically detects and classifies recyclable materials from general waste. This project uses deep learning to identify different types of waste: **Plastic**, **Metal**, **Paper**, **Glass**, **Cardboard**, and **General Waste**.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

##  Overview

This system uses a fine-tuned **MobileNetV2** convolutional neural network (CNN) to classify waste items into 6 categories. The model is lightweight, accurate, and suitable for deployment on edge devices.

**Goal**: Classify waste items with **85-90% accuracy** for real-world deployment.

##  Features

- **Multi-class waste classification** (6 categories)
- **Transfer learning** with MobileNetV2 (pre-trained on ImageNet)
- **Data augmentation** for improved robustness
- **Interactive web interface** with Streamlit
- **Image upload** and **webcam capture** support
- **Visual predictions** with confidence scores
- **Production-ready** code with error handling

## 🛠 Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **Streamlit** - Web UI
- **NumPy, Pandas** - Data processing
- **Matplotlib, Seaborn** - Visualizations
- **scikit-learn** - Data splitting and evaluation

## Model Architecture

### Base Model: MobileNetV2

- **Pretrained on**: ImageNet (1.4M images, 1000 classes)
- **Input size**: 224×224×3 (RGB images)
- **Parameters**: ~3.4M (lightweight for deployment)

### Custom Classification Head

```
MobileNetV2 (frozen features)
    ↓
Global Average Pooling (GAP)
    ↓
Dropout (0.5)
    ↓
Dense(128, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense(6, Softmax) → Output
```

**Optimizer**: Adam (learning_rate=0.0001)  
**Loss**: Categorical Cross-Entropy  
**Metrics**: Accuracy, Top-K Categorical Accuracy

## Dataset

### TrashNet Dataset

- **Source**: [Gary Thung's TrashNet](https://github.com/garythung/trashnet)
- **Total images**: ~2,500
- **Classes**: 6 categories
  - Cardboard (~400 images)
  - Glass (~500 images)
  - Metal (~400 images)
  - Paper (~595 images)
  - Plastic (~482 images)
  - Trash (~140 images)
- **Split**: 80% training, 20% validation

### Data Augmentation

Training images are augmented to improve generalization:
- Rotation (±20°)
- Width/height shifts (±20%)
- Shear (±20°)
- Zoom (±20%)
- Horizontal flip

##  Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "AI waste management"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Download the Dataset** (see below)
2. **Train the Model** (or use pretrained)
3. **Run the Streamlit App**

### 1. Download TrashNet Dataset

The TrashNet dataset is available from:

- **Kaggle**: https://www.kaggle.com/datasets/techsash/waste-classification-data
- **GitHub**: https://github.com/garythung/trashnet

**Expected directory structure:**

```
data/
├── cardboard/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── glass/
│   ├── 0.jpg
│   └── ...
├── metal/
├── paper/
├── plastic/
└── trash/
```

### 2. Train the Model

```bash
python train_model.py
```

**Options:**
```bash
python train_model.py data 50 32
# Arguments: [data_dir] [epochs] [batch_size]
```

**Output:**
- Trained model: `models/waste_classifier_best.h5`
- Training plots: `models/training_history.png`
- Confusion matrix: `models/confusion_matrix.png`

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- **Upload Image**: Upload a waste item photo
- **Webcam Capture**: Take a photo using your webcam
- **Sample Predictions**: View predictions on test images

## Training

### Training Pipeline

1. **Data Loading**: Load images from class directories
2. **Preprocessing**: Resize to 224×224, normalize [0, 1]
3. **Augmentation**: Apply transformations to training set
4. **Training**: Fine-tune MobileNetV2 with early stopping
5. **Evaluation**: Generate classification report and confusion matrix

### Callbacks

- **Early Stopping**: Prevents overfitting (patience=10)
- **Model Checkpoint**: Saves best model based on validation accuracy

### Expected Training Time

- **Epochs**: 30-50 (with early stopping)
- **Time per epoch**: ~2-5 minutes (CPU) / ~30 seconds (GPU)
- **Total training time**: ~2-4 hours (CPU) / ~20-40 minutes (GPU)

##  Performance

### Target Metrics

- **Validation Accuracy**: ≥85%
- **Test Accuracy**: ≥85%
- **Inference Speed**: <100ms per image (CPU)

### Model Evaluation

After training, you'll see:
- Accuracy/loss plots over epochs
- Classification report (precision, recall, F1-score)
- Confusion matrix

## Project Structure

```
AI waste management/
├── app.py                 # Streamlit web application
├── train_model.py        # Training pipeline
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
├── models/              # Saved models (created after training)
│   ├── waste_classifier_best.h5
│   ├── waste_classifier_final.h5
│   ├── training_history.png
│   └── confusion_matrix.png
├── data/                # Dataset (download separately)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── samples/            # Sample test images (optional)
    └── *.jpg
```

## Academic Submission Notes

### Model Architecture Explained

1. **Transfer Learning**: MobileNetV2 is pretrained on ImageNet, providing powerful feature extraction.
2. **Fine-tuning**: We add a custom classification head to adapt to 6 waste classes.
3. **Regularization**: Dropout layers (50%, 30%) prevent overfitting on small dataset.
4. **Optimization**: Adam optimizer with low learning rate (0.0001) for stable training.

### Key Technical Decisions

- **MobileNetV2**: Lightweight, mobile-friendly architecture
- **Data Augmentation**: Critical for generalization with limited data
- **Early Stopping**: Prevents overfitting automatically
- **Categorical Cross-Entropy**: Standard for multi-class classification

### Dataset

- **TrashNet**: Open-source dataset with real waste images
- **Class imbalance**: Handled through stratified sampling
- **Data augmentation**: Reduces overfitting on limited samples

## Future Enhancements

- [ ] Add more classes (batteries, electronics, organic waste)
- [ ] Object detection (multiple items per image)
- [ ] Mobile app deployment
- [ ] Real-time video stream processing
- [ ] Database integration for tracking
- [ ] Multi-angle classification
- [ ] Transfer to more robust models (EfficientNet, ResNet)

## Troubleshooting

### Common Issues

**1. Model file not found**
```bash
# Train the model first:
python train_model.py
```

**2. Dataset not found**
```bash
# Download TrashNet dataset to 'data/' directory
```

**3. Out of memory errors**
```bash
# Reduce batch size in train_model.py
# Change batch_size=32 to batch_size=16 or 8
```

**4. Streamlit issues**
```bash
# Update streamlit
pip install --upgrade streamlit
```

## License

This project is open-source and available for educational purposes.

##  Author

Rudraksh Bhardwaj

---


