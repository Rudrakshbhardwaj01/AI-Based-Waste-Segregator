# Quick Start Guide

Get up and running with the AI Waste Segregation system in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for downloading dependencies and dataset)

## Step-by-Step Setup

### 1. Setup Python Environment (2 minutes)

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset (5 minutes)

**Option A: Using Kaggle (Recommended)**

1. Visit: https://www.kaggle.com/datasets/techsash/waste-classification-data
2. Click "Download" button
3. Extract the ZIP file
4. Rename the extracted folder to `data`
5. Place it in the project root

**Option B: Using GitHub**

1. Visit: https://github.com/garythung/trashnet
2. Follow their instructions to download
3. Organize into the `data/` directory with subdirectories

**Expected structure after download:**
```
data/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

### 3. Train the Model (30-120 minutes)

```bash
python train_model.py
```

This will:
- Load and preprocess all images
- Train the MobileNetV2 model
- Save the best model to `models/waste_classifier_best.h5`
- Generate training plots

**Note**: Training time depends on your system:
- CPU: ~2-4 hours
- GPU: ~20-40 minutes

### 4. Run the Web App (Instant)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Testing the Model (Optional)

Test on a single image:

```bash
python test_model.py path/to/image.jpg --visualize
```

## What You Can Do

### In the Web App:

1. **Upload Image**: Upload a photo of a waste item
2. **Webcam Capture**: Take a photo using your webcam
3. **Sample Predictions**: View predictions on test images

### From Command Line:

```bash
# Train model with custom parameters
python train_model.py data 50 32
# Arguments: [data_dir] [epochs] [batch_size]

# Test single image
python test_model.py samples/plastic_bottle.jpg --visualize
```

## Expected Results

After training, you should achieve:
- **Validation Accuracy**: 85-90%
- **Model Size**: ~15 MB
- **Inference Speed**: <100ms per image (CPU)

## Troubleshooting

### Problem: "Model not found"
**Solution**: Train the model first: `python train_model.py`

### Problem: "Dataset not found"
**Solution**: Download the TrashNet dataset to the `data/` directory

### Problem: "Out of memory"
**Solution**: Reduce batch size in `train_model.py` (change `batch_size=32` to `16` or `8`)

### Problem: Streamlit not working
**Solution**: Update streamlit: `pip install --upgrade streamlit`

## Next Steps

- ✅ Read the full documentation in `README.md`
- ✅ Explore the code in `train_model.py`, `app.py`, `utils.py`
- ✅ Add your own test images to `samples/` directory
- ✅ Customize model architecture in `train_model.py`
- ✅ Adjust training parameters in `config.py`

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review error messages carefully
3. Ensure all dependencies are installed correctly
4. Verify dataset is properly structured

---

**You're all set! Happy classifying! ♻️**

