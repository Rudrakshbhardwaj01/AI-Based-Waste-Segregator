# 📥 How to Download the Dataset

## Quick Answer

Download the TrashNet dataset from one of these sources:

1. **Kaggle** (Recommended - Easiest): https://www.kaggle.com/datasets/techsash/waste-classification-data
2. **GitHub**: https://github.com/garythung/trashnet

---

## Method 1: Kaggle (Recommended) 🎯

### Step-by-Step Instructions:

1. **Visit the Kaggle Dataset Page**
   - Go to: https://www.kaggle.com/datasets/techsash/waste-classification-data

2. **Download the Dataset**
   - Click the "Download" button (top right)
   - You may need to sign in to Kaggle (free account)
   - Choose "Download All" when prompted

3. **Extract the ZIP File**
   - Extract the downloaded `waste-classification-data.zip` file
   - You should see a folder structure like:
     ```
     waste-classification-data/
     ├── DATASET/
     │   ├── TEST/
     │   │   ├── O/
     │   │   └── R/
     │   └── TRAIN/
     │       ├── O/
     │       └── R/
     ```

4. **Reorganize the Data**
   - The Kaggle version has "O" (organic) and "R" (recyclable) folders
   - For our 6-class model, download the **full TrashNet dataset** instead

### Better Option - Use Original TrashNet:

1. **Visit Original GitHub**: https://github.com/garythung/trashnet
2. Follow their download instructions
3. The dataset should have folders: cardboard, glass, metal, paper, plastic, trash

---

## Method 2: Direct Download (Alternative)

### Option A: Using Python Script

Run the helper script:

```bash
python download_dataset.py
```

This will show you download instructions.

### Option B: Manual Download

1. Visit: https://github.com/garythung/trashnet
2. Download the "data-original" folder
3. Extract to your project directory

---

## Expected Directory Structure

After downloading, your project should look like this:

```
AI waste management/
├── data/                    # ← You need to create this
│   ├── cardboard/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── glass/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
```

### Important Notes:

- Make sure the folder is named **`data`** (lowercase)
- Each class should be in its own subdirectory
- Images can be .jpg, .jpeg, or .png
- Total: ~2,500 images across 6 classes

---

## Dataset Information

**TrashNet Dataset** (Original Research):
- **Source**: Gary Thung, Mindy Yang
- **Paper**: https://arxiv.org/abs/1604.07975
- **GitHub**: https://github.com/garythung/trashnet

**Class Distribution**:
- Cardboard: ~400 images
- Glass: ~500 images
- Metal: ~400 images
- Paper: ~595 images
- Plastic: ~482 images
- Trash: ~140 images

---

## Alternative Dataset Options

If you can't access TrashNet, here are alternatives:

### 1. Taco Dataset (waste object detection)
- URL: http://tacodataset.org/

### 2. WasteNet Dataset
- Search for "waste classification dataset" on Kaggle
- Several alternatives available

---

## Troubleshooting

### Problem: "Dataset not found"
**Solution**: 
1. Create a folder named `data` in your project root
2. Download and extract the dataset into it
3. Make sure each class has its own subdirectory

### Problem: Can't access Kaggle
**Solution**:
1. Create a free Kaggle account
2. Or download from GitHub directly
3. Or use an alternative dataset

### Problem: "No images found"
**Solution**:
1. Check that images are in .jpg, .jpeg, or .png format
2. Verify the directory structure is correct
3. Make sure images are actually in the class folders

---

## Quick Start After Download

Once you have the dataset:

```bash
# Verify dataset structure
python -c "import os; print(os.listdir('data'))"
# Should show: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Train the model
python train_model.py

# Run the web app
streamlit run app.py
```

---

## Need Help?

If you encounter issues:
1. Check the `README.md` file
2. Verify your directory structure matches the expected format
3. Make sure images are properly named (.jpg, .png, etc.)
4. Run `python download_dataset.py` for instructions

---

**Note**: The dataset is approximately 200-500 MB in size. Make sure you have enough disk space!

