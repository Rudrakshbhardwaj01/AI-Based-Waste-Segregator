"""
AI-Powered Waste Segregation - Streamlit Web Application
========================================================
Interactive web app for waste classification using uploaded images or webcam.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from utils import (
    load_trained_model, 
    CLASS_NAMES, 
    NUM_CLASSES,
    get_class_display_name,
    get_class_color,
    preprocess_frame
)

# Page configuration
st.set_page_config(
    page_title="AI Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .confidence-bar {
        background-color: #4CAF50;
        height: 30px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the trained model with caching."""
    model_path = 'models/waste_classifier_best.h5'
    
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found! Please train the model first using train_model.py")
        return None
    
    model = load_trained_model(model_path)
    return model

def predict_image(model, image):
    """
    Predict waste category for an image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    predictions = model.predict(image_batch, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[0][predicted_idx]
    
    return predicted_class, confidence, predictions[0]

def main():
    """Main Streamlit application."""
    st.markdown('<div class="main-header">♻️ AI Waste Segregation System</div>', unsafe_allow_html=True)
    st.markdown("### Computer Vision-Powered Recyclable Material Detection")
    
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Upload Options")
        upload_option = st.radio(
            "Choose input method:",
            ["📁 Upload Image", "📷 Webcam Capture", "🎲 Sample Predictions"]
        )
    
    if upload_option == "📁 Upload Image":
        st.subheader("Upload a waste item image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a waste item (plastic, metal, paper, glass, cardboard, or trash)"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, all_predictions = predict_image(model, image)
                
                display_name = get_class_display_name(predicted_class)
                confidence_pct = confidence * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction: {display_name}</h2>
                    <p style="font-size: 2rem; font-weight: bold; color: #2E8B57;">
                        {confidence_pct:.1f}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(float(confidence))
                
                st.markdown("### All Class Probabilities")
                for idx, class_name in enumerate(CLASS_NAMES):
                    prob = all_predictions[idx] * 100
                    st.write(f"**{get_class_display_name(class_name)}:** {prob:.1f}%")
    
    elif upload_option == "📷 Webcam Capture":
        st.subheader("Capture image from webcam")
        picture = st.camera_input("Take a picture", key="webcam_input")
        
        if picture is not None:
            image = Image.open(picture)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)
            
            with col2:
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, all_predictions = predict_image(model, image)
                
                display_name = get_class_display_name(predicted_class)
                confidence_pct = confidence * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction: {display_name}</h2>
                    <p style="font-size: 2rem; font-weight: bold; color: #2E8B57;">
                        {confidence_pct:.1f}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # ✅ Fixed progress bar
                st.progress(float(confidence))
                
                st.markdown("### 📊 All Class Probabilities")
                for idx, class_name in enumerate(CLASS_NAMES):
                    prob = all_predictions[idx] * 100
                    st.write(f"**{get_class_display_name(class_name)}:** {prob:.1f}%")
    
    elif upload_option == "🎲 Sample Predictions":
        st.subheader("Sample predictions from test set")
        st.info("📌 Note: Place sample test images in a 'samples' directory for demonstration.")
        
        samples_dir = 'samples'
        
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if sample_files:
                st.write(f"Found {len(sample_files)} sample images")
                
                for filename in sample_files:
                    with st.expander(f"📸 {filename}"):
                        img_path = os.path.join(samples_dir, filename)
                        image = Image.open(img_path)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.image(image, use_container_width=True)
                        
                        with col2:
                            with st.spinner("Analyzing..."):
                                predicted_class, confidence, all_predictions = predict_image(model, image)
                            
                            display_name = get_class_display_name(predicted_class)
                            st.success(f"**Prediction:** {display_name}")
                            st.info(f"**Confidence:** {confidence*100:.1f}%")
                            
                            st.bar_chart(
                                {get_class_display_name(name): prob*100 
                                 for name, prob in zip(CLASS_NAMES, all_predictions)}
                            )
            else:
                st.warning("No sample images found in 'samples' directory")
        else:
            st.warning("'samples' directory not found. Create it and add test images.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with TensorFlow, Keras, and Streamlit</p>
        <p>Model: MobileNetV2 | Dataset: TrashNet</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
