#!/usr/bin/env python

import streamlit as st
import sys
import os
# import subprocess # No longer needed for pip/git
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Faster R-CNN vs Mask R-CNN Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure CBNetV2 is in the Python path if it's a submodule at the root
# This helps Python find modules within the CBNetV2 directory.
# model.py also does this for its own context, which is fine.
cbnet_path = Path("CBNetV2")
if cbnet_path.exists() and cbnet_path.is_dir():
    sys.path.insert(0, str(cbnet_path.resolve()))
else:
    # If CBNetV2 is not found, it's a critical error for the app to run.
    # This might happen if the submodule didn't checkout correctly.
    st.error(f"CRITICAL ERROR: CBNetV2 directory not found at {str(cbnet_path.resolve())}. Ensure the submodule is correctly checked out.")
    st.stop()

# Try to import required modules
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    # import io # Not used
    # import base64 # Not used
    
    # Try to import model
    try:
        from model import Model # model.py should be in the same directory as app.py
        model_available = True
    except ImportError as e:
        st.error(f"Failed to import `Model` from model.py: {e}")
        st.info("This could be due to issues with the CBNetV2 submodule path or missing dependencies from MMDetection/MMCV not covered in requirements.txt.")
        model_available = False
    except Exception as e:
        st.error(f"An unexpected error occurred while importing `Model`: {e}")
        model_available = False
        
except ImportError as e:
    st.error(f"Missing critical dependencies (e.g., numpy, torch, Pillow): {e}")
    st.info("Please ensure your `requirements.txt` is complete and Streamlit Cloud can install them.")
    model_available = False # Ensure this is set
    st.stop() # Stop if basic imports fail


# Custom CSS (keep as is)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .comparison-table {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if model_available:
    if 'model' not in st.session_state:
        try:
            with st.spinner('Loading models... This may take a few minutes on first run.'):
                st.session_state.model = Model() # This now relies on model.py
                st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.info("This might be due to missing model weight files (check `weights` directory and download logic in `model.py`), issues with model configurations, or problems with the CBNetV2/MMDetection setup.")
            model_available = False # Update status

if 'results' not in st.session_state:
    st.session_state.results = {}

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Object Detection Model Comparison</h1>
    <h3>Faster R-CNN vs Mask R-CNN</h3>
    <p>Compare performance and accuracy of two state-of-the-art object detection models</p>
</div>
""", unsafe_allow_html=True)

if not model_available:
    st.error("Models are not available. Please check the setup and error messages above.")
    st.info("""
    **Possible solutions:**
    1. Ensure your GitHub repository has the `CBNetV2` submodule correctly initialized and pushed (`git submodule update --init --recursive`, then commit and push `.gitmodules` and the `CBNetV2` reference).
    2. Verify that all dependencies in `requirements.txt` are correct and compatible.
    3. Check that model configuration files and weight download paths in `model.py` are accurate.
    4. Review the Streamlit deployment logs for specific error messages.
    """)
    st.stop()

# Sidebar (keep as is)
with st.sidebar:
    st.header("⚙️ Configuration")
    
    score_threshold = st.slider(
        "Detection Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    st.markdown("---")
    
    st.subheader("📊 Model Information")
    
    with st.expander("Faster R-CNN (DB-ResNet50)", expanded=True):
        st.markdown("""
        **Architecture:** Two-stage detector
        - **Backbone:** ResNet50 with CBNetV2
        - **Strengths:** Fast inference, good accuracy
        - **Use case:** Real-time applications
        """)
    
    with st.expander("Mask R-CNN (DB-Swin-T)", expanded=True):
        st.markdown("""
        **Architecture:** Instance segmentation
        - **Backbone:** Swin Transformer Tiny with CBNetV2
        - **Strengths:** Pixel-level segmentation
        - **Use case:** Detailed object analysis
        """)

# Main content (largely keep as is)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to test both models"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_np = np.array(image.convert("RGB")) # Ensure RGB
        
        if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
            if not model_available or 'model' not in st.session_state:
                st.error("Model is not loaded. Cannot run comparison.")
            else:
                try:
                    with st.spinner('Running detection on both models...'):
                        progress_bar = st.progress(0)
                        
                        # Model 1: Faster R-CNN
                        progress_bar.progress(25)
                        st.session_state.model.set_model_name("Faster R-CNN (DB-ResNet50)")
                        start_time = time.time()
                        results_faster, vis_faster = st.session_state.model.detect_and_visualize(
                            image_np, score_threshold
                        )
                        faster_time = time.time() - start_time
                        
                        progress_bar.progress(50)
                        
                        # Model 2: Mask R-CNN
                        progress_bar.progress(75)
                        st.session_state.model.set_model_name("Mask R-CNN (DB-Swin-T)")
                        start_time = time.time()
                        results_mask, vis_mask = st.session_state.model.detect_and_visualize(
                            image_np, score_threshold
                        )
                        mask_time = time.time() - start_time
                        
                        progress_bar.progress(100)
                        
                        st.session_state.results = {
                            'faster_rcnn': {
                                'results': results_faster,
                                'visualization': vis_faster,
                                'inference_time': faster_time,
                                'detections': sum(len(bbox_class_results) for bbox_class_results in results_faster[0]) if isinstance(results_faster, tuple) and len(results_faster) > 0 else sum(len(r) for r in results_faster) # Handle potential tuple output from mmdet for bbox
                            },
                            'mask_rcnn': {
                                'results': results_mask,
                                'visualization': vis_mask,
                                'inference_time': mask_time,
                                # MMDetection often returns (bbox_results, segm_results) for Mask R-CNN
                                'detections': sum(len(bbox_class_results) for bbox_class_results in results_mask[0]) if isinstance(results_mask, tuple) and len(results_mask) > 0 else sum(len(r) for r in results_mask)
                            }
                        }
                        
                        st.success("✅ Comparison completed!")
                        
                except Exception as e:
                    st.error(f"Error during model inference: {e}")
                    st.exception(e) # Show full traceback for debugging
                    st.info("Please try with a different image or check the model setup and logs.")

# Results display and Comparison Table (keep as is, but be mindful of the structure of 'results_faster' and 'results_mask')
# ... (rest of your app.py) ...
# Important: Adjust how 'detections' and 'Detection Breakdown' are calculated based on the actual structure returned by inference_detector for your models.
# For MMDetection:
# - Object detection models usually return a list of arrays (one per class), where each array contains [x1, y1, x2, y2, score].
# - Instance segmentation models often return a tuple: (bbox_results, segm_results). bbox_results is similar to above. segm_results contains masks.
# You might need to adjust the `detections` count and breakdown accordingly.

# Example adjustment for detection count and breakdown (inside `with col2` and tabs):

# Adjust inside 'with col2':
# ...
        faster_data = st.session_state.results['faster_rcnn']['results']
        mask_data = st.session_state.results['mask_rcnn']['results']

        # For Faster R-CNN (typically direct list of bbox results)
        faster_detections = sum(len(r) for r in faster_data)

        # For Mask R-CNN (often tuple: (bbox_results, segm_results))
        if isinstance(mask_data, tuple) and len(mask_data) == 2:
            mask_detections = sum(len(r) for r in mask_data[0]) # Detections from bounding boxes
        else: # Fallback if it's just bbox results
            mask_detections = sum(len(r) for r in mask_data)

        # Update stored detections for captions etc.
        st.session_state.results['faster_rcnn']['detections'] = faster_detections
        st.session_state.results['mask_rcnn']['detections'] = mask_detections
# ...

# Adjust in 'tab2' (Faster R-CNN Details):
# ...
            st.markdown("#### 🏷️ Detection Breakdown")
            faster_results_data = st.session_state.results['faster_rcnn']['results']
            for i, class_detections in enumerate(faster_results_data):
                if len(class_detections) > 0:
                    st.markdown(f"**Class {i}:** {len(class_detections)} detections")
# ...

# Adjust in 'tab3' (Mask R-CNN Details):
# ...
            st.markdown("#### 🏷️ Detection Breakdown")
            mask_results_data = st.session_state.results['mask_rcnn']['results']
            # If Mask R-CNN results are a tuple (bbox_results, segm_results)
            if isinstance(mask_results_data, tuple) and len(mask_results_data) > 0:
                bbox_results = mask_results_data[0]
                for i, class_detections in enumerate(bbox_results):
                    if len(class_detections) > 0:
                        st.markdown(f"**Class {i}:** {len(class_detections)} detections (bounding boxes)")
            else: # If it's just a list of detection arrays
                for i, class_detections in enumerate(mask_results_data):
                    if len(class_detections) > 0:
                        st.markdown(f"**Class {i}:** {len(class_detections)} detections")

# ... (rest of your app.py)
# Ensure you have the footer and tips section as before.
# ...

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🚀 Built with Streamlit | 🔬 Powered by CBNetV2 | 🎯 Object Detection Comparison Tool</p>
    <p><small>Upload an image above to start comparing the models!</small></p>
</div>
""", unsafe_allow_html=True)

# Sample images section
if not st.session_state.results:
    st.markdown("---")
    st.subheader("💡 Tips for Better Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📷 Image Quality**
        - Use high-resolution images
        - Ensure good lighting
        - Avoid blurry images
        """)
    
    with col2:
        st.markdown("""
        **🎯 Detection Tips**
        - Lower threshold for more detections
        - Higher threshold for confident detections
        - Test with different object types
        """)
    
    with col3:
        st.markdown("""
        **⚡ Performance**
        - Faster R-CNN: Speed optimized
        - Mask R-CNN: Accuracy optimized
        - Consider use case requirements
        """)
