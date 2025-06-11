#!/usr/bin/env python

import streamlit as st
import sys
import os
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
cbnet_path = Path("CBNetV2")
if cbnet_path.exists() and cbnet_path.is_dir():
    sys.path.insert(0, str(cbnet_path.resolve()))
else:
    st.error(f"CRITICAL ERROR: CBNetV2 directory not found at {str(cbnet_path.resolve())}. Ensure the submodule is correctly checked out.")
    st.stop()

# Try to import required modules
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    
    # Try to import model
    try:
        from model import Model
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
    model_available = False
    st.stop()


# Custom CSS
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
                st.session_state.model = Model()
                st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.exception(e) # Show full traceback for debugging
            st.info("This might be due to missing model weight files (check `weights` directory and download logic in `model.py`), issues with model configurations, or problems with the CBNetV2/MMDetection setup.")
            model_available = False

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

# Sidebar
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

# Main content
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
        image_np = np.array(image.convert("RGB"))
        
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
                            },
                            'mask_rcnn': {
                                'results': results_mask,
                                'visualization': vis_mask,
                                'inference_time': mask_time,
                            }
                        }
                        
                        st.success("✅ Comparison completed!")
                        
                except Exception as e:
                    st.error(f"Error during model inference: {e}")
                    st.exception(e)
                    st.info("Please try with a different image or check the model setup and logs.")

with col2:
    if st.session_state.results:
        st.subheader("📊 Comparison Results")
        
        # Performance metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        faster_time = st.session_state.results['faster_rcnn']['inference_time']
        mask_time = st.session_state.results['mask_rcnn']['inference_time']
        
        faster_data = st.session_state.results['faster_rcnn']['results']
        mask_data = st.session_state.results['mask_rcnn']['results']

        # For Faster R-CNN (typically direct list of bbox results)
        faster_detections = sum(len(r) for r in faster_data)

        # For Mask R-CNN (often tuple: (bbox_results, segm_results))
        if isinstance(mask_data, tuple) and len(mask_data) > 0:
            mask_detections = sum(len(r) for r in mask_data[0]) # Detections from bounding boxes
        else: # Fallback if it's just bbox results
            mask_detections = sum(len(r) for r in mask_data)

        # Update stored detections for captions etc.
        st.session_state.results['faster_rcnn']['detections'] = faster_detections
        st.session_state.results['mask_rcnn']['detections'] = mask_detections
        
        with col_metric1:
            st.metric(
                "⚡ Faster R-CNN Time",
                f"{faster_time:.3f}s",
                delta=f"{((faster_time - mask_time) / mask_time * 100):+.1f}%" if mask_time > 0 else "N/A"
            )
        
        with col_metric2:
            st.metric(
                "🎯 Mask R-CNN Time",
                f"{mask_time:.3f}s",
                delta=f"{((mask_time - faster_time) / faster_time * 100):+.1f}%" if faster_time > 0 else "N/A"
            )
        
        with col_metric3:
            speed_up = mask_time / faster_time if faster_time > 0 else 0
            st.metric(
                "📈 Speed Ratio",
                f"{speed_up:.2f}x",
                help="How many times faster is Faster R-CNN compared to Mask R-CNN"
            )

# Results display
if st.session_state.results:
    st.markdown("---")
    st.subheader("🖼️ Visual Comparison")
    
    faster_detections = st.session_state.results['faster_rcnn']['detections']
    faster_time = st.session_state.results['faster_rcnn']['inference_time']
    mask_detections = st.session_state.results['mask_rcnn']['detections']
    mask_time = st.session_state.results['mask_rcnn']['inference_time']
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Side by Side", "Faster R-CNN Details", "Mask R-CNN Details"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🏃‍♂️ Faster R-CNN (DB-ResNet50)")
            st.image(
                st.session_state.results['faster_rcnn']['visualization'],
                caption=f"Detections: {faster_detections} | Time: {faster_time:.3f}s",
                use_column_width=True
            )
        
        with col_right:
            st.markdown("### 🎭 Mask R-CNN (DB-Swin-T)")
            st.image(
                st.session_state.results['mask_rcnn']['visualization'],
                caption=f"Detections: {mask_detections} | Time: {mask_time:.3f}s",
                use_column_width=True
            )
    
    #=====================================================#
    # KODE YANG DIPERBAIKI UNTUK TAB 2
    #=====================================================#
    with tab2:
        st.markdown("### 🏃‍♂️ Faster R-CNN Detailed Results")
        col_img, col_stats = st.columns([2, 1])
        
        with col_img:
            st.image(
                st.session_state.results['faster_rcnn']['visualization'],
                use_column_width=True
            )
        
        with col_stats:
            st.markdown("#### 📈 Performance Metrics")
            st.markdown(f"**Inference Time:** {faster_time:.3f} seconds")
            st.markdown(f"**Total Detections:** {faster_detections}")
            st.markdown(f"**Score Threshold:** {score_threshold}")
            
            st.markdown("#### 🏷️ Detection Breakdown")
            faster_results_data = st.session_state.results['faster_rcnn']['results']
            
            if faster_results_data and any(len(arr) > 0 for arr in faster_results_data):
                for i, class_detections in enumerate(faster_results_data):
                    if len(class_detections) > 0:
                        st.markdown(f"**Class {i}:** {len(class_detections)} detections")
            else:
                st.markdown("No detections found with the current score threshold.")
    
    #=====================================================#
    # KODE YANG DIPERBAIKI UNTUK TAB 3
    #=====================================================#
    with tab3:
        st.markdown("### 🎭 Mask R-CNN Detailed Results")
        col_img, col_stats = st.columns([2, 1])
        
        with col_img:
            st.image(
                st.session_state.results['mask_rcnn']['visualization'],
                use_column_width=True
            )
        
        with col_stats:
            st.markdown("#### 📈 Performance Metrics")
            st.markdown(f"**Inference Time:** {mask_time:.3f} seconds")
            st.markdown(f"**Total Detections:** {mask_detections}")
            st.markdown(f"**Score Threshold:** {score_threshold}")
            
            st.markdown("#### 🏷️ Detection Breakdown")
            mask_results_data = st.session_state.results['mask_rcnn']['results']

            if isinstance(mask_results_data, tuple) and len(mask_results_data) > 0:
                bbox_results = mask_results_data[0]
                if any(len(arr) > 0 for arr in bbox_results):
                    for i, class_detections in enumerate(bbox_results):
                        if len(class_detections) > 0:
                            st.markdown(f"**Class {i}:** {len(class_detections)} detections (bounding boxes)")
                else:
                    st.markdown("No detections found with the current score threshold.")
            else:
                if mask_results_data and any(len(arr) > 0 for arr in mask_results_data):
                    for i, class_detections in enumerate(mask_results_data):
                        if len(class_detections) > 0:
                            st.markdown(f"**Class {i}:** {len(class_detections)} detections")
                else:
                    st.markdown("No detections found with the current score threshold.")

    # Comparison table
    st.markdown("---")
    st.subheader("📋 Detailed Comparison Table")
    
    comparison_data = {
        "Metric": [
            "Inference Time (seconds)",
            "Total Detections",
            "Speed (FPS)",
            "Model Type",
            "Backbone",
            "Primary Use Case"
        ],
        "Faster R-CNN (DB-ResNet50)": [
            f"{faster_time:.3f}",
            f"{faster_detections}",
            f"{1/faster_time:.1f}" if faster_time > 0 else "N/A",
            "Two-stage detector",
            "ResNet50 + CBNetV2",
            "Real-time detection"
        ],
        "Mask R-CNN (DB-Swin-T)": [
            f"{mask_time:.3f}",
            f"{mask_detections}",
            f"{1/mask_time:.1f}" if mask_time > 0 else "N/A",
            "Instance segmentation",
            "Swin Transformer Tiny",
            "Detailed segmentation"
        ]
    }
    
    st.table(comparison_data)

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
