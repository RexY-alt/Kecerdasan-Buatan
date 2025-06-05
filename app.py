#!/usr/bin/env python

import streamlit as st
import sys
import os
import subprocess
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Faster R-CNN vs Mask R-CNN Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def install_dependencies():
    """Install required dependencies for Streamlit Cloud"""
    with st.spinner('Setting up environment... This may take a few minutes.'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Install PyTorch CPU version
            status_text.text("Installing PyTorch...")
            progress_bar.progress(20)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "--index-url", 
                "https://download.pytorch.org/whl/cpu"
            ])
            
            # Install OpenMIM
            status_text.text("Installing OpenMIM...")
            progress_bar.progress(40)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openmim"])
            
            # Install MMCV
            status_text.text("Installing MMCV...")
            progress_bar.progress(60)
            subprocess.check_call([
                sys.executable, "-m", "mim", "install", "mmcv-full==1.5.0"
            ])
            
            # Install MMDetection
            status_text.text("Installing MMDetection...")
            progress_bar.progress(80)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "mmdet"
            ])
            
            progress_bar.progress(100)
            status_text.text("Setup completed!")
            
            return True
            
        except Exception as e:
            st.error(f"Failed to install dependencies: {e}")
            return False

def setup_cbnetv2():
    """Setup CBNetV2 repository"""
    cbnet_dir = Path("CBNetV2")
    if not cbnet_dir.exists():
        with st.spinner('Downloading CBNetV2...'):
            try:
                subprocess.check_call([
                    "git", "clone", "https://github.com/VDIGPKU/CBNetV2.git"
                ])
                st.success("CBNetV2 downloaded successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to download CBNetV2: {e}")
                return False
    return True

def check_dependencies():
    """Check if all dependencies are available"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import mmdet
    except ImportError:
        missing_deps.append("mmdet")
    
    try:
        import mmcv
    except ImportError:
        missing_deps.append("mmcv")
    
    return missing_deps

# Check if we're in Streamlit Cloud
if 'STREAMLIT_SHARING_MODE' in os.environ or 'STREAMLIT_CLOUD' in os.environ:
    st.title("🔍 Model Comparison Setup")
    st.info("First-time setup detected. Installing dependencies...")
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        st.warning(f"Missing dependencies: {missing_deps}")
        if st.button("Install Dependencies"):
            if install_dependencies():
                st.success("Dependencies installed! Please refresh the page.")
                st.balloons()
            else:
                st.error("Failed to install dependencies. Please check the logs.")
        st.stop()
    
    # Setup CBNetV2
    if not setup_cbnetv2():
        st.stop()

# Try to import required modules
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    import io
    import base64
    
    # Add CBNetV2 to path if it exists
    if Path("CBNetV2").exists():
        sys.path.insert(0, "CBNetV2")
    
    # Try to import model
    try:
        from model import Model
        model_available = True
    except ImportError as e:
        st.error(f"Failed to import model: {e}")
        model_available = False
        
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.info("Please install required dependencies or use the setup button above.")
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
            st.info("This might be due to missing model files or network issues.")
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
    st.error("Models are not available. Please check the setup.")
    st.info("""
    **Possible solutions:**
    1. Refresh the page and try again
    2. Check if all dependencies are installed
    3. Ensure CBNetV2 repository is available
    4. Check your internet connection for model downloads
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Score threshold
    score_threshold = st.slider(
        "Detection Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    st.markdown("---")
    
    # Model information
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
        - **Backbone:** Swin Transformer Tiny
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
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Run comparison button
        if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
            try:
                with st.spinner('Running detection on both models...'):
                    # Progress bar
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
                    
                    # Store results
                    st.session_state.results = {
                        'faster_rcnn': {
                            'results': results_faster,
                            'visualization': vis_faster,
                            'inference_time': faster_time,
                            'detections': sum(len(r) for r in results_faster)
                        },
                        'mask_rcnn': {
                            'results': results_mask,
                            'visualization': vis_mask,
                            'inference_time': mask_time,
                            'detections': sum(len(r) for r in results_mask)
                        }
                    }
                    
                    st.success("✅ Comparison completed!")
                    
            except Exception as e:
                st.error(f"Error during model inference: {e}")
                st.info("Please try with a different image or check the model setup.")

with col2:
    if st.session_state.results:
        st.subheader("📊 Comparison Results")
        
        # Performance metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        faster_time = st.session_state.results['faster_rcnn']['inference_time']
        mask_time = st.session_state.results['mask_rcnn']['inference_time']
        faster_detections = st.session_state.results['faster_rcnn']['detections']
        mask_detections = st.session_state.results['mask_rcnn']['detections']
        
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
            
            # Detection breakdown by class
            st.markdown("#### 🏷️ Detection Breakdown")
            for i, class_detections in enumerate(st.session_state.results['faster_rcnn']['results']):
                if len(class_detections) > 0:
                    st.markdown(f"**Class {i}:** {len(class_detections)} detections")
    
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
            
            # Detection breakdown by class
            st.markdown("#### 🏷️ Detection Breakdown")
            for i, class_detections in enumerate(st.session_state.results['mask_rcnn']['results']):
                if len(class_detections) > 0:
                    st.markdown(f"**Class {i}:** {len(class_detections)} detections")

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
