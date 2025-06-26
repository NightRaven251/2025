import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from movement_detector import detect_camera_vs_object_movement, MovementResult, load_frames_from_video

# Page configuration
st.set_page_config(
    page_title="Camera Movement Detection",
    page_icon="📹",
    layout="wide"
)

def main():
    # Header
    st.title("📹 Camera Movement Detection")
    st.write("Upload images or video to detect significant camera movement.")
    
    # Sidebar for basic settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Input type selection
        input_type = st.radio(
            "Input Type:",
            ["Image Sequence", "Video File"]
        )
        
        # Basic threshold
        threshold = st.slider(
            "Movement Threshold",
            min_value=10.0,
            max_value=100.0,
            value=30.0,
            step=5.0,
            help="Higher values = less sensitive to movement"
        )
    
    # Main content
    st.header("📤 Upload Files")
    
    if input_type == "Image Sequence":
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            frames = load_images(uploaded_files)
            if frames:
                process_frames(frames, threshold)
    
    else:  # Video file
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov"]
        )
        
        if uploaded_video:
            frames = []
            with st.spinner("Extracting video frames..."):
                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    tmp_path = tmp_file.name
                try:
                    frames = load_frames_from_video(tmp_path, sample_rate=5)
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    os.unlink(tmp_path)
            if frames:
                st.success(f"✅ Extracted {len(frames)} frames from video.")
                process_frames(frames, threshold)

def load_images(uploaded_files):
    """Load uploaded images"""
    frames = []
    
    with st.spinner("Loading images..."):
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                frame = np.array(image)
                
                # Convert RGBA to RGB if necessary
                if len(frame.shape) == 3 and frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                
                # Convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    st.success(f"✅ Loaded {len(frames)} images")
    return frames

def process_frames(frames, threshold):
    """Process frames and display results"""
    
    if len(frames) < 2:
        st.warning("Need at least 2 frames for movement detection")
        return
    
    # Run movement detection
    with st.spinner("Analyzing movement..."):
        result = detect_camera_vs_object_movement(
            frames, camera_threshold=threshold, object_threshold=threshold/2
        )
    
    # Display results
    display_results(frames, result)

def display_results(frames, result):
    """Display movement detection results"""
    
    st.header("📊 Results")
    
    # Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Frames", len(frames))
    
    with col2:
        camera_movements = [i for i, t in zip(result.frame_indices, result.movement_types) if t == 'camera']
        st.metric("Camera Movements", len(camera_movements))
    
    with col3:
        object_movements = [i for i, t in zip(result.frame_indices, result.movement_types) if t == 'object']
        st.metric("Object Movements", len(object_movements))
    
    # Show detected movements
    if result.frame_indices:
        st.subheader("🎯 Detected Movements")
        
        for i, (frame_idx, movement_type, confidence) in enumerate(
            zip(result.frame_indices, result.movement_types, result.confidence_scores)
        ):
            with st.expander(f"Frame {frame_idx} - {movement_type.title()} Movement"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display the frame
                    if frame_idx < len(frames):
                        # Convert BGR to RGB for display
                        display_frame = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                        st.image(display_frame, caption=f"Frame {frame_idx}", use_column_width=True)
                
                with col2:
                    # Movement info
                    if movement_type == 'camera':
                        st.markdown("🔴 **Camera Movement**")
                    else:
                        st.markdown("🟢 **Object Movement**")
                    
                    st.write(f"Confidence: {confidence:.2f}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
    
    else:
        st.info("🎉 No significant movement detected!")
    
    # Show all frames with movement indicators
    if result.frame_indices:
        st.subheader("🖼️ All Frames with Movement")
        
        # Create a simple visualization
        for frame_idx in result.frame_indices:
            if frame_idx < len(frames):
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                
                # Add simple annotation
                movement_idx = result.frame_indices.index(frame_idx)
                movement_type = result.movement_types[movement_idx]
                
                if movement_type == 'camera':
                    st.markdown(f"🔴 **Frame {frame_idx}** - Camera Movement")
                else:
                    st.markdown(f"🟢 **Frame {frame_idx}** - Object Movement")
                
                st.image(display_frame, use_column_width=True)

if __name__ == "__main__":
    main()