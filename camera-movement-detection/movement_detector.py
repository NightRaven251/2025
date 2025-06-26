import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MovementResult:
    """Data class to store movement detection results"""
    frame_indices: List[int]
    movement_types: List[str]  # 'camera' or 'object'
    confidence_scores: List[float]
    transformation_matrices: List[np.ndarray]

def detect_camera_vs_object_movement(frames: List[np.ndarray], 
                                   camera_threshold: float = 30.0,
                                   object_threshold: float = 15.0,
                                   min_matches: int = 10) -> MovementResult:
    """
    Advanced movement detection that distinguishes between camera and object movement.
    """
    
    # Initialize result containers
    movement_indices = []
    movement_types = []
    confidence_scores = []
    transformation_matrices = []
    
    # Initialize feature detector and matcher
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    
    for idx, frame in enumerate(frames):
        if prev_frame is None:
            # First frame - just store for next iteration
            prev_frame = frame.copy()
            prev_keypoints, prev_descriptors = orb.detectAndCompute(frame, None)
            continue
        
        # Convert frames to grayscale for feature detection
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        current_keypoints, current_descriptors = orb.detectAndCompute(frame, None)
        
        # Skip if not enough features
        if (prev_descriptors is None or current_descriptors is None or 
            len(prev_descriptors) < min_matches or len(current_descriptors) < min_matches):
            prev_frame = frame.copy()
            prev_keypoints, prev_descriptors = current_keypoints, current_descriptors
            continue
        
        # Match features between frames
        matches = bf.match(prev_descriptors, current_descriptors)
        
        # Sort matches by distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use only good matches
        good_matches = matches[:min(len(matches), 50)]
        
        if len(good_matches) < min_matches:
            prev_frame = frame.copy()
            prev_keypoints, prev_descriptors = current_keypoints, current_descriptors
            continue
        
        # Extract matched keypoints
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Estimate transformation matrix
        transformation_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if transformation_matrix is not None:
            # Analyze the transformation
            movement_type, confidence = analyze_movement_type(
                transformation_matrix, src_pts, dst_pts, 
                camera_threshold, object_threshold
            )
            
            # Calculate overall movement magnitude
            translation_magnitude = np.sqrt(
                transformation_matrix[0, 2]**2 + transformation_matrix[1, 2]**2
            )
            
            # Determine if movement is significant
            if movement_type == 'camera' and translation_magnitude > camera_threshold:
                movement_indices.append(idx)
                movement_types.append('camera')
                confidence_scores.append(confidence)
                transformation_matrices.append(transformation_matrix)
                
            elif movement_type == 'object' and translation_magnitude > object_threshold:
                movement_indices.append(idx)
                movement_types.append('object')
                confidence_scores.append(confidence)
                transformation_matrices.append(transformation_matrix)
        
        # Update previous frame data
        prev_frame = frame.copy()
        prev_keypoints, prev_descriptors = current_keypoints, current_descriptors
    
    return MovementResult(
        frame_indices=movement_indices,
        movement_types=movement_types,
        confidence_scores=confidence_scores,
        transformation_matrices=transformation_matrices
    )

def analyze_movement_type(transformation_matrix: np.ndarray, 
                         src_pts: np.ndarray, 
                         dst_pts: np.ndarray,
                         camera_threshold: float,
                         object_threshold: float) -> Tuple[str, float]:
    """
    Analyze whether the detected movement is camera movement or object movement.
    """
    
    # Extract transformation parameters
    translation_x = transformation_matrix[0, 2]
    translation_y = transformation_matrix[1, 2]
    rotation = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
    scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[0, 1]**2)
    scale_y = np.sqrt(transformation_matrix[1, 0]**2 + transformation_matrix[1, 1]**2)
    
    # Calculate movement magnitude
    translation_magnitude = np.sqrt(translation_x**2 + translation_y**2)
    
    # Calculate keypoint movement consistency
    keypoint_movements = dst_pts - src_pts
    movement_consistency = np.std(keypoint_movements, axis=0)
    
    # Heuristics for distinguishing camera vs object movement
    
    # 1. Global vs Local movement
    global_movement_score = 1.0 - (np.mean(movement_consistency) / translation_magnitude) if translation_magnitude > 0 else 0
    
    # 2. Rotation and scale changes (more likely camera movement)
    rotation_score = min(abs(rotation) * 10, 1.0)
    scale_score = min(abs(scale_x - 1.0) + abs(scale_y - 1.0), 1.0)
    
    # 3. Movement magnitude (camera movements are usually larger)
    magnitude_score = min(translation_magnitude / camera_threshold, 1.0)
    
    # Combine scores to determine movement type
    camera_score = (global_movement_score * 0.4 + 
                   rotation_score * 0.3 + 
                   scale_score * 0.2 + 
                   magnitude_score * 0.1)
    
    object_score = 1.0 - camera_score
    
    # Determine movement type based on scores
    if camera_score > object_score and translation_magnitude > camera_threshold:
        movement_type = 'camera'
        confidence = camera_score
    elif translation_magnitude > object_threshold:
        movement_type = 'object'
        confidence = object_score
    else:
        movement_type = 'none'
        confidence = 0.0
    
    return movement_type, confidence

def visualize_movement_results(frames: List[np.ndarray], 
                             result: MovementResult,
                             output_path: str = None) -> List[np.ndarray]:
    """
    Visualize the movement detection results by drawing on frames.
    """
    
    annotated_frames = []
    
    for idx, frame in enumerate(frames):
        annotated_frame = frame.copy()
        
        # Check if this frame has detected movement
        if idx in result.frame_indices:
            movement_idx = result.frame_indices.index(idx)
            movement_type = result.movement_types[movement_idx]
            confidence = result.confidence_scores[movement_idx]
            
            # Draw bounding box and label
            height, width = frame.shape[:2]
            
            if movement_type == 'camera':
                color = (0, 0, 255)  # Red for camera movement
                label = f"Camera Movement ({confidence:.2f})"
            else:
                color = (0, 255, 0)  # Green for object movement
                label = f"Object Movement ({confidence:.2f})"
            
            # Draw rectangle around frame
            cv2.rectangle(annotated_frame, (10, 10), (width-10, height-10), color, 3)
            
            # Add text label
            cv2.putText(annotated_frame, label, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        annotated_frames.append(annotated_frame)
    
    return annotated_frames

# Backward compatibility function
def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    """
    Simple movement detection for backward compatibility.
    """
    result = detect_camera_vs_object_movement(frames, camera_threshold=threshold)
    return result.frame_indices

def load_frames_from_video(video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
    """
    Load frames from a video file (e.g., mp4, avi, mov).
    Args:
        video_path: Path to the video file
        sample_rate: Only keep every Nth frame (default 1 = keep all)
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames