import cv2
import dlib
import numpy as np
import math
from collections import deque
import time

# Import centralized configuration
try:
    from config import get_config
    HEAD_POSE_CONFIG = get_config().get("head_pose")
except ImportError:
    # Fallback configuration if config module not available
    HEAD_POSE_CONFIG = {
        'yaw_threshold': 12,
        'pitch_threshold_up': 10,
        'pitch_threshold_down': 12,
        'roll_threshold': 8,
        'center_yaw_tolerance': 7,
        'center_pitch_tolerance': 7,
        'center_roll_tolerance': 6,
        'angle_history_size': 15,
        'calibration_frames': 20,
        'state_change_delay': 0.2,
        'outlier_threshold': 30,
        'focal_length_multiplier': 1.2,
        'min_contrast_threshold': 30
    }

# Load face detector & landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Improved 3D Model Points (more accurate facial geometry)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip - reference point
    (0.0, -40.0, -20.0),    # Chin - adjusted depth
    (-25.0, 25.0, -15.0),   # Left eye - adjusted position
    (25.0, 25.0, -15.0),    # Right eye - adjusted position  
    (-20.0, -15.0, -10.0),  # Left mouth corner
    (20.0, -15.0, -10.0)    # Right mouth corner
], dtype=np.float64)

# Dynamic camera calibration based on face size
def get_camera_matrix(face_width):
    # Estimate focal length based on face width for better accuracy
    focal_length = face_width * HEAD_POSE_CONFIG['focal_length_multiplier']
    center_x = 320  # Assuming 640 width
    center_y = 240  # Assuming 480 height
    
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return camera_matrix

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Configurable smoothing parameters
yaw_history = deque(maxlen=HEAD_POSE_CONFIG['angle_history_size'])
pitch_history = deque(maxlen=HEAD_POSE_CONFIG['angle_history_size'])
roll_history = deque(maxlen=HEAD_POSE_CONFIG['angle_history_size'])
calibration_pitch_values = []
calibration_yaw_values = []
calibration_roll_values = []

# Global variables for state management
previous_state = "Looking at Screen"
calibrated_angles = None
state_change_time = time.time()
# Use configurable delay
STATE_CHANGE_DELAY = HEAD_POSE_CONFIG['state_change_delay']

def reset_calibration():
    """Reset all calibration data"""
    global calibrated_angles, calibration_pitch_values, calibration_yaw_values, calibration_roll_values
    global yaw_history, pitch_history, roll_history, previous_state
    
    calibrated_angles = None
    calibration_pitch_values.clear()
    calibration_yaw_values.clear()
    calibration_roll_values.clear()
    yaw_history.clear()
    pitch_history.clear()
    roll_history.clear()
    previous_state = "Looking at Screen"

def update_head_pose_config(new_config):
    """Update head pose configuration parameters"""
    global HEAD_POSE_CONFIG
    HEAD_POSE_CONFIG.update(new_config)
    print(f"Head pose config updated: {new_config}")

def get_head_pose_config():
    """Get current head pose configuration"""
    return HEAD_POSE_CONFIG.copy()

def auto_tune_thresholds(calibrated_angles, tolerance_multiplier=1.5):
    """Auto-tune thresholds based on calibration data"""
    global HEAD_POSE_CONFIG
    
    if not calibrated_angles:
        return HEAD_POSE_CONFIG
    
    pitch_std = np.std(calibration_pitch_values) if calibration_pitch_values else 5
    yaw_std = np.std(calibration_yaw_values) if calibration_yaw_values else 5
    roll_std = np.std(calibration_roll_values) if calibration_roll_values else 5
    
    # Auto-tune based on calibration stability
    new_config = {
        'yaw_threshold': max(8, int(yaw_std * tolerance_multiplier * 2)),
        'pitch_threshold_up': max(8, int(pitch_std * tolerance_multiplier * 2)),
        'pitch_threshold_down': max(8, int(pitch_std * tolerance_multiplier * 2)),
        'roll_threshold': max(6, int(roll_std * tolerance_multiplier * 2)),
        
        'center_yaw_tolerance': max(5, int(yaw_std * tolerance_multiplier)),
        'center_pitch_tolerance': max(5, int(pitch_std * tolerance_multiplier)),
        'center_roll_tolerance': max(4, int(roll_std * tolerance_multiplier))
    }
    
    update_head_pose_config(new_config)
    
    print(f"Auto-tuned thresholds based on calibration stability:")
    print(f"  Std deviations - Pitch: {pitch_std:.1f}, Yaw: {yaw_std:.1f}, Roll: {roll_std:.1f}")
    print(f"  New thresholds - Yaw: {new_config['yaw_threshold']}, Pitch: {new_config['pitch_threshold_up']}, Roll: {new_config['roll_threshold']}")
    
    return new_config

def get_head_pose_angles(image_points, face_width):
    camera_matrix = get_camera_matrix(face_width)
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles with improved calculation
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        # Standard extraction
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Gimbal lock case
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    # Convert to degrees and apply correction
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw) 
    roll_deg = np.degrees(roll)
    
    return pitch_deg, yaw_deg, roll_deg

def smooth_angle(angle_history, new_angle):
    """Improved smoothing with less aggressive filtering"""
    if len(angle_history) > 3:
        # Calculate median for outlier detection with smaller threshold
        median_val = np.median(angle_history)
        # Reject only extreme outliers
        if abs(new_angle - median_val) > HEAD_POSE_CONFIG['outlier_threshold']:  # Configurable outlier threshold
            new_angle = median_val
    
    angle_history.append(new_angle)
    
    # Less aggressive smoothing for better responsiveness
    if len(angle_history) >= 3:
        # Use simple moving average instead of weighted average
        return np.mean(list(angle_history)[-5:])  # Only use last 5 values
    else:
        return np.mean(angle_history)

def calibrate_neutral_position(angles):
    """Improved calibration using multiple frames"""
    global calibration_pitch_values, calibration_yaw_values, calibration_roll_values
    global calibrated_angles
    
    pitch, yaw, roll = angles
    calibration_pitch_values.append(pitch)
    calibration_yaw_values.append(yaw)
    calibration_roll_values.append(roll)
    
    if len(calibration_pitch_values) >= HEAD_POSE_CONFIG['calibration_frames']:
        # Remove outliers before calculating median
        pitch_values = np.array(calibration_pitch_values)
        yaw_values = np.array(calibration_yaw_values)
        roll_values = np.array(calibration_roll_values)
        
        # Remove extreme outliers (beyond 2 standard deviations)
        pitch_clean = pitch_values[np.abs(pitch_values - np.mean(pitch_values)) < 2 * np.std(pitch_values)]
        yaw_clean = yaw_values[np.abs(yaw_values - np.mean(yaw_values)) < 2 * np.std(yaw_values)]
        roll_clean = roll_values[np.abs(roll_values - np.mean(roll_values)) < 2 * np.std(roll_values)]
        
        # Use median of cleaned data
        calibrated_pitch = np.median(pitch_clean) if len(pitch_clean) > 0 else np.median(pitch_values)
        calibrated_yaw = np.median(yaw_clean) if len(yaw_clean) > 0 else np.median(yaw_values)
        calibrated_roll = np.median(roll_clean) if len(roll_clean) > 0 else np.median(roll_values)
        
        calibrated_angles = (calibrated_pitch, calibrated_yaw, calibrated_roll)
        
        print(f"Calibration complete: Pitch={calibrated_pitch:.1f}, Yaw={calibrated_yaw:.1f}, Roll={calibrated_roll:.1f}")
        print(f"Std deviations: P={np.std(pitch_values):.1f}, Y={np.std(yaw_values):.1f}, R={np.std(roll_values):.1f}")
        
        # Auto-tune thresholds based on calibration quality
        auto_tune_thresholds(calibrated_angles)
        
        return calibrated_angles
    
    return None

def process_head_pose(frame, calibrated_angles_input=None):
    """Main function for head pose detection - compatible with existing test system"""
    global previous_state, state_change_time, calibrated_angles

    # Use global calibrated_angles if not provided
    if calibrated_angles_input is not None:
        calibrated_angles = calibrated_angles_input

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    head_direction = "Looking at Screen"

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Calculate face width for dynamic camera matrix
        face_width = face.right() - face.left()
        
        # Extract key facial landmarks
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)

        angles = get_head_pose_angles(image_points, face_width)
        if angles is None:
            continue

        # Apply smoothing
        pitch = smooth_angle(pitch_history, angles[0])
        yaw = smooth_angle(yaw_history, angles[1])
        roll = smooth_angle(roll_history, angles[2])
        
        # If calibrating, collect calibration data
        if calibrated_angles is None:
            result = calibrate_neutral_position((pitch, yaw, roll))
            if result is not None:
                return frame, result  # Return calibrated angles
            else:
                remaining = HEAD_POSE_CONFIG['calibration_frames'] - len(calibration_pitch_values)
                cv2.putText(frame, f"Calibrating... {remaining} frames left", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Look straight at the screen!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                return frame, None

        # Use calibrated angles for head pose detection
        pitch_offset, yaw_offset, roll_offset = calibrated_angles
        
        # Calculate relative angles
        rel_pitch = pitch - pitch_offset
        rel_yaw = yaw - yaw_offset
        rel_roll = roll - roll_offset
        
        # Configurable thresholds for better detection
        YAW_THRESHOLD = HEAD_POSE_CONFIG['yaw_threshold']
        PITCH_THRESHOLD_UP = HEAD_POSE_CONFIG['pitch_threshold_up']
        PITCH_THRESHOLD_DOWN = HEAD_POSE_CONFIG['pitch_threshold_down']
        ROLL_THRESHOLD = HEAD_POSE_CONFIG['roll_threshold']
        
        # Configurable center tolerance
        CENTER_YAW_TOLERANCE = HEAD_POSE_CONFIG['center_yaw_tolerance']
        CENTER_PITCH_TOLERANCE = HEAD_POSE_CONFIG['center_pitch_tolerance']
        CENTER_ROLL_TOLERANCE = HEAD_POSE_CONFIG['center_roll_tolerance']
        
        current_time = time.time()
        new_direction = previous_state  # Default to previous state
        
        # Determine head direction with improved logic
        if (abs(rel_yaw) <= CENTER_YAW_TOLERANCE and 
            abs(rel_pitch) <= CENTER_PITCH_TOLERANCE and 
            abs(rel_roll) <= CENTER_ROLL_TOLERANCE):
            new_direction = "Looking at Screen"
            
        elif rel_yaw < -YAW_THRESHOLD:  # Looking left
            new_direction = "Looking Left"
            
        elif rel_yaw > YAW_THRESHOLD:   # Looking right
            new_direction = "Looking Right"
            
        elif rel_pitch > PITCH_THRESHOLD_UP:    # Looking up
            new_direction = "Looking Up"
            
        elif rel_pitch < -PITCH_THRESHOLD_DOWN: # Looking down
            new_direction = "Looking Down"
            
        elif abs(rel_roll) > ROLL_THRESHOLD:    # Head tilted
            new_direction = "Tilted"
        
        # Only apply delay when returning to "Looking at Screen"
        # Immediate response for movements away from screen
        if new_direction != previous_state:
            if new_direction == "Looking at Screen":
                # Apply delay when returning to center
                if current_time - state_change_time >= STATE_CHANGE_DELAY:
                    previous_state = new_direction
                    head_direction = new_direction
                    state_change_time = current_time
                else:
                    head_direction = previous_state
            else:
                # Immediate response for movements away from center
                previous_state = new_direction
                head_direction = new_direction
                state_change_time = current_time
        else:
            head_direction = new_direction
            state_change_time = current_time

        # Add comprehensive debug information
        if frame.shape[0] > 200:  # Only if frame is large enough
            # Display relative angles with color coding
            color_yaw = (0, 255, 0) if abs(rel_yaw) <= CENTER_YAW_TOLERANCE else (0, 0, 255)
            color_pitch = (0, 255, 0) if abs(rel_pitch) <= CENTER_PITCH_TOLERANCE else (0, 0, 255)
            color_roll = (0, 255, 0) if abs(rel_roll) <= CENTER_ROLL_TOLERANCE else (0, 0, 255)
            
            cv2.putText(frame, f"Yaw: {rel_yaw:.1f} (thr: {YAW_THRESHOLD})", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yaw, 1)
            cv2.putText(frame, f"Pitch: {rel_pitch:.1f} (thr: {PITCH_THRESHOLD_DOWN}/{PITCH_THRESHOLD_UP})", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pitch, 1)
            cv2.putText(frame, f"Roll: {rel_roll:.1f} (thr: {ROLL_THRESHOLD})", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_roll, 1)
            
            # Show which condition is triggering
            if new_direction != "Looking at Screen":
                if rel_yaw < -YAW_THRESHOLD:
                    cv2.putText(frame, "TRIGGER: Yaw < -threshold (LEFT)", 
                               (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                elif rel_yaw > YAW_THRESHOLD:
                    cv2.putText(frame, "TRIGGER: Yaw > +threshold (RIGHT)", 
                               (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                elif rel_pitch > PITCH_THRESHOLD_UP:
                    cv2.putText(frame, "TRIGGER: Pitch > +threshold (UP)", 
                               (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                elif rel_pitch < -PITCH_THRESHOLD_DOWN:
                    cv2.putText(frame, "TRIGGER: Pitch < -threshold (DOWN)", 
                               (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                elif abs(rel_roll) > ROLL_THRESHOLD:
                    cv2.putText(frame, "TRIGGER: Roll > threshold (TILT)", 
                               (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return frame, head_direction