import cv2
import dlib
import numpy as np
from collections import deque

# Import centralized configuration
try:
    from config import get_config
    EYE_DETECTION_CONFIG = get_config().get("eye_detection")
except ImportError:
    # Fallback configuration if config module not available
    EYE_DETECTION_CONFIG = {
        'horizontal_threshold': 0.12,
        'vertical_threshold': 0.10, 
        'high_confidence_threshold': 0.8,
        'min_contrast': 30,
        'min_eye_region_size': 144,  # 12x12 pixels
        'history_size': 7,
        'confidence_history_size': 5,
        'majority_vote_threshold': 2,
        'pupil_area_min': 20,
        'pupil_circularity_min': 0.3
    }

# Load dlib's face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Improved smoothing with configurable buffer
gaze_history = deque(maxlen=EYE_DETECTION_CONFIG['history_size'])
confidence_history = deque(maxlen=EYE_DETECTION_CONFIG['confidence_history_size'])

def detect_pupil_improved(eye_region):
    """Improved pupil detection with better filtering"""
    if eye_region.size == 0 or eye_region.shape[0] < 12 or eye_region.shape[1] < 12:
        return None
    
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for better pupil detection
    # Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    enhanced = clahe.apply(gray_eye)
    
    # Multiple blur techniques for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    median_blurred = cv2.medianBlur(blurred, 3)
    
    # Find darkest regions (potential pupils)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(median_blurred)
    
    x, y = min_loc
    h, w = eye_region.shape[:2]
    
    # More restrictive pupil validation
    # Check if the darkest point is in reasonable pupil location
    if not (w * 0.15 < x < w * 0.85 and h * 0.25 < y < h * 0.75):
        # If not valid, try to find pupil using contours
        return find_pupil_contour(median_blurred, w, h)
    
    # Additional validation: check intensity difference
    if (max_val - min_val) < EYE_DETECTION_CONFIG['min_contrast']:  # Not enough contrast
        return find_pupil_contour(median_blurred, w, h)
    
    return (x, y)

def find_pupil_contour(gray_eye, width, height):
    """Alternative pupil detection using contour analysis"""
    try:
        # Apply threshold to find dark regions
        _, thresh = cv2.threshold(gray_eye, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)  # Invert to make pupil white
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the most circular contour in reasonable size range
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Filter by size (pupil should be reasonable size)
                if area < EYE_DETECTION_CONFIG['pupil_area_min'] or area > (width * height * 0.3):
                    continue
                
                if perimeter == 0:
                    continue
                    
                # Circularity measure (4π*area/perimeter²)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > best_score and circularity > EYE_DETECTION_CONFIG['pupil_circularity_min']:  # Reasonable circularity
                    best_score = circularity
                    best_contour = contour
            
            if best_contour is not None:
                # Get centroid of best contour
                M = cv2.moments(best_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
    
    except Exception:
        pass
    
    # Fallback to center
    return (width // 2, height // 2)


def get_eye_center_ratio(eye_landmarks):
    """Calculate relative position within eye boundaries"""
    left_corner = eye_landmarks[0]
    right_corner = eye_landmarks[3]
    top_point = eye_landmarks[1]
    bottom_point = eye_landmarks[5]
    
    eye_width = right_corner[0] - left_corner[0]
    eye_height = bottom_point[1] - top_point[1]
    eye_center_x = left_corner[0] + eye_width / 2
    eye_center_y = top_point[1] + eye_height / 2
    
    return eye_center_x, eye_center_y, eye_width, eye_height

def process_eye_movement_improved(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_direction = "Looking Center"
    confidence = 0.0

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract eye landmarks
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        
        # Get eye centers and dimensions from landmarks
        left_center_x, left_center_y, left_width, left_height = get_eye_center_ratio(left_eye_points)
        right_center_x, right_center_y, right_width, right_height = get_eye_center_ratio(right_eye_points)
        
        # Create eye regions with minimal padding
        padding = 3
        
        # Left eye region
        left_x1 = max(0, int(left_center_x - left_width/2 - padding))
        left_y1 = max(0, int(left_center_y - left_height/2 - padding))
        left_x2 = min(frame.shape[1], int(left_center_x + left_width/2 + padding))
        left_y2 = min(frame.shape[0], int(left_center_y + left_height/2 + padding))
        
        # Right eye region
        right_x1 = max(0, int(right_center_x - right_width/2 - padding))
        right_y1 = max(0, int(right_center_y - right_height/2 - padding))
        right_x2 = min(frame.shape[1], int(right_center_x + right_width/2 + padding))
        right_y2 = min(frame.shape[0], int(right_center_y + right_height/2 + padding))
        
        # Extract eye regions
        left_eye = frame[left_y1:left_y2, left_x1:left_x2]
        right_eye = frame[right_y1:right_y2, right_x1:right_x2]
        
        # Skip if regions too small (configurable minimum size)
        min_size = EYE_DETECTION_CONFIG['min_eye_region_size']
        if left_eye.size < min_size or right_eye.size < min_size:
            continue
        
        # Detect pupils using improved algorithm
        left_pupil = detect_pupil_improved(left_eye)
        right_pupil = detect_pupil_improved(right_eye)
        
        # Draw eye regions (minimal visual feedback)
        cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 1)
        cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), 1)
        
        if left_pupil and right_pupil:
            # Draw pupils
            cv2.circle(frame, (left_x1 + left_pupil[0], left_y1 + left_pupil[1]), 2, (0, 0, 255), -1)
            cv2.circle(frame, (right_x1 + right_pupil[0], right_y1 + right_pupil[1]), 2, (0, 0, 255), -1)
            
            # Calculate pupil positions relative to eye regions
            left_region_width = left_x2 - left_x1
            left_region_height = left_y2 - left_y1
            right_region_width = right_x2 - right_x1
            right_region_height = right_y2 - right_y1
            
            # Normalize pupil positions
            norm_left_x = left_pupil[0] / left_region_width if left_region_width > 0 else 0.5
            norm_left_y = left_pupil[1] / left_region_height if left_region_height > 0 else 0.5
            norm_right_x = right_pupil[0] / right_region_width if right_region_width > 0 else 0.5
            norm_right_y = right_pupil[1] / right_region_height if right_region_height > 0 else 0.5
            
            # Average both eyes
            avg_x = (norm_left_x + norm_right_x) / 2
            avg_y = (norm_left_y + norm_right_y) / 2
            
            # Configurable thresholds for better accuracy
            horizontal_threshold = EYE_DETECTION_CONFIG['horizontal_threshold']
            vertical_threshold = EYE_DETECTION_CONFIG['vertical_threshold']
            
            # Calculate distances from center
            center_x, center_y = 0.5, 0.5
            dist_x = abs(avg_x - center_x)
            dist_y = abs(avg_y - center_y)
            
            # Determine direction
            raw_direction = "Looking Center"
            raw_confidence = 0.0
            
            if dist_x > horizontal_threshold:
                # Fixed mirror effect: when user looks right, pupil moves left in camera frame
                if avg_x < center_x:
                    raw_direction = "Looking Right"  # User looking right = pupil left in frame
                else:
                    raw_direction = "Looking Left"   # User looking left = pupil right in frame
                raw_confidence = min(1.0, dist_x / 0.15)
            elif dist_y > vertical_threshold:
                if avg_y < center_y:
                    raw_direction = "Looking Up"
                else:
                    raw_direction = "Looking Down"
                raw_confidence = min(1.0, dist_y / 0.15)
            else:
                raw_direction = "Looking Center"
                max_dist = max(dist_x, dist_y)
                raw_confidence = max(0.3, 1.0 - max_dist / 0.1)
            
            # Enhanced temporal smoothing
            gaze_history.append((raw_direction, raw_confidence))
            confidence_history.append(raw_confidence)
            
            if len(gaze_history) >= 3:
                # Get recent samples
                recent_data = list(gaze_history)[-5:]  # Use more samples
                recent_directions = [item[0] for item in recent_data]
                recent_confidences = [item[1] for item in recent_data]
                
                # Count occurrences of each direction
                direction_counts = {}
                for direction in recent_directions:
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                
                # Find most common direction
                most_common_direction = max(direction_counts, key=direction_counts.get)
                most_common_count = direction_counts[most_common_direction]
                
                # Use majority voting with confidence weighting
                if most_common_count >= EYE_DETECTION_CONFIG['majority_vote_threshold'] or raw_confidence > EYE_DETECTION_CONFIG['high_confidence_threshold']:
                    gaze_direction = raw_direction if raw_confidence > 0.8 else most_common_direction
                    # Weighted confidence based on consistency
                    avg_confidence = np.mean(list(confidence_history)[-3:])
                    confidence = (raw_confidence + avg_confidence) / 2
                else:
                    # Use previous stable direction
                    gaze_direction = recent_directions[-2] if len(recent_directions) > 1 else raw_direction
                    confidence = np.mean(recent_confidences[-2:]) * 0.9
                    
            else:
                gaze_direction = raw_direction
                confidence = raw_confidence
    
    return frame, gaze_direction, confidence

# Test function (if running standalone)
def test_eye_gaze():
    cap = cv2.VideoCapture(1)
    
    print("=== EYE GAZE TEST ===")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result_frame, gaze_dir, conf = process_eye_movement_improved(frame)
        
        # Clean display
        cv2.putText(result_frame, f"Gaze: {gaze_dir}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Confidence: {conf:.2f}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Eye Gaze Detection', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_eye_gaze()