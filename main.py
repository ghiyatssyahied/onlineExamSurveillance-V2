import cv2
import time
import os
from datetime import datetime
from eye_movement import process_eye_movement_improved
from head_pose import process_head_pose, reset_calibration
from mobile_detection import process_mobile_detection
from config import get_config

# Load configuration
config = get_config()

# Initialize webcam with configuration
camera_index = config.get("system", "camera_index")
fallback_camera = config.get("system", "fallback_camera_index")

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    cap = cv2.VideoCapture(fallback_camera)

if not cap.isOpened():
    print("Error: Cannot open camera!")
    exit()

# Create absolute path for log directory
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log")

# Create log directory with proper error handling
try:
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory created/verified at: {log_dir}")
    
    # Test write permissions
    test_file = os.path.join(log_dir, "test_write.txt")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print("Write permissions verified")
except Exception as e:
    print(f"Error creating log directory: {e}")
    print("Screenshots will not be saved!")
    log_dir = None

# Calibration variables
calibrated_angles = None
calibration_complete = False
start_time = time.time()

# Timers for each functionality
head_misalignment_start_time = None
eye_misalignment_start_time = None
mobile_detection_start_time = None

# Initialize variables with default values
head_direction = "Looking at Screen"
gaze_direction = "Looking Center"
gaze_confidence = 0.0

# Counters for saved screenshots
head_screenshots = 0
eye_screenshots = 0
mobile_screenshots = 0

def calibrate_head_pose_properly(cap):
    """Proper head pose calibration like in test_accuracy.py"""
    global calibrated_angles
    
    print("\n=== HEAD POSE CALIBRATION ===")
    print("Look straight at the camera!")
    print("Keep your head still and centered")
    print("Calibration will take about 10-15 seconds")
    
    calibration_start = time.time()
    calibrated_angles = None
    
    while calibrated_angles is None and (time.time() - calibration_start) < 25:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame, result = process_head_pose(frame, None)
        
        if result is not None:
            calibrated_angles = result
            print(f"Head pose calibration completed: {calibrated_angles}")
            return True
            
        # Show calibration progress
        elapsed = time.time() - calibration_start
        cv2.putText(frame, "HEAD POSE CALIBRATION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Look straight at the camera!", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Keep your head still and centered", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Calibrating... {elapsed:.1f}s", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'r' to restart calibration, 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Head Pose Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('r'):
            reset_calibration()
            calibration_start = time.time()
            print("Calibration restarted!")
    
    if calibrated_angles is None:
        print("Calibration timeout! Please try again.")
        return False
    
    return True

def save_screenshot(frame, event_type, detail, timestamp):
    """Save screenshot with proper error handling"""
    global head_screenshots, eye_screenshots, mobile_screenshots
    
    if log_dir is None:
        print(f"Cannot save screenshot - log directory not available")
        return False
    
    try:
        dt = datetime.fromtimestamp(timestamp)
        filename = f"{event_type}_{detail}_{dt.strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(log_dir, filename)
        
        success = cv2.imwrite(filepath, frame)
        
        if success:
            if event_type == "head":
                head_screenshots += 1
            elif event_type == "eye":
                eye_screenshots += 1
            elif event_type == "mobile":
                mobile_screenshots += 1
            
            print(f"Screenshot saved: {filename}")
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False
            
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

print("=== Cheating Detection System Started ===")
print(f"Configuration loaded: {len(config.config)} sections")
print("First, we need to calibrate head pose detection")
print("Press SPACE to start calibration, 'q' to quit")

# Wait for user to start calibration
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    cv2.putText(frame, "CHEATING DETECTION SYSTEM", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Press SPACE to start HEAD POSE calibration", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Note: Eye movement needs no calibration", (10, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Cheating Detection System", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        # Start calibration
        if calibrate_head_pose_properly(cap):
            calibration_complete = True
            print("Head pose calibration successful!")
            print("Starting main detection system...")
            break
        else:
            print("Calibration failed!")
            continue
    elif key == ord('q'):
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

if not calibration_complete:
    print("No calibration completed. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Reset start time for main detection
start_time = time.time()

print("\n=== Main Detection System Started ===")
print("System is now monitoring for suspicious behavior")
print("Controls:")
print("- 'q': Quit system")
print("- 'r': Recalibrate head pose")
print("- 'c': Show current configuration")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        current_time = time.time()
        
        # Run all detections with proper calibration
        frame, head_direction = process_head_pose(frame, calibrated_angles)
        frame, gaze_direction, gaze_confidence = process_eye_movement_improved(frame)
        frame, mobile_detected = process_mobile_detection(frame)
        
        # Process detections and save screenshots
        # Initialize suspicious indicators
        head_suspicious = False
        eye_suspicious = False
        mobile_suspicious = False
        
        # Check for head misalignment
        if head_direction != "Looking at Screen":
            if head_misalignment_start_time is None:
                head_misalignment_start_time = current_time
                print(f"Head movement detected: {head_direction}")
            else:
                # Mark as suspicious if timer >= 3 seconds and keep it true
                if current_time - head_misalignment_start_time >= 3:
                    head_suspicious = True
                    # Take screenshot only once at 3 seconds
                    if current_time - head_misalignment_start_time < 3.1:  # Small window to capture once
                        detail = head_direction.replace(' ', '_').lower()
                        if save_screenshot(frame, "head", detail, current_time):
                            print(f"Head misalignment documented!")
        else:
            head_misalignment_start_time = None
        
        # Check for eye misalignment
        if gaze_direction != "Looking Center" and gaze_confidence > 0.4:
            if eye_misalignment_start_time is None:
                eye_misalignment_start_time = current_time
                print(f"Eye movement detected: {gaze_direction}")
            else:
                # Mark as suspicious if timer >= 3 seconds and keep it true
                if current_time - eye_misalignment_start_time >= 3:
                    eye_suspicious = True
                    # Take screenshot only once at 3 seconds
                    if current_time - eye_misalignment_start_time < 3.1:  # Small window to capture once
                        detail = gaze_direction.replace(' ', '_').lower()
                        if save_screenshot(frame, "eye", detail, current_time):
                            print(f"Eye misalignment documented!")
        else:
            eye_misalignment_start_time = None
        
        # Check for mobile detection
        if mobile_detected:
            if mobile_detection_start_time is None:
                mobile_detection_start_time = current_time
                print("Mobile device detected")
                # Mobile is suspicious immediately when first detected
                mobile_suspicious = True
            else:
                # Continue being suspicious while mobile is detected
                mobile_suspicious = True
                # Take screenshot only once at 2 seconds
                if current_time - mobile_detection_start_time >= 2 and current_time - mobile_detection_start_time < 2.1:
                    if save_screenshot(frame, "mobile", "detected", current_time):
                        print(f"Mobile detection documented!")
        else:
            mobile_detection_start_time = None
        
        # Overall suspicious status
        suspicious = head_suspicious or eye_suspicious or mobile_suspicious
        
        # CLEAN DISPLAY - Only show essential information
        cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Head Direction: {head_direction}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display suspicious status with appropriate color
        suspicious_color = (0, 0, 255) if suspicious else (0, 255, 0)  # Red if suspicious, Green if not
        cv2.putText(frame, f"Suspicious: {suspicious}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, suspicious_color, 2)
        
        # Show active timers only (minimal display)
        timer_y = 160
        if head_misalignment_start_time:
            head_duration = current_time - head_misalignment_start_time
            cv2.putText(frame, f"Head tracking: {head_duration:.1f}s", (10, timer_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            timer_y += 25
        
        if eye_misalignment_start_time:
            eye_duration = current_time - eye_misalignment_start_time
            cv2.putText(frame, f"Eye tracking: {eye_duration:.1f}s", (10, timer_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            timer_y += 25
        
        if mobile_detection_start_time:
            mobile_duration = current_time - mobile_detection_start_time
            cv2.putText(frame, f"Mobile tracking: {mobile_duration:.1f}s", (10, timer_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show calibration status
        cv2.putText(frame, "HEAD CALIBRATED", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Control instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to recalibrate, 'c' for config", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Cheating Detection System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Recalibrate head pose
            print("Recalibrating head pose...")
            cv2.destroyWindow("Cheating Detection System")
            reset_calibration()
            
            if calibrate_head_pose_properly(cap):
                print("Recalibration successful!")
                # Reset timers
                head_misalignment_start_time = None
                eye_misalignment_start_time = None
                mobile_detection_start_time = None
            else:
                print("Recalibration failed! Using previous calibration.")
        elif key == ord('c'):
            # Show current configuration
            print("\nCurrent Configuration:")
            config.print_config()

except KeyboardInterrupt:
    print("\nSystem interrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    print(f"System closed.")
    print(f"Total screenshots - Head: {head_screenshots}, Eye: {eye_screenshots}, Mobile: {mobile_screenshots}")
    if log_dir:
        print(f"Screenshots saved in: {log_dir}")