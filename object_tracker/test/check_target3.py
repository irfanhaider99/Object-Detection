import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
from collections import deque

# === Configuration ===
TARGET_AREA_CAM1 = [(808, 701), (908, 801)]  # Your Camera 1 target
TARGET_AREA_CAM2 = [(355, 233), (455, 333)]  # Your Camera 2 target
TARGET_COLOR_PRESENT = (0, 255, 0)  # Green when object present
TARGET_COLOR_ABSENT = (0, 0, 255)   # Red when no object
OBJECT_COLOR = (0, 255, 255)        # Yellow for object markers
CONFIDENCE_THRESHOLD = 0.4          # Lowered threshold for better detection
HISTORY_LENGTH = 5                  # Frames to confirm object presence

# === Initialize ===
model = YOLO("best_5.pt").to("cuda")
cams = pylon.TlFactory.GetInstance().EnumerateDevices()[:2]
cameras = [pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cam)) for cam in cams]
for cam in cameras:
    cam.Open()
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

def grab_frame(cam):
    result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if result.GrabSucceeded():
        img = result.Array
        result.Release()
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    return None

def is_in_target(box, target_area):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (target_area[0][0] <= cx <= target_area[1][0] and 
            target_area[0][1] <= cy <= target_area[1][1])

# For flicker reduction
detection_history = deque(maxlen=HISTORY_LENGTH)

print("Running enhanced target monitoring. Press 'q' to quit")

# === Main Loop ===
while True:
    frames = [grab_frame(cam) for cam in cameras]
    if any(f is None for f in frames):
        break

    # Process both cameras
    results = [model.track(f, conf=CONFIDENCE_THRESHOLD, verbose=False)[0] for f in frames]
    
    # Check for objects in target areas
    cam1_objects = []
    cam2_objects = []
    
    for i, (res, frame) in enumerate(zip(results, frames)):
        target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
        
        if res.boxes and res.boxes.xyxy is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if conf >= CONFIDENCE_THRESHOLD and is_in_target(box, target_area):
                    if i == 0:
                        cam1_objects.append((box, conf))
                    else:
                        cam2_objects.append((box, conf))
    
    # Update detection history for flicker reduction
    current_detection = len(cam1_objects) > 0 and len(cam2_objects) > 0
    detection_history.append(current_detection)
    
    # Require consistent detections to confirm presence
    confirmed_detections = sum(detection_history)
    object_present = confirmed_detections > 0  # Immediate detection
    
    # Update display for both cameras
    for i, (res, frame) in enumerate(zip(results, frames)):
        target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
        objects = cam1_objects if i == 0 else cam2_objects
        
        # Draw target area with appropriate color
        target_color = TARGET_COLOR_PRESENT if object_present else TARGET_COLOR_ABSENT
        cv2.rectangle(frame, target_area[0], target_area[1], target_color, 3)
        
        # Draw objects if any detected in this camera
        if objects:
            for box, conf in objects:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Tight bounding box (yellow)
                cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_COLOR, 2)
                
                # Yellow center dot (larger and brighter)
                cv2.circle(frame, (cx, cy), 10, OBJECT_COLOR, -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)  # Inner dot for better visibility
                
                # Confidence text
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, OBJECT_COLOR, 2)
        
        # Add status text
        status = "OBJECT DETECTED" if object_present else "SEARCHING"
        text_color = TARGET_COLOR_PRESENT if object_present else TARGET_COLOR_ABSENT
        text_pos = (target_area[0][0], target_area[0][1] - 20)
        cv2.putText(frame, status, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        # Display
        cv2.imshow(f"Camera {i+1}", cv2.resize(frame, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cam in cameras:
    cam.StopGrabbing()
    cam.Close()
cv2.destroyAllWindows()
