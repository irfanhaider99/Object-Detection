import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon

# === Configuration ===
TARGET_AREA_CAM1 = [(808, 701), (908, 801)]  # Your Camera 1 target
TARGET_AREA_CAM2 = [(355, 233), (455, 333)]  # Your Camera 2 target
TARGET_COLOR_PRESENT = (0, 255, 0)  # Green when object present
TARGET_COLOR_ABSENT = (0, 0, 255)   # Red when no object
TARGET_THICKNESS = 3                # Thicker border for visibility

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

print("Running target point monitoring. Press 'q' to quit")

# === Main Loop ===
while True:
    frames = [grab_frame(cam) for cam in cameras]
    if any(f is None for f in frames):
        break

    # Process both cameras
    results = [model.track(f, conf=0.25, verbose=False)[0] for f in frames]
    
    # Check for objects in target areas
    cam1_objects = []
    cam2_objects = []
    
    for i, (res, frame) in enumerate(zip(results, frames)):
        target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
        
        if res.boxes and res.boxes.xyxy is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            for box in boxes:
                if is_in_target(box, target_area):
                    if i == 0:
                        cam1_objects.append(box)
                    else:
                        cam2_objects.append(box)
    
    # Determine if object is present in BOTH views
    object_present = len(cam1_objects) > 0 and len(cam2_objects) > 0
    
    # Update display for both cameras
    for i, frame in enumerate(frames):
        target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
        
        # Draw target area with appropriate color
        color = TARGET_COLOR_PRESENT if object_present else TARGET_COLOR_ABSENT
        cv2.rectangle(frame, target_area[0], target_area[1], color, TARGET_THICKNESS)
        
        # Add status text
        status = "OBJECT PRESENT" if object_present else "NO OBJECT"
        cv2.putText(frame, status, (target_area[0][0], target_area[0][1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display
        cv2.imshow(f"Camera {i+1}", cv2.resize(frame, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cam in cameras:
    cam.StopGrabbing()
    cam.Close()
cv2.destroyAllWindows()
