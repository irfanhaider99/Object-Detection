import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
from collections import deque

# === Load models ===
object_model = YOLO("best_5.pt")
pose_model = YOLO("yolo11x-pose.pt")
object_model.to("cuda")
pose_model.to("cuda")

# === Camera Setup ===
def setup_cams():
    factory = pylon.TlFactory.GetInstance()
    cams = [pylon.InstantCamera(factory.CreateDevice(dev)) for dev in factory.EnumerateDevices()[:2]]
    for cam in cams:
        cam.Open()
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return cams

def grab(cam):
    result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if result.GrabSucceeded():
        img = result.Array
        result.Release()
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result.Release()
    return None

cams = setup_cams()

# === Fixed Target Areas (updated with your coordinates) ===
TARGET_AREA_CAM1 = [(808, 701), (908, 801)]  # Camera 1 target area (100x100)
TARGET_AREA_CAM2 = [(355, 233), (455, 333)]  # Camera 2 target area (100x100)

# === Enhanced Visualization Parameters ===
TARGET_COLOR = (255, 255, 0)  # Cyan for target area
IN_TARGET_COLOR = (0, 255, 0)  # Green for objects in target
OUT_TARGET_COLOR = (0, 0, 255) # Red for objects outside target
WRIST_COLOR = (0, 255, 255)    # Yellow for wrist

# === Main Loop ===
print("Tracking objects in predefined target areas. Press 'q' to quit")

def draw_target_info(frame, target_area, objects_in_target):
    """Helper function to draw target information"""
    # Draw target area
    cv2.rectangle(frame, target_area[0], target_area[1], TARGET_COLOR, 2)
    
    # Calculate center
    center = ((target_area[0][0] + target_area[1][0]) // 2, 
              (target_area[0][1] + target_area[1][1]) // 2)
    
    # Draw target info
    info_text = f"Target: {objects_in_target} object(s)"
    cv2.putText(frame, info_text, (center[0] - 60, target_area[0][1] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, TARGET_COLOR, 2)

while True:
    frames = [grab(cam) for cam in cams]
    if any(f is None for f in frames):
        print("Frame capture failed.")
        break

    # Process frames
    obj_results = [object_model.track(source=f, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0] for f in frames]
    pose_results = [pose_model(f, verbose=False)[0] for f in frames]

    # Process each camera view
    for i, (obj_res, pose_res, frame) in enumerate(zip(obj_results, pose_results, frames)):
        target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
        objects_in_target = 0

        # Object detection and tracking
        if obj_res.boxes and obj_res.boxes.xyxy is not None:
            boxes = obj_res.boxes.xyxy.cpu().numpy()
            clss = obj_res.boxes.cls.cpu().numpy().astype(int)
            ids = obj_res.boxes.id.cpu().numpy().astype(int) if obj_res.boxes.id is not None else np.arange(len(boxes))

            for j in range(len(ids)):
                x1, y1, x2, y2 = boxes[j]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check if center is in target area
                in_target = (target_area[0][0] <= center_x <= target_area[1][0]) and \
                           (target_area[0][1] <= center_y <= target_area[1][1])
                
                if in_target:
                    objects_in_target += 1
                
                # Set visual properties
                color = IN_TARGET_COLOR if in_target else OUT_TARGET_COLOR
                label = f"{object_model.names[clss[j]]} ID:{ids[j]}"
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Additional visualization for objects in target
                if in_target:
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, WRIST_COLOR, -1)

        # Draw target information
        draw_target_info(frame, target_area, objects_in_target)

        # Pose estimation visualization (optional)
        if pose_res.keypoints:
            for kpts in pose_res.keypoints.xy:
                kpts = kpts.cpu().numpy()
                if kpts.shape[0] >= 11:
                    # Draw right wrist (index 10)
                    rwrist = tuple(map(int, kpts[10]))
                    cv2.circle(frame, rwrist, 8, WRIST_COLOR, -1)

    # Display
    cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
    if len(frames) > 1:
        cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
for cam in cams:
    cam.StopGrabbing()
    cam.Close()
cv2.destroyAllWindows()
