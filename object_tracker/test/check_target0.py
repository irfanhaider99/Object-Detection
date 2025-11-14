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

# === Fixed Target Areas (from your output) ===
TARGET_AREA_CAM1 = [(840, 727), (940, 827)]  # Camera 1 target area
TARGET_AREA_CAM2 = [(342, 250), (442, 350)]  # Camera 2 target area
TARGET_SIZE = 100  # Size of target area (100x100 pixels based on your coordinates)

# === Main Loop ===
print("Tracking objects in predefined target areas. Press 'q' to quit")

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
        cam_label = f"Cam{i+1}"
        target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2

        # Draw target area
        cv2.rectangle(frame, target_area[0], target_area[1], (255, 255, 0), 2)
        cv2.putText(frame, "Target Area", (target_area[0][0], target_area[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Object detection and tracking
        if obj_res.boxes and obj_res.boxes.xyxy is not None:
            boxes = obj_res.boxes.xyxy.cpu().numpy()
            clss = obj_res.boxes.cls.cpu().numpy().astype(int)
            ids = obj_res.boxes.id.cpu().numpy().astype(int) if obj_res.boxes.id is not None else np.arange(len(boxes))

            for j in range(len(ids)):
                x1, y1, x2, y2 = boxes[j]
                
                # Calculate box center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check if center is in target area
                in_target = (target_area[0][0] <= center_x <= target_area[1][0]) and \
                           (target_area[0][1] <= center_y <= target_area[1][1])
                
                # Set color and label
                color = (0, 255, 0) if in_target else (0, 0, 255)
                label = f"{object_model.names[clss[j]]} ID:{ids[j]}{' (IN TARGET)' if in_target else ''}"
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Additional visualization for objects in target area
                if in_target:
                    # Draw center point
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 255), -1)
                    
                    # Draw line from center to target area center
                    target_center = (
                        (target_area[0][0] + target_area[1][0]) // 2,
                        (target_area[0][1] + target_area[1][1]) // 2
                    )
                    cv2.line(frame, (int(center_x), int(center_y)), target_center, (255, 0, 255), 2)

        # Pose estimation visualization (optional)
        if pose_res.keypoints:
            for kpts in pose_res.keypoints.xy:
                kpts = kpts.cpu().numpy()
                if kpts.shape[0] >= 11:
                    # Draw right wrist (index 10)
                    rwrist = tuple(map(int, kpts[10]))
                    cv2.circle(frame, rwrist, 5, (0, 255, 255), -1)

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
