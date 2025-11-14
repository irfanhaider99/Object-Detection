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

# === Target Configuration ===
TARGET_SIZE = 100  # Size of target area (100x100 pixels)
target_area_cam1 = None
target_area_cam2 = None
target_set_mode = False
right_wrist_id = 10  # YOLO pose keypoint index for right wrist

# === Main Loop ===
print("Press 's' to set target area using right wrist position, 'q' to quit")

def get_wrist_position(pose_results):
    """Get right wrist position from pose estimation results"""
    if pose_results.keypoints:
        for kpts in pose_results.keypoints.xy:
            kpts = kpts.cpu().numpy()
            if kpts.shape[0] > right_wrist_id:  # Ensure we have enough keypoints
                return tuple(map(int, kpts[right_wrist_id]))
    return None

while True:
    frames = [grab(cam) for cam in cams]
    if any(f is None for f in frames):
        print("Frame capture failed.")
        break

    # Process frames
    obj_results = [object_model.track(source=f, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0] for f in frames]
    pose_results = [pose_model(f, verbose=False)[0] for f in frames]

    # Check for target set command
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        target_set_mode = True
        print("Setting target area - place your right wrist on the desired target location")

    # Set target areas based on wrist position (single area per camera)
    if target_set_mode:
        wrist_positions = [get_wrist_position(pr) for pr in pose_results]
        
        if all(wp is not None for wp in wrist_positions):
            # Camera 1 target area (single area)
            target_area_cam1 = [
                (wrist_positions[0][0] - TARGET_SIZE//2, wrist_positions[0][1] - TARGET_SIZE//2),
                (wrist_positions[0][0] + TARGET_SIZE//2, wrist_positions[0][1] + TARGET_SIZE//2)
            ]
            
            # Camera 2 target area (single area)
            target_area_cam2 = [
                (wrist_positions[1][0] - TARGET_SIZE//2, wrist_positions[1][1] - TARGET_SIZE//2),
                (wrist_positions[1][0] + TARGET_SIZE//2, wrist_positions[1][1] + TARGET_SIZE//2)
            ]
            
            print(f"Target areas set:")
            print(f"Camera 1: {target_area_cam1}")
            print(f"Camera 2: {target_area_cam2}")
            target_set_mode = False

    # Process each camera view
    for i, (obj_res, pose_res, frame) in enumerate(zip(obj_results, pose_results, frames)):
        cam_label = f"Cam{i+1}"
        target_area = target_area_cam1 if i == 0 else target_area_cam2

        # Draw target area if set (single area)
        if target_area:
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
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check if center is in target area
                in_target = False
                if target_area:
                    in_target = (target_area[0][0] <= center_x <= target_area[1][0]) and \
                               (target_area[0][1] <= center_y <= target_area[1][1])
                
                color = (0, 255, 0) if in_target else (0, 0, 255)
                label = f"{object_model.names[clss[j]]} ID:{ids[j]}{' (IN TARGET)' if in_target else ''}"
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pose estimation visualization (right wrist only)
        wrist_pos = get_wrist_position(pose_res)
        if wrist_pos:
            cv2.circle(frame, wrist_pos, 8, (0, 255, 255), -1)
            cv2.putText(frame, "Wrist", (wrist_pos[0] + 10, wrist_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display
    cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
    if len(frames) > 1:
        cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

    if key == ord('q'):
        break

# Cleanup
for cam in cams:
    cam.StopGrabbing()
    cam.Close()
cv2.destroyAllWindows()
