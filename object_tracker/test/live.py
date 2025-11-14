import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
import time
from collections import deque
import torch

# === Device selection: use CUDA if available, else CPU ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load your trained object detector and pose estimator ===
object_model = YOLO("best_5.pt")
pose_model = YOLO("yolo11x-pose.pt")

object_model.to(device)
pose_model.to(device)

# === Setup Basler Cameras ===
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

# === Setup Output Videos ===
out1 = cv2.VideoWriter("cam1_live_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (800, 600))
out2 = cv2.VideoWriter("cam2_live_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (800, 600))

# === Zone Monitoring Setup (High Precision) ===
# Your exact coordinates for both cameras
TOOL_ZONES = {
    "Cam1": {"name": "ToolZone", "coords": (894, 725, 913, 761), "target_class": None},
    "Cam2": {"name": "ToolZone", "coords": (428, 284, 443, 324), "target_class": None}
}

def check_object_in_zone_precise(zone_coords, detected_boxes, detected_classes, target_class=None):
    """
    High precision zone checking with false detection minimization
    Returns True if significant object presence is detected, False otherwise
    """
    zx1, zy1, zx2, zy2 = zone_coords
    zone_area = (zx2 - zx1) * (zy2 - zy1)
    zone_center_x = (zx1 + zx2) / 2
    zone_center_y = (zy1 + zy2) / 2
    
    best_object_found = False
    best_score = 0
    
    for i, box in enumerate(detected_boxes):
        bx1, by1, bx2, by2 = box
        
        # Check if target class matches (if specified)
        if target_class is not None and detected_classes[i] != target_class:
            continue
            
        # Calculate intersection area
        ix1 = max(zx1, bx1)
        iy1 = max(zy1, by1)
        ix2 = min(zx2, bx2)
        iy2 = min(zy2, by2)
        
        # If there's an intersection
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            object_area = (bx2 - bx1) * (by2 - by1)
            
            # Calculate ratios
            object_in_zone_ratio = intersection_area / object_area if object_area > 0 else 0
            zone_coverage_ratio = intersection_area / zone_area if zone_area > 0 else 0
            
            # Calculate object center and distance to zone center
            obj_center_x = (bx1 + bx2) / 2
            obj_center_y = (by1 + by2) / 2
            center_distance = ((obj_center_x - zone_center_x)**2 + (obj_center_y - zone_center_y)**2)**0.5
            
            # Enhanced scoring system to minimize false detections
            score = 0
            
            # High score if object center is inside zone
            if zx1 <= obj_center_x <= zx2 and zy1 <= obj_center_y <= zy2:
                score += 50
                
            # Score based on zone coverage
            if zone_coverage_ratio > 0.4:  # Good coverage
                score += 40
            elif zone_coverage_ratio > 0.2:  # Moderate coverage
                score += 20
                
            # Score based on object overlap
            if object_in_zone_ratio > 0.5:  # Most of object in zone
                score += 30
            elif object_in_zone_ratio > 0.3:  # Partial overlap
                score += 15
                
            # Penalty for objects too far from center
            if center_distance > 30:  # Adjust based on zone size
                score -= 20
                
            # Minimum intersection area requirement
            if intersection_area < 100:  # At least 100 pixels
                score -= 30
                
            if score > 30:  # Threshold for detection
                if score > best_score:
                    best_score = score
                    best_object_found = True
                print(f" Object detected: score={score:.0f}, zone_cov={zone_coverage_ratio:.2f}, obj_in_zone={object_in_zone_ratio:.2f}, center_dist={center_distance:.1f}")
    
    return best_object_found

# === Wrist motion history ===
motion_history = {
    "Cam1_lwrist": deque(maxlen=5),
    "Cam1_rwrist": deque(maxlen=5),
    "Cam2_lwrist": deque(maxlen=5),
    "Cam2_rwrist": deque(maxlen=5),
}
MOTION_THRESHOLD = 2.5  # pixels per frame average

def is_actively_moving(label, new_pos):
    history = motion_history[label]
    history.append(new_pos)
    if len(history) < 2:
        return False
    diffs = [np.linalg.norm(np.array(history[i]) - np.array(history[i - 1])) for i in range(1, len(history))]
    avg_motion = np.mean(diffs)
    return avg_motion > MOTION_THRESHOLD

print("Real-time Object Detection + Zone Monitoring + Wrist Motion Started (press 'q' to quit)")

frame_count = 0
start_time = time.time()

while True:
    frames = [grab(cam) for cam in cams]
    if any(f is None for f in frames):
        print("Frame capture failed.")
        break

    frame_count += 1
    
    # Show live processing info every 30 frames to avoid spam
    if frame_count % 30 == 0:
        elapsed_time = time.time() - start_time
        fps_processing = frame_count / elapsed_time
        print(f"Live processing: Frame {frame_count} - Speed: {fps_processing:.1f} FPS")

    # Object detection and tracking
    obj_results = [object_model.track(source=f, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False, imgsz=640)[0] for f in frames]
    pose_results = [pose_model(f, verbose=False, imgsz=640)[0] for f in frames]

    for i, (obj_res, pose_res, frame) in enumerate(zip(obj_results, pose_results, frames)):
        cam_label = f"Cam{i+1}"
        
        # Collect detected boxes and classes for zone monitoring
        detected_boxes = []
        detected_classes = []

        # === Object Detection and Tracking ===
        if obj_res.boxes and obj_res.boxes.xyxy is not None:
            ids = obj_res.boxes.id
            boxes = obj_res.boxes.xyxy.cpu().numpy()
            clss = obj_res.boxes.cls.cpu().numpy().astype(int)
            if ids is None:
                ids = np.arange(len(boxes))
            else:
                ids = ids.cpu().numpy().astype(int)

            # Store for zone monitoring
            detected_boxes = boxes
            detected_classes = clss

            for j in range(len(ids)):
                x1, y1, x2, y2 = boxes[j]
                label = f"{object_model.names[clss[j]]} ID:{ids[j]}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # === High-Precision Zone Monitoring ===
        if cam_label in TOOL_ZONES:
            tool_zone = TOOL_ZONES[cam_label]
            zone_coords = tool_zone["coords"]
            target_class = tool_zone.get("target_class", None)
            zone_name = tool_zone["name"]
            
            # Check if object is present in zone with high precision
            object_present = check_object_in_zone_precise(zone_coords, detected_boxes, detected_classes, target_class)
            
            # Draw zone with appropriate color
            x1, y1, x2, y2 = zone_coords
            if object_present:
                zone_color = (0, 255, 0)  # Green - object present
                status_text = "OBJECT PLACED"
                status_color = (0, 255, 0)
                if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f" {cam_label} {zone_name}: TOOL/OBJECT PRESENT")
            else:
                zone_color = (0, 0, 255)  # Red - zone empty
                status_text = "NO OBJECT"
                status_color = (0, 0, 255)
                if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f" {cam_label} {zone_name}: ZONE EMPTY")
            
            # Draw zone rectangle with thick border for visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 4)
            
            # Draw zone name inside the zone
            text_x = x1 + 2
            text_y = y1 + 15
            cv2.putText(frame, zone_name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)
            
            # === ON-SCREEN STATUS DISPLAY ===
            # Large status text at top of video
            status_display = f"{cam_label} ZONE: {status_text}"
            text_size = cv2.getTextSize(status_display, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2  # Center horizontally
            text_y = 40  # Top of screen
            
            # Background rectangle for better visibility
            cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, status_display, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

        # === Wrist Motion Detection ===
        if pose_res.keypoints:
            for kpts in pose_res.keypoints.xy:
                kpts = kpts.cpu().numpy()
                if kpts.shape[0] >= 11:
                    lwrist = tuple(map(int, kpts[9]))
                    rwrist = tuple(map(int, kpts[10]))
                    lelbow = tuple(map(int, kpts[7]))
                    relbow = tuple(map(int, kpts[8]))

                    scale_factor = 0.3  # smaller bounding box for hand only
                    lwrist_box_size = int(np.linalg.norm(np.array(lwrist) - np.array(lelbow)) * scale_factor)
                    rwrist_box_size = int(np.linalg.norm(np.array(rwrist) - np.array(relbow)) * scale_factor)

                    wrist_color = (0, 255, 255)

                    if is_actively_moving(f"{cam_label}_lwrist", lwrist):
                        cv2.rectangle(frame,
                                      (lwrist[0] - lwrist_box_size, lwrist[1] - lwrist_box_size),
                                      (lwrist[0] + lwrist_box_size, lwrist[1] + lwrist_box_size),
                                      wrist_color, 2)
                        cv2.putText(frame, "LWrist", (lwrist[0] - lwrist_box_size, lwrist[1] - lwrist_box_size - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, wrist_color, 2)

                    if is_actively_moving(f"{cam_label}_rwrist", rwrist):
                        cv2.rectangle(frame,
                                      (rwrist[0] - rwrist_box_size, rwrist[1] - rwrist_box_size),
                                      (rwrist[0] + rwrist_box_size, rwrist[1] + rwrist_box_size),
                                      wrist_color, 2)
                        cv2.putText(frame, "RWrist", (rwrist[0] - rwrist_box_size, rwrist[1] - rwrist_box_size - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, wrist_color, 2)

    # === Display Live Feed ===
    cv2.imshow("Camera 1 - Live Zone Monitoring", cv2.resize(frames[0], (800, 600)))
    if len(frames) > 1:
        cv2.imshow("Camera 2 - Live Zone Monitoring", cv2.resize(frames[1], (800, 600)))

    # === Save Output Videos (Optional) ===
    out1.write(cv2.resize(frames[0], (800, 600)))
    if len(frames) > 1:
        out2.write(cv2.resize(frames[1], (800, 600)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
total_time = time.time() - start_time
print(f"\nLive processing complete! Processed {frame_count} frames in {total_time:.1f} seconds")
print(f"Average processing speed: {frame_count/total_time:.1f} FPS")
print("Cleaning up...")

out1.release()
out2.release()

for cam in cams:
    cam.StopGrabbing()
    cam.Close()

cv2.destroyAllWindows()

print("Live output videos saved as cam1_live_output.avi and cam2_live_output.avi")
