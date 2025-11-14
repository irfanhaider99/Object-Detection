import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
from collections import deque
import time
import os

# === Configuration ===
STABILIZATION_FRAMES = 5
MIN_DETECTION_CONFIDENCE = 0.3
MIN_CONSECUTIVE_DETECTIONS = 3

# === Terminal Display Setup ===
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_status(cam1_status, cam1_count, cam2_status, cam2_count):
    clear_terminal()
    print("=== OBJECT DETECTION STATUS ===")
    print(f"Camera 1: {'AVAILABLE' if cam1_status else 'NOT AVAILABLE'} (Objects: {cam1_count})")
    print(f"Camera 2: {'AVAILABLE' if cam2_status else 'NOT AVAILABLE'} (Objects: {cam2_count})")
    print("\nPress 'q' to quit")

# === Load models ===
#print("Loading YOLO models...")
object_model = YOLO("best_5.pt")
pose_model = YOLO("yolo11x-pose.pt")
object_model.to("cuda")
pose_model.to("cuda")

# === Camera Setup ===
def setup_cams():
    print("Initializing cameras...")
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) < 2:
        raise RuntimeError("Need at least 2 cameras connected")
    cams = [pylon.InstantCamera(factory.CreateDevice(dev)) for dev in devices[:2]]
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

# === Target Areas ===
TARGET_AREA_CAM1 = [(808, 701), (908, 801)]
TARGET_AREA_CAM2 = [(355, 233), (455, 333)]

# === Visualization Colors ===
COLORS = {
    'target': (255, 255, 0),
    'in_target': (0, 255, 0),
    'wrist': (0, 255, 255),
    'unavailable': (0, 0, 255)
}

# === Stabilization ===
class StatusTracker:
    def __init__(self):
        self.history = deque(maxlen=STABILIZATION_FRAMES)
        self.stable_status = False
        self.consecutive_count = 0
        self.last_print_status = None
        
    def update(self, current_status, object_count):
        self.history.append(current_status)
        
        # Update consecutive count
        if current_status:
            self.consecutive_count = min(self.consecutive_count + 1, MIN_CONSECUTIVE_DETECTIONS)
        else:
            self.consecutive_count = max(self.consecutive_count - 1, 0)
            
        # Update stable status if threshold reached
        if self.consecutive_count >= MIN_CONSECUTIVE_DETECTIONS:
            self.stable_status = True
        elif self.consecutive_count <= 0:
            self.stable_status = False
            
        # Print status if changed
        if self.stable_status != self.last_print_status:
            self.last_print_status = self.stable_status
            return True
        return False

# Initialize trackers
tracker_cam1 = StatusTracker()
tracker_cam2 = StatusTracker()

# === Visualization Functions ===
def draw_detection_info(frame, target_area, is_available, object_count, cam_id):
    color = COLORS['in_target'] if is_available else COLORS['unavailable']
    
    # Draw target area
    cv2.rectangle(frame, target_area[0], target_area[1], color, 3)
    
    # Draw info panel
    info_panel = frame[target_area[0][1]-40:target_area[0][1], target_area[0][0]:target_area[1][0]]
    overlay = info_panel.copy()
    cv2.rectangle(overlay, (0, 0), (info_panel.shape[1], info_panel.shape[0]), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, info_panel, 0.3, 0, info_panel)
    
    # Draw text
    status = "AVAILABLE" if is_available else "NOT AVAILABLE"
    cv2.putText(frame, f"Status: {status}", (target_area[0][0]+10, target_area[0][1]-25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Objects: {object_count}", (target_area[0][0]+10, target_area[0][1]-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# === Main Loop ===
try:
    display_status(False, 0, False, 0)  # Initial status
    
    while True:
        frames = [grab(cam) for cam in cams]
        if any(f is None for f in frames):
            time.sleep(0.1)
            continue

        # Process detections
        obj_results = [object_model.track(f, persist=True, conf=MIN_DETECTION_CONFIDENCE, verbose=False)[0] for f in frames]
        
        # Process each camera
        terminal_update = False
        cam_status = [False, False]
        cam_counts = [0, 0]
        
        for i, (frame, obj_res) in enumerate(zip(frames, obj_results)):
            target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
            current_objects = 0
            current_detected = False
            
            if obj_res.boxes and obj_res.boxes.xyxy is not None:
                boxes = obj_res.boxes.xyxy.cpu().numpy()
                confs = obj_res.boxes.conf.cpu().numpy()
                
                for j in range(len(boxes)):
                    if confs[j] < MIN_DETECTION_CONFIDENCE:
                        continue
                        
                    x1, y1, x2, y2 = boxes[j]
                    center_x, center_y = (x1 + x2)/2, (y1 + y2)/2
                    
                    if (target_area[0][0] <= center_x <= target_area[1][0] and
                        target_area[0][1] <= center_y <= target_area[1][1]):
                        current_objects += 1
                        current_detected = True
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS['in_target'], 2)
                        cv2.circle(frame, (int(center_x), int(center_y)), 8, COLORS['wrist'], -1)
            
            # Update tracker
            if i == 0:
                if tracker_cam1.update(current_detected, current_objects):
                    terminal_update = True
                cam_status[0] = tracker_cam1.stable_status
                cam_counts[0] = current_objects
            else:
                if tracker_cam2.update(current_detected, current_objects):
                    terminal_update = True
                cam_status[1] = tracker_cam2.stable_status
                cam_counts[1] = current_objects
            
            # Draw visualization
            draw_detection_info(frame, target_area, cam_status[i], cam_counts[i], i)
        
        # Update terminal if status changed
        if terminal_update:
            display_status(cam_status[0], cam_counts[0], cam_status[1], cam_counts[1])
        
        # Show frames
        cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
        if len(frames) > 1:
            cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cam in cams:
        cam.StopGrabbing()
        cam.Close()
    cv2.destroyAllWindows()
    print("Detection stopped.")
