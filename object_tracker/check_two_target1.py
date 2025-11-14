import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
from collections import deque
import time
import os

# === Configuration ===
STABILIZATION_FRAMES = 5
MIN_DETECTION_CONFIDENCE = 0.15
MIN_CONSECUTIVE_DETECTIONS = 2

# === Terminal Display Setup ===
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_status(cam1_status, cam1_count, cam2_status, cam2_count):
    clear_terminal()
    print("=== OBJECT DETECTION STATUS ===")
    if TARGET_AREAS[0]:  # If Camera 1 has active areas
        print(f"Camera 1 - Area: {'AVAILABLE' if cam1_status[0] else 'NOT AVAILABLE'} (Objects: {cam1_count[0]})")
    if TARGET_AREAS[1]:  # If Camera 2 has active areas
        print(f"Camera 2 - Area: {'AVAILABLE' if cam2_status[0] else 'NOT AVAILABLE'} (Objects: {cam2_count[0]})")
    print("\nPress 'q' to quit")

# === Load models ===
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
# Comment out all areas except the one you want to track
TARGET_AREAS = [
    [ # Camera 1 areas
        #[(808, 701), (908, 801)],  # Area 1 (comment out if not needed)
        [(1287, 584), (1387, 684)],  # Area 2 (uncomment to track this area)
    ],
    [ # Camera 2 areas
        #[(355, 233), (455, 333)],  # Area 1 (comment out if not needed)
        [(853, 376), (953, 476)],    # Area 2 (uncomment to track this area)
    ]
]

# Filter out empty camera area lists (if all areas are commented out)
TARGET_AREAS = [cam_areas for cam_areas in TARGET_AREAS if cam_areas]

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
        
        if current_status:
            self.consecutive_count = min(self.consecutive_count + 1, MIN_CONSECUTIVE_DETECTIONS)
        else:
            self.consecutive_count = max(self.consecutive_count - 1, 0)
            
        if self.consecutive_count >= MIN_CONSECUTIVE_DETECTIONS:
            self.stable_status = True
        elif self.consecutive_count <= 0:
            self.stable_status = False
            
        if self.stable_status != self.last_print_status:
            self.last_print_status = self.stable_status
            return True
        return False

# Initialize trackers only for active cameras and areas
trackers = [[StatusTracker() for _ in cam_areas] for cam_areas in TARGET_AREAS]

# === Visualization Functions ===
def draw_detection_info(frame, target_areas, is_available_list, object_counts, cam_id):
    for area_idx, (target_area, is_available, object_count) in enumerate(zip(target_areas, is_available_list, object_counts)):
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
        cv2.putText(frame, f"Area: {status}", (target_area[0][0]+10, target_area[0][1]-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Objects: {object_count}", (target_area[0][0]+10, target_area[0][1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# === Main Loop ===
try:
    # Initialize status variables based on active areas
    cam_status = [[False for _ in cam_areas] for cam_areas in TARGET_AREAS]
    cam_counts = [[0 for _ in cam_areas] for cam_areas in TARGET_AREAS]
    
    display_status(cam_status[0] if len(TARGET_AREAS) > 0 else [], 
                 cam_counts[0] if len(TARGET_AREAS) > 0 else [],
                 cam_status[1] if len(TARGET_AREAS) > 1 else [], 
                 cam_counts[1] if len(TARGET_AREAS) > 1 else [])
    
    while True:
        frames = [grab(cam) for cam in cams]
        if any(f is None for f in frames):
            time.sleep(0.1)
            continue

        # Process detections
        obj_results = [object_model.track(f, persist=True, conf=MIN_DETECTION_CONFIDENCE, verbose=False)[0] for f in frames]
        
        # Process each camera that has active areas
        terminal_update = False
        cam_status = [[False for _ in cam_areas] for cam_areas in TARGET_AREAS]
        cam_counts = [[0 for _ in cam_areas] for cam_areas in TARGET_AREAS]
        
        for i in range(min(len(TARGET_AREAS), len(frames))):  # Only process cameras with active areas
            frame = frames[i]
            obj_res = obj_results[i]
            target_areas = TARGET_AREAS[i]
            current_object_counts = [0] * len(target_areas)
            current_detected = [False] * len(target_areas)
            
            if obj_res.boxes and obj_res.boxes.xyxy is not None:
                boxes = obj_res.boxes.xyxy.cpu().numpy()
                confs = obj_res.boxes.conf.cpu().numpy()
                
                for j in range(len(boxes)):
                    if confs[j] < MIN_DETECTION_CONFIDENCE:
                        continue
                        
                    x1, y1, x2, y2 = boxes[j]
                    center_x, center_y = (x1 + x2)/2, (y1 + y2)/2
                    
                    for area_idx, target_area in enumerate(target_areas):
                        if (target_area[0][0] <= center_x <= target_area[1][0] and
                            target_area[0][1] <= center_y <= target_area[1][1]):
                            current_object_counts[area_idx] += 1
                            current_detected[area_idx] = True
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS['in_target'], 2)
                            cv2.circle(frame, (int(center_x), int(center_y)), 8, COLORS['wrist'], -1)
            
            # Update trackers for active areas of this camera
            for area_idx in range(len(target_areas)):
                if trackers[i][area_idx].update(current_detected[area_idx], current_object_counts[area_idx]):
                    terminal_update = True
                cam_status[i][area_idx] = trackers[i][area_idx].stable_status
                cam_counts[i][area_idx] = current_object_counts[area_idx]
            
            # Draw visualization for active areas
            draw_detection_info(frame, target_areas, cam_status[i], cam_counts[i], i)
        
        # Update terminal if status changed
        if terminal_update:
            display_status(cam_status[0] if len(TARGET_AREAS) > 0 else [], 
                         cam_counts[0] if len(TARGET_AREAS) > 0 else [],
                         cam_status[1] if len(TARGET_AREAS) > 1 else [], 
                         cam_counts[1] if len(TARGET_AREAS) > 1 else [])
        
        # Show frames for cameras that have active areas
        for i in range(min(len(TARGET_AREAS), len(frames))):
            cv2.imshow(f"Camera {i+1}", cv2.resize(frames[i], (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cam in cams:
        cam.StopGrabbing()
        cam.Close()
    cv2.destroyAllWindows()
    print("Detection stopped.")
