import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
from collections import deque
import time
import os

# === Enhanced Configuration ===
STABILIZATION_FRAMES = 5               # Faster response than original 10
MIN_DETECTION_CONFIDENCE = 0.15        # More sensitive than original 0.25
MIN_CONSECUTIVE_DETECTIONS = 2         # Same as original
AREA_MARGIN = 0.15                     # 15% margin around target areas
DEBUG_MODE = True                      # Show extra detection info

# === Terminal Display Setup ===
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_status(cam1_status, cam1_count, cam2_status, cam2_count):
    clear_terminal()
    print("=== ENHANCED OBJECT DETECTION STATUS ===")
    if TARGET_AREAS[0]:  # Camera 1
        status = "AVAILABLE" if cam1_status[0] else "NOT AVAILABLE"
        print(f"Camera 1 - Area: {status} (Objects: {cam1_count[0]})")
    if TARGET_AREAS[1]:  # Camera 2
        status = "AVAILABLE" if cam2_status[0] else "NOT AVAILABLE"
        print(f"Camera 2 - Area: {status} (Objects: {cam2_count[0]})")
    
    if DEBUG_MODE:
        print("\n[DEBUG MODE ACTIVE]")
        print(f"- Confidence Threshold: {MIN_DETECTION_CONFIDENCE}")
        print(f"- Area Margin: {AREA_MARGIN*100}%")
        print(f"- Stabilization Frames: {STABILIZATION_FRAMES}")
    
    print("\nPress 'q' to quit")

# === Load Models ===
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

# === Flexible Target Areas ===
# Comment/uncomment areas to control tracking
# (Replaced by your requested TARGET_AREAS definition below)

# === Visualization Colors ===
COLORS = {
    'target': (255, 255, 0),        # Yellow for target area
    'in_target': (0, 255, 0),       # Green when object detected
    'wrist': (0, 255, 255),         # Cyan for object center
    'unavailable': (0, 0, 255),     # Red when no detection
    'debug': (255, 0, 255)          # Purple for debug/expanded area
}

# === Enhanced Area Checking ===
def is_in_target_area(center_x, center_y, target_area):
    """Check if point is in target area with margin"""
    x1, y1 = target_area[0]
    x2, y2 = target_area[1]
    w, h = x2 - x1, y2 - y1
    
    # Apply margin expansion
    expanded_x1 = x1 - w * AREA_MARGIN
    expanded_x2 = x2 + w * AREA_MARGIN
    expanded_y1 = y1 - h * AREA_MARGIN
    expanded_y2 = y2 + h * AREA_MARGIN
    
    return (expanded_x1 <= center_x <= expanded_x2 and 
            expanded_y1 <= center_y <= expanded_y2)

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

# === Improved Object Tracking (ADDED) ===
def get_object_center(box, frame):
    """Calculate precise object center point"""
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # For debugging - verify center calculation
    if DEBUG_MODE:
        # Get small ROI around center point
        roi_size = 20
        roi = frame[max(0, center_y - roi_size):center_y + roi_size,
                    max(0, center_x - roi_size):center_x + roi_size]
        if roi.size > 0:
            # Calculate average color in ROI
            avg_color = np.mean(roi, axis=(0, 1))
            cv2.putText(frame,
                        f"Center: {avg_color[0]:.0f},{avg_color[1]:.0f},{avg_color[2]:.0f}",
                        (center_x + 15, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return center_x, center_y

# === Your requested TARGET_AREAS override (ADDED) ===
TARGET_AREAS = [
    [  # Camera 1
        [(1287, 584), (1387, 684)],  # Only tracking your important area
    ],
    [  # Camera 2
        [(853, 376), (953, 476)],
    ]
]

# Filter out empty camera area lists
TARGET_AREAS = [cam_areas for cam_areas in TARGET_AREAS if cam_areas]

# Initialize trackers only for active cameras and areas
trackers = [[StatusTracker() for _ in cam_areas] for cam_areas in TARGET_AREAS]

# === Enhanced Visualization ===
def draw_detection_info(frame, target_areas, is_available_list, object_counts, cam_id):
    for area_idx, (target_area, is_available, object_count) in enumerate(zip(target_areas, is_available_list, object_counts)):
        color = COLORS['in_target'] if is_available else COLORS['unavailable']
        
        # Draw expanded detection area (debug)
        if DEBUG_MODE:
            x1, y1 = target_area[0]
            x2, y2 = target_area[1]
            w, h = x2 - x1, y2 - y1
            expanded_area = (
                (int(x1 - w * AREA_MARGIN), int(y1 - h * AREA_MARGIN)),
                (int(x2 + w * AREA_MARGIN), int(y2 + h * AREA_MARGIN))
            )
            cv2.rectangle(frame, expanded_area[0], expanded_area[1], COLORS['debug'], 1)
        
        # Draw target area
        cv2.rectangle(frame, target_area[0], target_area[1], color, 3)
        
        # Draw info panel
        info_panel = frame[target_area[0][1]-50:target_area[0][1], target_area[0][0]:target_area[1][0]]
        overlay = info_panel.copy()
        cv2.rectangle(overlay, (0, 0), (info_panel.shape[1], info_panel.shape[0]), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.8, info_panel, 0.2, 0, info_panel)
        
        # Draw text
        status = "AVAILABLE" if is_available else "NOT AVAILABLE"
        cv2.putText(frame, f"Area: {status}", (target_area[0][0]+10, target_area[0][1]-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Objects: {object_count}", (target_area[0][0]+10, target_area[0][1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# === Main Loop ===
try:
    # Initialize status variables
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
        
        # Process each camera with active areas
        terminal_update = False
        cam_status = [[False for _ in cam_areas] for cam_areas in TARGET_AREAS]
        cam_counts = [[0 for _ in cam_areas] for cam_areas in TARGET_AREAS]
        
        # === Modified Detection Loop (REPLACED as requested) ===
        for i in range(min(len(TARGET_AREAS), len(frames))):
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
                        
                    box = boxes[j]
                    center_x, center_y = get_object_center(box, frame)
                    
                    # Debug: Show raw detection point
                    if DEBUG_MODE:
                        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)  # Red dot for raw center
                    
                    for area_idx, target_area in enumerate(target_areas):
                        if is_in_target_area(center_x, center_y, target_area):
                            current_object_counts[area_idx] += 1
                            current_detected[area_idx] = True
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['in_target'], 2)
                            
                            # Draw tracking point (yellow)
                            cv2.circle(frame, (center_x, center_y), 8, COLORS['wrist'], -1)
                            
                            # Debug: Show vector from box to center
                            if DEBUG_MODE:
                                box_center_x = (x1 + x2) // 2
                                box_center_y = (y1 + y2) // 2
                                cv2.arrowedLine(frame, (box_center_x, box_center_y),
                                                (center_x, center_y), (255, 255, 0), 2)

            # Update trackers
            for area_idx in range(len(target_areas)):
                if trackers[i][area_idx].update(current_detected[area_idx], current_object_counts[area_idx]):
                    terminal_update = True
                cam_status[i][area_idx] = trackers[i][area_idx].stable_status
                cam_counts[i][area_idx] = current_object_counts[area_idx]
            
            # Draw visualization
            draw_detection_info(frame, target_areas, cam_status[i], cam_counts[i], i)
        
        # Update terminal if status changed
        if terminal_update:
            display_status(cam_status[0] if len(TARGET_AREAS) > 0 else [], 
                         cam_counts[0] if len(TARGET_AREAS) > 0 else [],
                         cam_status[1] if len(TARGET_AREAS) > 1 else [], 
                         cam_counts[1] if len(TARGET_AREAS) > 1 else [])
        
        # Show frames
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

