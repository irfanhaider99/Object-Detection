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

# === Enhanced Configuration (kept for parity) ===
AREA_MARGIN = 0.15  # not directly used in this variant
DEBUG_MODE = True   # toggle on-screen debug text

# === Camera-specific confidence thresholds ===
CAMERA_SPECIFIC_CONFIDENCE = {
    0: 0.10,  # CAM1 - High Sensitivity
    1: 0.15   # CAM2 - Standard
}

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

# === Target Point Configuration ===
TARGET_POINTS = [
    (1317, 634),  # Camera 1 target point (x,y)
    (903, 426)    # Camera 2 target point (x,y)
]

# Radius around target point to consider as detection
DETECTION_RADIUS = 50  # pixels

# === Target Area Creation (square around the point) ===
def create_target_area(center_point, size=100):
    """Create target area around center point"""
    x, y = center_point
    half_size = size // 2
    return [(x - half_size, y - half_size), (x + half_size, y + half_size)]

# Build TARGET_AREAS from TARGET_POINTS (one area per camera)
TARGET_AREAS = [
    [create_target_area(TARGET_POINTS[0])],  # Camera 1
    [create_target_area(TARGET_POINTS[1])]   # Camera 2
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

# === Visualization Functions (camera labels) ===
def draw_detection_info(frame, target_areas, is_available_list, object_counts, cam_id):
    # Camera label once per frame
    if cam_id == 0:
        cv2.putText(frame, "CAM1 (High Sensitivity) - Locked to CAM2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "CAM2 (Standard) - Reference", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for area_idx, (target_area, is_available, object_count) in enumerate(
        zip(target_areas, is_available_list, object_counts)
    ):
        color = COLORS['in_target'] if is_available else COLORS['unavailable']
        
        # Draw target area
        cv2.rectangle(frame, target_area[0], target_area[1], color, 3)
        
        # Draw info panel
        info_panel = frame[target_area[0][1]-40:target_area[0][1], target_area[0][0]:target_area[1][0]]
        if info_panel.size > 0:
            overlay = info_panel.copy()
            cv2.rectangle(overlay, (0, 0), (info_panel.shape[1], info_panel.shape[0]), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, info_panel, 0.3, 0, info_panel)
        
        # Draw text
        status = "AVAILABLE" if is_available else "NOT AVAILABLE"
        cv2.putText(frame, f"Area: {status}", (target_area[0][0]+10, target_area[0][1]-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Objects: {object_count}", (target_area[0][0]+10, target_area[0][1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# === Object Center ===
def get_object_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

# === Enhanced Area Checking (margin multiplier) ===
def is_in_target_area(center_x, center_y, target_area, target_point, margin_multiplier=1.0):
    """Improved detection with adjustable margins"""
    tx, ty = target_point
    
    # Dynamic circular radius
    dynamic_radius = DETECTION_RADIUS * margin_multiplier
    distance = np.sqrt((center_x - tx)**2 + (center_y - ty)**2)
    
    # Rectangular area
    in_rect = (target_area[0][0] <= center_x <= target_area[1][0] and
               target_area[0][1] <= center_y <= target_area[1][1])
    # Circular radius
    in_circle = distance <= dynamic_radius
    
    return in_rect or in_circle

# === Appearance features for cross-camera matching ===
def extract_hsv_hist(frame, box, bins=(16,16,16)):
    """Extract normalized HSV 3D histogram for the bbox."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180, 0,256, 0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def bhattacharyya_distance(hist1, hist2):
    if hist1 is None or hist2 is None:
        return 1.0
    # Using OpenCV compareHist with Bhattacharyya
    return float(cv2.compareHist(hist1.astype('float32'), hist2.astype('float32'), cv2.HISTCMP_BHATTACHARYYA))

# Keep last good reference to survive brief occlusions
last_ref = {
    "hist": None,
    "cls": None,
    "box": None,
    "center": None,
    "conf": None,
}

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

        # Use min threshold across cameras so CAM1 can keep low-conf detections
        min_conf_for_tracking = min([MIN_DETECTION_CONFIDENCE] + list(CAMERA_SPECIFIC_CONFIDENCE.values()))
        obj_results = [object_model.track(f, persist=True, conf=min_conf_for_tracking, verbose=False)[0] for f in frames]
        
        # Process each camera that has active areas
        terminal_update = False
        cam_status = [[False for _ in cam_areas] for cam_areas in TARGET_AREAS]
        cam_counts = [[0 for _ in cam_areas] for cam_areas in TARGET_AREAS]

        # ---------- 1) Select REFERENCE on Camera 2 ----------
        ref_found = False
        ref_box = None
        ref_center = None
        ref_conf = None
        ref_cls = None
        cam2_idx = 1  # camera 2 index
        if cam2_idx < len(frames) and cam2_idx < len(obj_results):
            frame2 = frames[cam2_idx]
            obj2 = obj_results[cam2_idx]
            target_area2 = TARGET_AREAS[cam2_idx][0]
            target_point2 = TARGET_POINTS[cam2_idx]
            current_min_conf2 = CAMERA_SPECIFIC_CONFIDENCE.get(cam2_idx, MIN_DETECTION_CONFIDENCE)

            best_score = -1e9
            if obj2.boxes and obj2.boxes.xyxy is not None:
                boxes2 = obj2.boxes.xyxy.cpu().numpy()
                confs2 = obj2.boxes.conf.cpu().numpy()
                clss2 = obj2.boxes.cls.cpu().numpy() if obj2.boxes.cls is not None else np.full(len(boxes2), -1)

                for j in range(len(boxes2)):
                    if confs2[j] < current_min_conf2:
                        continue
                    bx = boxes2[j]
                    cx, cy = get_object_center(bx)
                    # prioritize being inside target area/radius and close to it, plus confidence
                    in_area = is_in_target_area(cx, cy, target_area2, target_point2, margin_multiplier=1.0)
                    if not in_area:
                        continue
                    dist = np.hypot(cx - target_point2[0], cy - target_point2[1])
                    # scoring: higher conf and smaller distance are better
                    score = float(confs2[j] * 2.0 - 0.005 * dist)
                    if score > best_score:
                        best_score = score
                        ref_box = bx
                        ref_center = (cx, cy)
                        ref_conf = float(confs2[j])
                        ref_cls = int(clss2[j]) if clss2[j] >= 0 else None
                        ref_found = True

                # Draw chosen reference on cam2
                if ref_found:
                    x1, y1, x2, y2 = map(int, ref_box)
                    cv2.rectangle(frame2, (x1, y1), (x2, y2), COLORS['in_target'], 2)
                    cv2.circle(frame2, (int(ref_center[0]), int(ref_center[1])), 8, COLORS['wrist'], -1)
                    cv2.line(frame2, (int(ref_center[0]), int(ref_center[1])), target_point2, (255, 255, 255), 2)
                    if DEBUG_MODE:
                        cv2.putText(frame2, f"REF Conf: {ref_conf:.2f}/{current_min_conf2:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Update/keep reference histogram
        if ref_found:
            ref_hist = extract_hsv_hist(frames[cam2_idx], ref_box)
            if ref_hist is not None:
                last_ref.update({
                    "hist": ref_hist,
                    "cls": ref_cls,
                    "box": ref_box,
                    "center": ref_center,
                    "conf": ref_conf,
                })
        ref_hist = last_ref["hist"]
        ref_cls  = last_ref["cls"]

        # ---------- 2) Camera loops (CAM1 matches to CAM2 reference) ----------
        for i in range(min(len(TARGET_AREAS), len(frames))):  # Only process cameras with active areas
            frame = frames[i]
            obj_res = obj_results[i]
            target_area = TARGET_AREAS[i][0]  # One area per camera here
            target_point = TARGET_POINTS[i]

            # Camera-specific confidence threshold
            current_min_confidence = CAMERA_SPECIFIC_CONFIDENCE.get(i, MIN_DETECTION_CONFIDENCE)

            # Arrays to integrate with existing tracker/update logic
            current_object_counts = [0]
            current_detected = [False]
            
            if obj_res.boxes and obj_res.boxes.xyxy is not None:
                boxes = obj_res.boxes.xyxy.cpu().numpy()
                confs = obj_res.boxes.conf.cpu().numpy()
                clss = obj_res.boxes.cls.cpu().numpy() if obj_res.boxes.cls is not None else np.full(len(boxes), -1)
                
                # CAM2: we already picked the reference above; here we just count/display it
                if i == 1:
                    if ref_found:
                        current_object_counts[0] = 1
                        current_detected[0] = True
                    # draw target visuals always
                    radius_to_draw = int(DETECTION_RADIUS * 1.0)
                    cv2.circle(frame, target_point, 5, (0, 0, 255), -1)
                    cv2.circle(frame, target_point, radius_to_draw, (0, 255, 255), 2)

                # CAM1: find best match to CAM2 reference by appearance + area check
                else:
                    best_dist = 1e9
                    best_idx = -1
                    best_box = None
                    best_center = None
                    margin_multiplier = 1.25  # CAM1 is more lenient

                    for j in range(len(boxes)):
                        if confs[j] < current_min_confidence:
                            continue

                        bx = boxes[j]
                        cx, cy = get_object_center(bx)

                        # Require being in/near target point for CAM1 (keeps it stable spatially)
                        if not is_in_target_area(cx, cy, target_area, target_point, margin_multiplier):
                            continue

                        # If we have a reference hist, use it; otherwise fall back to highest conf near area
                        if ref_hist is not None:
                            cand_hist = extract_hsv_hist(frame, bx)
                            dist = bhattacharyya_distance(ref_hist, cand_hist)
                            # Optional: if classes available, prefer same class
                            if ref_cls is not None and clss[j] >= 0 and int(clss[j]) != int(ref_cls):
                                dist *= 1.5  # soft penalty
                        else:
                            # No ref yet -> use distance to target + confidence as surrogate
                            dist = 1.0 - float(confs[j]) + 0.002 * np.hypot(cx - target_point[0], cy - target_point[1])

                        if dist < best_dist:
                            best_dist = dist
                            best_idx = j
                            best_box = bx
                            best_center = (cx, cy)

                    # Accept match if it’s sufficiently similar
                    accept = False
                    if ref_hist is not None:
                        # Bhattacharyya distance in [0..1]; < ~0.5 is often reasonable
                        accept = best_idx >= 0 and best_dist <= 0.5
                    else:
                        accept = best_idx >= 0  # no ref: accept best spatial/conf candidate

                    if accept:
                        current_object_counts[0] = 1
                        current_detected[0] = True

                        x1, y1, x2, y2 = map(int, best_box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['in_target'], 2)
                        cv2.circle(frame, (int(best_center[0]), int(best_center[1])), 8, COLORS['wrist'], -1)
                        cv2.line(frame, (int(best_center[0]), int(best_center[1])), target_point, (255, 255, 255), 2)

                        if DEBUG_MODE:
                            txt = f"MatchDist: {best_dist:.2f}"
                            if ref_hist is None:
                                txt = f"Conf: {confs[best_idx]:.2f}/{current_min_confidence:.2f}"
                            cv2.putText(frame, txt, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

                    # draw target visuals always
                    radius_to_draw = int(DETECTION_RADIUS * margin_multiplier)
                    cv2.circle(frame, target_point, 5, (0, 0, 255), -1)
                    cv2.circle(frame, target_point, radius_to_draw, (0, 255, 255), 2)

            # Update trackers for active areas of this camera
            for area_idx in range(len(TARGET_AREAS[i])):  # len == 1 here
                if trackers[i][area_idx].update(current_detected[area_idx], current_object_counts[area_idx]):
                    terminal_update = True
                cam_status[i][area_idx] = trackers[i][area_idx].stable_status
                cam_counts[i][area_idx] = current_object_counts[area_idx]
            
            # Draw visualization for active areas (includes CAM label)
            draw_detection_info(frame, TARGET_AREAS[i], cam_status[i], cam_counts[i], i)
        
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

