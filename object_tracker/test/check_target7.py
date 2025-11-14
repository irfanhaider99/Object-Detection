import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
from collections import deque

# === Configuration ===
STABILIZATION_FRAMES = 5  # Number of frames to consider for status stabilization
MIN_DETECTION_CONFIDENCE = 0.3  # Minimum confidence to consider a detection valid

# === Load models ===
print("Loading YOLO models...")
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
        raise RuntimeError(f"Expected at least 2 cameras, found {len(devices)}")
    cams = [pylon.InstantCamera(factory.CreateDevice(dev)) for dev in devices[:2]]
    for cam in cams:
        cam.Open()
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return cams

def grab(cam):
    """Capture frame from camera"""
    result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if result.GrabSucceeded():
        img = result.Array
        result.Release()
        if img.ndim == 2:  # Grayscale image
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:  # Color image
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result.Release()
    return None

cams = setup_cams()

# === Target Areas ===
TARGET_AREA_CAM1 = [(808, 701), (908, 801)]
TARGET_AREA_CAM2 = [(355, 233), (455, 333)]

# === Visualization Parameters ===
COLORS = {
    'target': (255, 255, 0),
    'in_target': (0, 255, 0),
    'wrist': (0, 255, 255),
    'unavailable': (0, 0, 255)
}

# === Stabilization Buffers ===
class StatusBuffer:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
        self.last_stable_status = False
    
    def update(self, current_status):
        self.buffer.append(current_status)
        # Require consistent status in 60% of recent frames
        if sum(self.buffer) / len(self.buffer) > 0.6:
            self.last_stable_status = True
        elif sum(self.buffer) / len(self.buffer) < 0.4:
            self.last_stable_status = False
        return self.last_stable_status

cam1_buffer = StatusBuffer(STABILIZATION_FRAMES)
cam2_buffer = StatusBuffer(STABILIZATION_FRAMES)

def process_detections(frame, detections, target_area):
    objects_in_target = 0
    if detections.boxes and detections.boxes.xyxy is not None:
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        clss = detections.boxes.cls.cpu().numpy().astype(int)
        
        for j in range(len(boxes)):
            if confs[j] < MIN_DETECTION_CONFIDENCE:
                continue
                
            x1, y1, x2, y2 = boxes[j]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            if (target_area[0][0] <= center_x <= target_area[1][0] and
                target_area[0][1] <= center_y <= target_area[1][1]):
                objects_in_target += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            COLORS['in_target'], 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 8, 
                          COLORS['wrist'], -1)
    return objects_in_target > 0

def draw_status(frame, target_area, is_available, cam_id):
    color = COLORS['in_target'] if is_available else COLORS['unavailable']
    # Draw target area
    cv2.rectangle(frame, target_area[0], target_area[1], color, 3)
    # Draw status text
    status = "AVAILABLE" if is_available else "NOT AVAILABLE"
    cv2.putText(frame, f"Status: {status}", (target_area[0][0] + 10, target_area[0][1] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # Display camera ID
    cv2.putText(frame, f"Cam {cam_id+1}", (target_area[0][0], target_area[1][1] + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

try:
    print("Starting detection loop. Press 'q' to quit...")
    while True:
        frames = [grab(cam) for cam in cams]
        if any(f is None for f in frames):
            print("Warning: Failed to grab frame from one or more cameras")
            continue

        # Process detections
        obj_results = [object_model.track(f, persist=True, conf=0.25, verbose=False)[0] for f in frames]
        
        for i, (frame, obj_res) in enumerate(zip(frames, obj_results)):
            target_area = TARGET_AREA_CAM1 if i == 0 else TARGET_AREA_CAM2
            current_status = process_detections(frame, obj_res, target_area)
            
            # Apply stabilization
            stable_status = cam1_buffer.update(current_status) if i == 0 else cam2_buffer.update(current_status)
            draw_status(frame, target_area, stable_status, i)

        # Display
        cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
        if len(frames) > 1:
            cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"Error: {str(e)}")
finally:
    print("Cleaning up resources...")
    for cam in cams:
        cam.StopGrabbing()
        cam.Close()
    cv2.destroyAllWindows()
    print("Done.")
