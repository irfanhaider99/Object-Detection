import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
import roslibpy
import time
from collections import deque

# === ROSLIBPY Setup ===
ROS_HOST = 'localhost'  # or your ROS bridge server IP
ROS_PORT = 9090
ros_client = roslibpy.Ros(host=ROS_HOST, port=ROS_PORT)
ros_client.run()

# Wait for connection
while not ros_client.is_connected:
    print("Waiting for ROS connection...")
    time.sleep(1)
print("Connected to ROS bridge!")

# Create publishers
pub_cam1 = roslibpy.Topic(ros_client, '/target_area/cam1/object_available', 'std_msgs/Bool')
pub_cam2 = roslibpy.Topic(ros_client, '/target_area/cam2/object_available', 'std_msgs/Bool')
pub_debug = roslibpy.Topic(ros_client, '/object_detection/debug', 'std_msgs/String')

def ros_log(message):
    """Helper function to log messages via ROSLIBPY"""
    print(message)
    pub_debug.publish(roslibpy.Message({'data': message}))

# === Load models ===
ros_log("Loading YOLO models...")
object_model = YOLO("best_5.pt")
pose_model = YOLO("yolo11x-pose.pt")
object_model.to("cuda")
pose_model.to("cuda")

# === Camera Setup ===
def setup_cams():
    ros_log("Initializing cameras...")
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

# === Fixed Target Areas ===
TARGET_AREA_CAM1 = [(808, 701), (908, 801)]  # Camera 1 target area
TARGET_AREA_CAM2 = [(355, 233), (455, 333)]  # Camera 2 target area

# === Visualization Parameters ===
TARGET_COLOR = (255, 255, 0)  # Cyan for target area
IN_TARGET_COLOR = (0, 255, 0)  # Green for objects in target
WRIST_COLOR = (0, 255, 255)    # Yellow for wrist
TARGET_THICKNESS = 2            # Thickness for target visualization

# === Main Loop ===
ros_log("Tracking objects in target areas. Press 'q' to quit")

def draw_target_info(frame, target_area, objects_in_target, cam_id):
    """Enhanced target information drawing with ROS status"""
    # Draw target area with different border based on availability
    border_color = IN_TARGET_COLOR if objects_in_target > 0 else (0, 0, 255)
    cv2.rectangle(frame, target_area[0], target_area[1], border_color, TARGET_THICKNESS + 2)
    
    # Add semi-transparent background for text
    text_bg = (target_area[0][0], target_area[0][1] - 40, target_area[1][0], target_area[0][1])
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_bg[0], text_bg[1]), (text_bg[2], text_bg[3]), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw target info
    status = "AVAILABLE" if objects_in_target > 0 else "NOT AVAILABLE"
    color = IN_TARGET_COLOR if objects_in_target > 0 else (0, 0, 255)
    
    # Publish to ROS via roslibpy
    msg = roslibpy.Message({'data': objects_in_target > 0})
    if cam_id == 0:
        pub_cam1.publish(msg)
    else:
        pub_cam2.publish(msg)
    
    cv2.putText(frame, f"STATUS: {status}", (target_area[0][0] + 10, target_area[0][1] - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"OBJECTS: {objects_in_target}", (target_area[0][0] + 10, target_area[0][1] - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"CAM {cam_id+1}", (target_area[0][0], target_area[1][1] + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

try:
    last_status_cam1 = None
    last_status_cam2 = None
    
    while ros_client.is_connected:
        frames = [grab(cam) for cam in cams]
        if any(f is None for f in frames):
            ros_log("Frame capture failed.")
            time.sleep(0.1)
            continue

        # Process frames
        obj_results = [object_model.track(source=f, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0] for f in frames]
        pose_results = [pose_model(f, verbose=False)[0] for f in frames]

        # Process each camera view
        current_status_cam1 = False
        current_status_cam2 = False
        
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
                    
                    # Check if object is in target area
                    in_target = (target_area[0][0] <= center_x <= target_area[1][0]) and \
                               (target_area[0][1] <= center_y <= target_area[1][1])
                    
                    if in_target:
                        objects_in_target += 1
                        # Draw bounding box and center dot
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), IN_TARGET_COLOR, 2)
                        cv2.circle(frame, (int(center_x), int(center_y)), 8, WRIST_COLOR, -1)
                        cv2.putText(frame, f"{object_model.names[clss[j]]} ID:{ids[j]}", 
                                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, IN_TARGET_COLOR, 2)

            # Update status and publish if changed
            if i == 0:
                current_status_cam1 = objects_in_target > 0
                if last_status_cam1 != current_status_cam1:
                    draw_target_info(frame, target_area, objects_in_target, i)
                    last_status_cam1 = current_status_cam1
            else:
                current_status_cam2 = objects_in_target > 0
                if last_status_cam2 != current_status_cam2:
                    draw_target_info(frame, target_area, objects_in_target, i)
                    last_status_cam2 = current_status_cam2

            # Pose estimation visualization (wrist)
            if pose_res.keypoints:
                for kpts in pose_res.keypoints.xy:
                    kpts = kpts.cpu().numpy()
                    if kpts.shape[0] >= 11:
                        # Draw right wrist (index 10)
                        rwrist = tuple(map(int, kpts[10]))
                        cv2.circle(frame, rwrist, 8, WRIST_COLOR, -1)
                        cv2.putText(frame, "Wrist", (rwrist[0] + 10, rwrist[1]),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, WRIST_COLOR, 2)

        # Display
        cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
        if len(frames) > 1:
            cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            ros_log("Shutting down by user request...")
            break

except Exception as e:
    ros_log(f"Error in main loop: {str(e)}")
finally:
    # Cleanup
    for cam in cams:
        cam.StopGrabbing()
        cam.Close()
    cv2.destroyAllWindows()
    ros_client.terminate()
    ros_log("Node shutdown complete")
