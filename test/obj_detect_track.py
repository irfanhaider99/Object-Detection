import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon
import time
from collections import deque

# === Load your trained object detector and pose estimator ===
object_model = YOLO("best_5.pt")
pose_model = YOLO("yolo11x-pose.pt")

object_model.to("cuda")
pose_model.to("cuda")

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

# === Real-time Loop ===
out1 = cv2.VideoWriter("cam1_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (800, 600))  #optional
out2 = cv2.VideoWriter("cam2_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (800, 600))  #optional


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


print("Real-time Object + Wrist Motion Detection Started (press 'q' to quit)")

while True:
    frames = [grab(cam) for cam in cams]
    if any(f is None for f in frames):
        print("Frame capture failed.")
        break

    obj_results = [object_model.track(source=f, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0] for f in frames]
    pose_results = [pose_model(f, verbose=False)[0] for f in frames]

    for i, (obj_res, pose_res, frame) in enumerate(zip(obj_results, pose_results, frames)):
        cam_label = f"Cam{i+1}"

        # === Object Detection and Tracking ===
        if obj_res.boxes and obj_res.boxes.xyxy is not None:
            ids = obj_res.boxes.id
            boxes = obj_res.boxes.xyxy.cpu().numpy()
            clss = obj_res.boxes.cls.cpu().numpy().astype(int)
            if ids is None:
                ids = np.arange(len(boxes))
            else:
                ids = ids.cpu().numpy().astype(int)

            for j in range(len(ids)):
                x1, y1, x2, y2 = boxes[j]
                label = f"{object_model.names[clss[j]]} ID:{ids[j]}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

    # === Display ===
    cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
    if len(frames) > 1:
        cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup

out1.release()
out2.release()

for cam in cams:
    cam.StopGrabbing()
    cam.Close()
cv2.destroyAllWindows()
