import cv2
import numpy as np
import time
from ultralytics import YOLO
from pypylon import pylon

# === Setup Basler Cameras ===
def setup_cams():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) < 2:
        raise RuntimeError("Less than 2 Basler cameras found.")
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
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    result.Release()
    return None

# === Tracking + Visual Enhancements ===
previous_positions = {}
object_trails = {}
MOTION_THRESHOLD = 20

def has_moved(cam_label, obj_id, center):
    key = f"{cam_label}_{obj_id}"
    prev = previous_positions.get(key)
    if prev:
        dist = np.linalg.norm(np.array(center) - np.array(prev))
        if dist > MOTION_THRESHOLD:
            previous_positions[key] = center
            return True
        return False
    else:
        previous_positions[key] = center
        return True

def id_to_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(100, 255, 3).tolist())

def update_trails(key, center, max_len=15):
    if key not in object_trails:
        object_trails[key] = []
    object_trails[key].append(center)
    if len(object_trails[key]) > max_len:
        object_trails[key] = object_trails[key][-max_len:]

def draw_trails(frame, key):
    if key not in object_trails or len(object_trails[key]) < 2:
        return
    trail = object_trails[key]
    for i in range(1, len(trail)):
        alpha = int(255 * (1 - i / len(trail)))
        color = (0, alpha, 255)
        cv2.line(frame,
                 (int(trail[i - 1][0]), int(trail[i - 1][1])),
                 (int(trail[i][0]), int(trail[i][1])),
                 color, 2)

def draw_status_bar(frame, cam_id, fps, num_objs):
    bar_color = (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), bar_color, -1)
    cv2.putText(frame, f"{cam_id} | \u23F0 {time.strftime('%H:%M:%S')} | \U0001F9CD {num_objs} | \U0001F680 FPS: {fps:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

def draw_sidebar(frame, class_counts):
    x0, y0 = frame.shape[1] - 180, 40
    width = 180
    height = 25 * len(class_counts) + 30
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (30, 30, 30), -1)
    y = y0 + 20
    for cls, count in class_counts.items():
        cv2.putText(frame, f"{cls}: {count}", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25

# === Load YOLOv11 Model ===
model = YOLO("yolo11n.pt")
model.to("cuda")

cams = setup_cams()
fps_timer = time.time()
fps_counter = 0
fps_value = 0.0

while True:
    frames = [grab(cam) for cam in cams]
    if any(f is None for f in frames):
        print("Failed to grab frame from one or both cameras.")
        break

    results = [model.track(f, persist=True, verbose=False)[0] for f in frames]

    for i, (result, frame) in enumerate(zip(results, frames)):
        cam_label = f"Cam{i+1}"
        class_counts = {}

        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for j in range(len(ids)):
                x1, y1, x2, y2 = boxes[j]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                label = model.names[clss[j]]
                class_counts[label] = class_counts.get(label, 0) + 1

                if has_moved(cam_label, ids[j], (cx, cy)):
                    key = f"{cam_label}_{ids[j]}"
                    update_trails(key, (cx, cy))
                    color = id_to_color(ids[j])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} ID:{ids[j]}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    draw_trails(frame, key)

        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_timer > 1.0:
            fps_value = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_status_bar(frame, cam_label, fps_value, len(class_counts))
        draw_sidebar(frame, class_counts)

    display_size = (800, 600)
    cv2.imshow("Camera 1", cv2.resize(frames[0], display_size))
    cv2.imshow("Camera 2", cv2.resize(frames[1], display_size))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cam in cams:
    cam.StopGrabbing()
    cam.Close()
cv2.destroyAllWindows()

