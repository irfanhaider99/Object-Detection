import cv2
import numpy as np
import time
from ultralytics import YOLO
from pypylon import pylon
from scipy.spatial.transform import Rotation as R

# === Load Models ===
pose_model = YOLO("yolo11x-pose.pt").to("cuda")
object_model = YOLO("best_5.pt").to("cuda")

# === Load Calibration Parameters ===
K = np.load('/home/ihaider/object_tracking/integration/K.npy').squeeze()
K_pix = np.load('/home/ihaider/object_tracking/integration/K_pix.npy').squeeze()
D = np.load('/home/ihaider/object_tracking/integration/D.npy').squeeze()
position = np.load('/home/ihaider/object_tracking/integration/position.npy').squeeze()
R_data = np.load('/home/ihaider/object_tracking/integration/R.npy').squeeze()
scale_factor = 1.6914893617
pixels_size = 3.45e-6
x, y = 0.200342, 0.136332
off_x_pix = x / pixels_size
off_y_pix = y / pixels_size

R1, R2 = R_data[0], R_data[1]
K1_pix, K2_pix = K_pix[0], K_pix[1]
D1, D2 = D[0], D[1]
T1, T2 = position[0], position[1]
T1_pix = T1/pixels_size
T2_pix = T2/pixels_size
C1_pix = (T1_pix + np.array([off_x_pix,off_y_pix,0])) * scale_factor
C2_pix = (T2_pix + np.array([off_x_pix,off_y_pix,0])) * scale_factor
P1 = K1_pix @ np.hstack((R1.T, (-R1.T @ C1_pix).reshape(3, 1)))
P2 = K2_pix @ np.hstack((R2.T, (-R2.T @ C2_pix).reshape(3, 1)))

# === Helper Functions ===
def setup_cams():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
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

def draw_skeleton(image, kpts, skeleton):
    for a, b in skeleton:
        if all(kpts[a]) and all(kpts[b]):
            x1, y1 = map(int, kpts[a])
            x2, y2 = map(int, kpts[b])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (x1, y1), 3, (0, 0, 255), -1)
            cv2.circle(image, (x2, y2), 3, (0, 0, 255), -1)

def triangulate(kptL, kptR):
    if kptL.shape[0] != 17 or kptR.shape[0] != 17:
        return None
    pts1 = cv2.undistortPoints(kptL.reshape(-1, 1, 2), K1_pix, D1, P=K1_pix).reshape(-1, 2).T
    pts2 = cv2.undistortPoints(kptR.reshape(-1, 1, 2), K2_pix, D2, P=K2_pix).reshape(-1, 2).T
    pts_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    return (pts_4d[:3] / pts_4d[3]).T * pixels_size

def has_moved(tracker, key, center):
    prev = tracker.get(key)
    if prev is None:
        tracker[key] = center
        return True
    dist = np.linalg.norm(np.array(center) - np.array(prev))
    if dist > 5:
        tracker[key] = center
        return True
    return False

def get_color(cls_id):
    np.random.seed(cls_id)
    return tuple(int(c) for c in np.random.randint(100, 255, size=3))

# === Main ===
def main():
    cams = setup_cams()
    tracker = {}
    skeleton = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16],[11,12]]
    print("Starting integrated detection...")

    while True:
        frames = [grab(cam) for cam in cams]
        if any(f is None for f in frames):
            print("Failed to grab frame.")
            break

        pose_results = [pose_model(f, verbose=False)[0] for f in frames]
        obj_results = [object_model.track(f, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0] for f in frames]

        for i, (frame, pose_res, obj_res) in enumerate(zip(frames, pose_results, obj_results)):
            cam_label = f"Cam{i+1}"
            if pose_res.keypoints:
                for kpt in pose_res.keypoints.xy:
                    kpt_np = kpt.cpu().numpy()
                    if kpt_np.shape[0] < 17:
                        continue
                    draw_skeleton(frame, kpt_np, skeleton)

            if obj_res.boxes:
                if obj_res.boxes.id is None or obj_res.boxes.id.numel() == 0:
                    ids = np.arange(len(obj_res.boxes.xyxy))
                else:
                    ids = obj_res.boxes.id.cpu().numpy().astype(int)

                boxes = obj_res.boxes.xyxy.cpu().numpy()
                clss = obj_res.boxes.cls.cpu().numpy().astype(int)

                for box, cls, obj_id in zip(boxes, clss, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if has_moved(tracker, f"{cam_label}_{obj_id}", (cx, cy)):
                        color = get_color(cls)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{object_model.names[cls]} ID:{obj_id}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Camera 1", cv2.resize(frames[0], (800, 600)))
        cv2.imshow("Camera 2", cv2.resize(frames[1], (800, 600)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for cam in cams:
        cam.StopGrabbing()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
