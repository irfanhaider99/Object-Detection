import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon

# ==== Configuration ====
MODEL_PATH = "best_5.pt"
CROP_SIZE = 200  # size of crop around projected point
TARGET_CLASS_NAME = "obj00"  # class to detect

# ==== Load model ====
model = YOLO(MODEL_PATH)

# Load calibration parameters
K = np.load("K.npy")
R_all = np.load("R.npy")          # (2, 3, 3)
T_all = np.load("position.npy")   # (2, 3)
point_3d = np.load("target_3d_point.npy").reshape(3,)

# Extract left/right camera extrinsics
R_left = np.eye(3)
T_left = np.zeros(3)
R_right = R_all[1]
T_right = T_all[1]

# ==== Project 3D point to 2D ====
def project_point(K, R, T, point_3d):
    point_3d = point_3d.reshape(3, 1) if point_3d.ndim == 1 else point_3d  # (3,1)
    R = R.reshape(3, 3)
    T = T.reshape(3, 1)

    pt_cam = R @ point_3d + T  # shape (3,1)
    if pt_cam[2, 0] <= 0:
        return None

    pt_img = K @ pt_cam  # shape (3,1)
    pt_img = (pt_img[:2] / pt_cam[2, 0]).flatten()
    return pt_img.astype(int)

# ==== Crop around 2D point ====
def crop_around(img, cx, cy, size):
    h, w = img.shape[:2]
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)
    return img[y1:y2, x1:x2], (x1, y1)

# ==== Draw bounding box ====
def draw_box(img, top_left, found, size=200):
    color = (0, 255, 0) if found else (0, 0, 255)
    x, y = top_left
    cv2.rectangle(img, (x, y), (x + size, y + size), color, 2)

# ==== Setup Basler cameras ====
SERIAL_LEFT = "40312164"
SERIAL_RIGHT = "40312157"

def setup_cameras():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    cam_map = {d.GetSerialNumber(): d for d in devices}
    camL = pylon.InstantCamera(factory.CreateDevice(cam_map[SERIAL_LEFT]))
    camR = pylon.InstantCamera(factory.CreateDevice(cam_map[SERIAL_RIGHT]))
    for cam in [camL, camR]:
        cam.Open()
        cam.PixelFormat.Value = "BGR8"
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camL, camR

# ==== Start ====
camL, camR = setup_cameras()
print("Started real-time detection at target 3D location (press 'q' to quit)")

while True:
    resL = camL.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    resR = camR.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    frameL = resL.Array.copy()
    frameR = resR.Array.copy()
    resL.Release()
    resR.Release()

    pt2dL = project_point(K, R_left, T_left, point_3d)
    pt2dR = project_point(K, R_right, T_right, point_3d)  

    foundL = False
    foundR = False

    if pt2dL is not None:
        cropL, topL = crop_around(frameL, pt2dL, CROP_SIZE)
        resultL = model(cropL, verbose=False)[0]
        if resultL.boxes:
            for box in resultL.boxes:
                cls = int(box.cls[0].item())
                name = model.names[cls]
                if name == TARGET_CLASS_NAME:
                    foundL = True
        draw_box(frameL, topL, foundL, CROP_SIZE)

    if pt2dR is not None:
        cropR, topR = crop_around(frameR, *pt2dR, CROP_SIZE)
        resultR = model(cropR, verbose=False)[0]
        if resultR.boxes:
            for box in resultR.boxes:
                cls = int(box.cls[0].item())
                name = model.names[cls]
                if name == TARGET_CLASS_NAME:
                    foundR = True
        draw_box(frameR, topR, foundR, CROP_SIZE)

    # Show output
    cv2.imshow("Left Camera", frameL)
    cv2.imshow("Right Camera", frameR)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
camL.Close()
camR.Close()

