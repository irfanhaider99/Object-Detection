import cv2
import numpy as np
from ultralytics import YOLO

# === Load Models ===
object_model = YOLO("best_5.pt").to("cuda")

# === Load Calibration ===
K_pix = np.load("K_pix.npy")     # shape (2, 3, 3)
R = np.load("R.npy")             # shape (2, 3, 3)
T = np.load("position.npy")      # shape (2, 3)

K_left, K_right = K_pix[0], K_pix[1]
R_left, R_right = R[0], R[1]
T_left, T_right = T[0].reshape(3, 1), T[1].reshape(3, 1)

P_left = K_left @ np.hstack((R_left, T_left))
P_right = K_right @ np.hstack((R_right, T_right))

# === Your Own Frame Inputs Here ===
# Replace with actual images from Basler cameras
frameL = cv2.imread("left_cam.png")     # Replace with live or saved image
frameR = cv2.imread("right_cam.png")

def get_center_of_obj00(result, model):
    if result.boxes:
        clss = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        for cls, box in zip(clss, boxes):
            if model.names[cls] == "obj00":
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                return np.array([[cx], [cy]])
    return None

# === Run YOLO detection ===
resultL = object_model(frameL, verbose=False)[0]
resultR = object_model(frameR, verbose=False)[0]

ptL = get_center_of_obj00(resultL, object_model)
ptR = get_center_of_obj00(resultR, object_model)

if ptL is not None and ptR is not None:
    # === Triangulate ===
    point_4d = cv2.triangulatePoints(P_left, P_right, ptL, ptR)
    point_3d = (point_4d[:3] / point_4d[3]).reshape(-1)  # X, Y, Z

    # === Save ===
    np.save("target_3d_point.npy", point_3d)
    print("Saved 3D point:", point_3d)
else:
    print("Object not detected in both views.")

