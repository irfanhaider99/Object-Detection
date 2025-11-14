import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon, genicam
import os
import time

# ========= Config =========
MODEL_PATH = "best_5.pt"
CONF_DET = 0.20  # Detection confidence threshold
DISPLAY_SIZE = (800, 600)

# Target locations
PRESET_PIX = [(1219, 656), (728, 383)]  # Updated with your actual targets
SAVE_FILE = "target_pix.npy"  # Saved target locations

# Detection parameters
ROI_HALF = 300  # Increased primary ROI size
ROI_HALF_FALL = 450  # Increased fallback ROI size
STRICT_RADIUS_PX = 100  # More lenient position requirement
FOCUS_SIDE = 200  # Larger focus area
MIN_CONF_GOOD = 0.25  # Lower confidence threshold
MIN_BOX_AREA = 500  # Minimum box area to consider valid (px²)

# Stability parameters
GREEN_NEED = 3  # Consecutive good frames needed to turn green
RED_NEED = 6  # Consecutive bad frames needed to turn red
POS_HISTORY_LEN = 5  # Number of positions to average

# Display settings
FALLBACK_SIDE = 40  # Default box size when no detection
SHOW_ROI_RECT = True  # Show ROI rectangles for debugging
SHOW_HUD = True  # Show detection info
DEBUG_MODE = True  # Enable debug prints and visualizations
# ==========================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def class_name_from_model(model, cls_idx):
    try:
        names = model.names
        if isinstance(names, dict):
            return names.get(int(cls_idx), str(int(cls_idx)))
        names = list(names)
        return names[int(cls_idx)] if 0 <= int(cls_idx) < len(names) else str(int(cls_idx))
    except Exception:
        return str(int(cls_idx))

def detect_in_roi(model, frame, roi, conf=0.12, imgsz=640):
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    
    # Add padding if ROI is too small
    if crop.size == 0 or (x2-x1) < 64 or (y2-y1) < 64:
        padding = 64
        x1 = max(0, x1-padding)
        y1 = max(0, y1-padding)
        x2 = min(frame.shape[1], x2+padding)
        y2 = min(frame.shape[0], y2+padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.empty((0, 4), float), np.array([], int), np.array([], float)
    
    r = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)[0]
    
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4), float), np.array([], int), np.array([], float)
    
    boxes = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    
    # Map coordinates back to original frame
    boxes[:, [0, 2]] += x1
    boxes[:, [1, 3]] += y1
    
    if DEBUG_MODE and len(boxes) > 0:
        print(f"Detected {len(boxes)} objects with confidences: {confs}")
    
    return boxes, clss, confs

def pick_nearest(boxes_xyxy, target_uv):
    if target_uv is None or boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None, None, None
    
    tu, tv = target_uv
    best_idx, best_d, best_c = None, 1e9, None
    
    for j, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        d = ((cx - tu) ** 2 + (cy - tv) ** 2) ** 0.5
        if d < best_d:
            best_idx, best_d, best_c = j, d, (cx, cy)
    
    return best_idx, best_d, best_c

def center_in_focus_area(cx, cy, tu, tv, side):
    half = side // 2
    return (tu - half) <= cx <= (tu + half) and (tv - half) <= cy <= (tv + half)

def tight_intersection(x1, y1, x2, y2, tu, tv, side, W, H):
    half = side // 2
    fx1 = clamp(int(tu) - half, 0, W - 1)
    fy1 = clamp(int(tv) - half, 0, H - 1)
    fx2 = clamp(int(tu) + half, 0, W - 1)
    fy2 = clamp(int(tv) + half, 0, H - 1)
    
    ix1 = max(int(x1), fx1)
    iy1 = max(int(y1), fy1)
    ix2 = min(int(x2), fx2)
    iy2 = min(int(y2), fy2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return fx1, fy1, fx2, fy2, True  # No overlap -> draw focus window
    return ix1, iy1, ix2, iy2, False

def maybe_load_targets(default_pairs):
    if os.path.exists(SAVE_FILE):
        try:
            arr = np.load(SAVE_FILE, allow_pickle=True)
            if len(arr) == 2 and all(a is None or (len(a) == 2) for a in arr):
                return [tuple(arr[0]) if arr[0] is not None else None,
                        tuple(arr[1]) if arr[1] is not None else None]
        except Exception as e:
            print("[WARN] Could not load targets:", e)
    return [tuple(default_pairs[0]), tuple(default_pairs[1])]

def is_high_quality_detection(box, conf):
    width = box[2] - box[0]
    height = box[3] - box[1]
    box_area = width * height
    
    if DEBUG_MODE:
        print(f"Box area: {box_area}, Confidence: {conf:.2f}")
    
    return box_area >= MIN_BOX_AREA and conf >= MIN_CONF_GOOD

# --- Initialize ---
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)
try:
    model.to("cuda")
    print("[INFO] Using GPU acceleration")
except:
    print("[INFO] Using CPU")

# Initialize cameras
print("[INFO] Initializing cameras...")
f = pylon.TlFactory.GetInstance()
devs = f.EnumerateDevices()
if len(devs) < 2:
    raise SystemExit("Need 2 Basler cameras.")

cams = [pylon.InstantCamera(f.CreateDevice(d)) for d in devs[:2]]
for i, cam in enumerate(cams, 1):
    cam.Open()
    try:
        cam.AcquisitionMode.SetValue("Continuous")
    except:
        pass
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:
        print(f"[INFO] Cam{i}: {devs[i-1].GetModelName()}  Serial={devs[i-1].GetSerialNumber()}")
    except Exception:
        pass

def grab(cam):
    try:
        r = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
    except genicam.RuntimeException:
        return None
    if not r.GrabSucceeded():
        r.Release()
        return None
    img = r.Array
    r.Release()
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Initialize state
TARGET_PIX = maybe_load_targets(PRESET_PIX)
OFFSETS = [(0, 0), (0, 0)]  # Per-camera offsets
good_cnt = [0, 0]  # Consecutive good frames
bad_cnt = [0, 0]  # Consecutive bad frames
is_green = [False, False]  # Current state
pos_history = [[] for _ in range(2)]  # Position history for smoothing

def apply_offset(pt, off):
    if pt is None:
        return None
    return (pt[0] + off[0], pt[1] + off[1])

print("[INFO] Using targets:", TARGET_PIX)
print("Keys: q=quit, s=save targets | Cam1 nudge: A/D(u-+), W/S(v-+) | Cam2 nudge: J/L(u-+), I/K(v-+)")

try:
    while True:
        frames = [grab(c) for c in cams]

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            H, W = frame.shape[:2]
            tu_tv = TARGET_PIX[i]
            tu_tv = apply_offset(tu_tv, OFFSETS[i])

            if tu_tv is None:
                cv2.putText(frame, "No target set", (15, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(f"Camera {i+1}", cv2.resize(frame, DISPLAY_SIZE))
                continue

            tu, tv = tu_tv

            # Draw target and ROIs for debugging
            cv2.circle(frame, (int(tu), int(tv)), 10, (0, 0, 255), 2)  # Red circle at target
            
            # Primary detection in ROI
            x1 = clamp(int(tu - ROI_HALF), 0, W - 1)
            y1 = clamp(int(tv - ROI_HALF), 0, H - 1)
            x2 = clamp(int(tu + ROI_HALF), 0, W - 1)
            y2 = clamp(int(tv + ROI_HALF), 0, H - 1)
            
            if SHOW_ROI_RECT:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue primary ROI
            
            boxes, clss, confs = detect_in_roi(model, frame, (x1, y1, x2, y2), conf=CONF_DET, imgsz=640)
            used_fallback = False

            # Fallback to wider ROI if primary is empty
            if len(boxes) == 0:
                xf1 = clamp(int(tu - ROI_HALF_FALL), 0, W - 1)
                yf1 = clamp(int(tv - ROI_HALF_FALL), 0, H - 1)
                xf2 = clamp(int(tu + ROI_HALF_FALL), 0, W - 1)
                yf2 = clamp(int(tv + ROI_HALF_FALL), 0, H - 1)
                
                if SHOW_ROI_RECT:
                    cv2.rectangle(frame, (xf1, yf1), (xf2, yf2), (0, 255, 255), 1)  # Yellow fallback ROI
                
                boxes, clss, confs = detect_in_roi(model, frame, (xf1, yf1, xf2, yf2), 
                                               conf=max(0.05, CONF_DET - 0.03), imgsz=800)
                used_fallback = True

            # Evaluate detections
            cv2.drawMarker(frame, (int(tu), int(tv)), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
            idx, min_d, cent = pick_nearest(boxes, (tu, tv))

            good_now = False
            if idx is not None and cent is not None and min_d is not None:
                cx, cy = cent
                box = boxes[idx]
                
                # Position smoothing
                pos_history[i].append((cx, cy))
                if len(pos_history[i]) > POS_HISTORY_LEN:
                    pos_history[i].pop(0)
                smoothed_pos = np.mean(pos_history[i], axis=0) if pos_history[i] else (cx, cy)
                
                # Quality checks
                if (min_d <= STRICT_RADIUS_PX and 
                    center_in_focus_area(smoothed_pos[0], smoothed_pos[1], tu, tv, FOCUS_SIDE) and
                    is_high_quality_detection(box, confs[idx])):
                    good_now = True
                    
                    if DEBUG_MODE:
                        print(f"Cam{i} Good detection at {smoothed_pos} (distance {min_d:.1f}px)")

            # Update hysteresis state
            if good_now:
                good_cnt[i] += 1
                bad_cnt[i] = 0
            else:
                bad_cnt[i] += 1
                good_cnt[i] = 0

            if not is_green[i] and good_cnt[i] >= GREEN_NEED:
                is_green[i] = True
                if DEBUG_MODE:
                    print(f"Cam{i} Switching to GREEN state")
            if is_green[i] and bad_cnt[i] >= RED_NEED:
                is_green[i] = False
                if DEBUG_MODE:
                    print(f"Cam{i} Switching to RED state")

            # Draw results
            if is_green[i] and idx is not None:
                bx1, by1, bx2, by2 = boxes[idx]
                ix1, iy1, ix2, iy2, _ = tight_intersection(bx1, by1, bx2, by2, tu, tv, FOCUS_SIDE, W, H)
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 220, 0), 2)
            else:
                half = FALLBACK_SIDE // 2
                rx1 = clamp(int(tu) - half, 0, W - 1)
                ry1 = clamp(int(tv) - half, 0, H - 1)
                rx2 = clamp(int(tu) + half, 0, W - 1)
                ry2 = clamp(int(tv) + half, 0, H - 1)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                if SHOW_HUD:
                    cv2.putText(frame, "NO OBJECT @ TARGET", (rx1, max(0, ry1 - 6)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if SHOW_HUD:
                hud = f"cand={len(boxes)}"
                if min_d is not None:
                    hud += f" d={min_d:.1f}px"
                hud += f"  G{good_cnt[i]}/R{bad_cnt[i]}  {'GREEN' if is_green[i] else 'RED'}"
                if used_fallback:
                    hud += " (fb)"
                cv2.putText(frame, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(f"Camera {i+1}", cv2.resize(frame, DISPLAY_SIZE))

        # Handle keyboard input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            try:
                np.save(SAVE_FILE, np.array(TARGET_PIX, dtype=object))
                print("[INFO] Saved targets:", TARGET_PIX)
            except Exception as e:
                print("[WARN] Could not save targets:", e)
        # Camera 1 nudges
        elif k == ord('a'): OFFSETS[0] = (OFFSETS[0][0] - 1, OFFSETS[0][1]); print("Cam1 u-1", OFFSETS[0])
        elif k == ord('d'): OFFSETS[0] = (OFFSETS[0][0] + 1, OFFSETS[0][1]); print("Cam1 u+1", OFFSETS[0])
        elif k == ord('w'): OFFSETS[0] = (OFFSETS[0][0], OFFSETS[0][1] - 1); print("Cam1 v-1", OFFSETS[0])
        elif k == ord('s'): OFFSETS[0] = (OFFSETS[0][0], OFFSETS[0][1] + 1); print("Cam1 v+1", OFFSETS[0])
        # Camera 2 nudges
        elif k == ord('j'): OFFSETS[1] = (OFFSETS[1][0] - 1, OFFSETS[1][1]); print("Cam2 u-1", OFFSETS[1])
        elif k == ord('l'): OFFSETS[1] = (OFFSETS[1][0] + 1, OFFSETS[1][1]); print("Cam2 u+1", OFFSETS[1])
        elif k == ord('i'): OFFSETS[1] = (OFFSETS[1][0], OFFSETS[1][1] - 1); print("Cam2 v-1", OFFSETS[1])
        elif k == ord('k'): OFFSETS[1] = (OFFSETS[1][0], OFFSETS[1][1] + 1); print("Cam2 v+1", OFFSETS[1])

finally:
    for cam in cams:
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
            if cam.IsOpen():
                cam.Close()
        except:
            pass
    cv2.destroyAllWindows()