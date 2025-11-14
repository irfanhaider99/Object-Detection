# roi_picker.py
import cv2
import json
import os

# ---- CONFIG ----
CAM_VIDEOS = {
    "Cam1": "cam1_recording.mp4",  # change to your files
    "Cam2": "cam2_recording.mp4"
}
OUTPUT_JSON = "zones.json"

def pick_rois_for_video(win_name, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Grab a frame at some time you like (frame 0 here)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from: {video_path}")

    # Let user select multiple ROIs on the frame
    # Keyboard:
    #  - ENTER/SPACE to confirm each ROI
    #  - C to cancel current ROI
    #  - ESC when done selecting (closes window)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    rois = cv2.selectROIs(win_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_name)
    cap.release()

    # Convert to list of (x1, y1, x2, y2)
    zones = []
    for (x, y, w, h) in rois:
        if w > 0 and h > 0:
            zones.append((int(x), int(y), int(x + w), int(y + h)))
    return zones, frame.shape[1], frame.shape[0]  # width, height

def main():
    all_zones = {}
    meta = {}

    for cam_label, path in CAM_VIDEOS.items():
        print(f"\nSelecting ROIs for {cam_label} from {path}")
        zones, w, h = pick_rois_for_video(f"{cam_label} - pick ROIs", path)
        print(f"Selected {len(zones)} zone(s) for {cam_label}: {zones}")
        all_zones[cam_label] = [{"name": f"Zone{i+1}", "coords": z, "target_class": None} for i, z in enumerate(zones)]
        meta[cam_label] = {"width": w, "height": h, "video": os.path.basename(path)}

    # Save to JSON
    data = {"zones": all_zones, "meta": meta}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved zones to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

