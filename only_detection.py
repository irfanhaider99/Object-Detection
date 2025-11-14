import cv2, numpy as np
from ultralytics import YOLO
from pypylon import pylon, genicam

MODEL_PATH = "best_5.pt"   # change if needed

model = YOLO(MODEL_PATH)
try: model.to("cuda")
except: pass

# --- open first two Baslers ---
f = pylon.TlFactory.GetInstance()
devs = f.EnumerateDevices()
if not devs: raise SystemExit("No Basler cameras found.")
cams = [pylon.InstantCamera(f.CreateDevice(d)) for d in devs[:2]]
for cam in cams:
    cam.Open()
    try: cam.AcquisitionMode.SetValue("Continuous")
    except: pass
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

def grab(cam):
    try:
        r = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
    except genicam.RuntimeException:
        return None
    if not r.GrabSucceeded():
        r.Release(); return None
    img = r.Array; r.Release()
    if img.ndim == 2: frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return frame

try:
    while True:
        frames = [grab(c) for c in cams]
        for i, frame in enumerate(frames):
            if frame is None: continue
            res = model(frame, conf=0.25, verbose=False)[0]
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                clss  = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros(len(boxes), int)
                confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones(len(boxes))
                for (x1,y1,x2,y2), c, p in zip(boxes, clss, confs):
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                    name = model.names.get(int(c), str(int(c))) if hasattr(model, "names") else str(int(c))
                    cv2.putText(frame, f"{name} {p:.2f}", (int(x1), max(0,int(y1)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow(f"Camera {i+1}", cv2.resize(frame, (800, 600)))
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
finally:
    for cam in cams:
        try:
            if cam.IsGrabbing(): cam.StopGrabbing()
            if cam.IsOpen(): cam.Close()
        except: pass
    cv2.destroyAllWindows()

