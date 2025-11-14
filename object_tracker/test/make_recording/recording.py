import cv2
from pypylon import pylon
import threading
import time

# Output video settings
frame_rate = 30
frame_width = 1936
frame_height = 1216
record_seconds = 59  # adjust as needed

# Output video files
output1 = "basler_cam1.avi"
output2 = "basler_cam2.avi"

# FourCC codec for AVI
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Discover Basler devices
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if len(devices) < 2:
    raise RuntimeError("At least two Basler cameras are required.")

# Create camera objects
camera1 = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
camera2 = pylon.InstantCamera(tl_factory.CreateDevice(devices[1]))

# Open both cameras
camera1.Open()
camera2.Open()

# Set resolutions (optional — depends on your camera model)
camera1.Width.SetValue(frame_width)
camera1.Height.SetValue(frame_height)
camera2.Width.SetValue(frame_width)
camera2.Height.SetValue(frame_height)

# Set pixel format to RGB8 if supported (avoids Bayer conversion)
try:
    camera1.PixelFormat.SetValue("RGB8")
    camera2.PixelFormat.SetValue("RGB8")
except Exception as e:
    print(f"Warning: Could not set RGB8 pixel format: {e}")

# Set acquisition mode
camera1.AcquisitionMode.SetValue("Continuous")
camera2.AcquisitionMode.SetValue("Continuous")

# Create video writers
writer1 = cv2.VideoWriter(output1, fourcc, frame_rate, (frame_width, frame_height))
writer2 = cv2.VideoWriter(output2, fourcc, frame_rate, (frame_width, frame_height))

# Frame grab loop
def record_camera(camera, writer, cam_name):
    camera.StartGrabbing()
    start_time = time.time()
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            img = grab_result.Array

            # Check image shape before converting
            if len(img.shape) == 2:  # Single-channel Bayer
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
                except cv2.error as e:
                    print(f"[{cam_name}] cvtColor error: {e}")
                    grab_result.Release()
                    continue
            elif len(img.shape) == 3 and img.shape[2] == 3:  # Already in BGR/RGB format
                img_bgr = img
            else:
                print(f"[{cam_name}] Unexpected image shape: {img.shape}")
                grab_result.Release()
                continue

            writer.write(img_bgr)
        grab_result.Release()

        if time.time() - start_time > record_seconds:
            break

    camera.StopGrabbing()
    writer.release()
    print(f"[{cam_name}] Recording complete.")

# Start recording threads
thread1 = threading.Thread(target=record_camera, args=(camera1, writer1, "Camera 1"))
thread2 = threading.Thread(target=record_camera, args=(camera2, writer2, "Camera 2"))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

# Close cameras
camera1.Close()
camera2.Close()

print("Both camera recordings saved.")

