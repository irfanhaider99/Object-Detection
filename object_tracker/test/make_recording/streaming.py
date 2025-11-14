from pypylon import pylon
import cv2
import os

# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Adjust camera width (optional)
new_width = camera.Width.Value - camera.Width.Inc
if new_width >= camera.Width.Min:
    camera.Width.Value = new_width

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

while cv2.waitKey(1) != 27:  # Press 'Esc' to exit
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Convert the image to OpenCV format
        img = grabResult.Array

        # Convert Bayer to RGB if needed (uncomment if images are grayscale)
        # img = cv2.cvtColor(img, cv2.COLOR_BG2RGB)

        # Show the camera feed
        cv2.imshow("Camera Feed", img)

    grabResult.Release()

# Cleanup
cv2.destroyAllWindows()
camera.Close()

