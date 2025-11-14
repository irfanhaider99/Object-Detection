from pypylon import pylon
import cv2
import os

# Create a directory for saving videos
save_directory = "recorded_videos"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Define video filename
video_filename = os.path.join(save_directory, "recorded_video11.mp4")

# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Adjust camera width (optional)
new_width = camera.Width.Value - camera.Width.Inc
if new_width >= camera.Width.Min:
    camera.Width.Value = new_width

# Start grabbing images
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Attempt to grab the first frame to determine size
frame = None
for _ in range(50):  # Try up to 50 times to get a valid frame
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        frame = grabResult.Array
        break
    grabResult.Release()

# Check if a valid frame was retrieved
if frame is None:
    print("Error: Unable to retrieve frame size. Exiting.")
    camera.Close()
    exit()

# Get frame dimensions
height, width = frame.shape[:2]

# Define video writer (MP4 format)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27:  # Press 'Esc' to stop recording
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        img = grabResult.Array

        # Convert Bayer to RGB if needed
        if len(img.shape) == 2:  # If it's grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Show streaming window
        cv2.imshow("Camera Feed", img)

        # Write frame to video file
        video_writer.write(img)

    grabResult.Release()

# Cleanup
cv2.destroyAllWindows()
video_writer.release()  # Save the video file
camera.Close()

print(f"Video saved as {video_filename}")

