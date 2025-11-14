from pypylon import pylon
import cv2
from multiprocessing import Process


def list_available_cameras():
    """
    Lists all connected Basler cameras with their serial numbers.
    """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    if not devices:
        print("No Basler cameras found.")
        return []

    print(f"\nFound {len(devices)} camera(s):")
    for i, device in enumerate(devices):
        print(f"[{i}] Model: {device.GetModelName()}, Serial Number: {device.GetSerialNumber()}")

    return devices


def configure_camera(camera, fps=30):
    """
    Configures the camera's settings for optimal image quality.
    """
    try:
        # Pixel Format
        if camera.GetNodeMap().GetNode("PixelFormat") is not None:
            if 'BGR8' in camera.PixelFormat.Symbolics:
                camera.PixelFormat.SetValue('BGR8')
                print("Pixel format set to BGR8.")
            else:
                print("BGR8 pixel format not supported by this camera.")

        # Resolution
        camera.Width.Value = 1900
        camera.Height.Value = 1200
        print(f"Resolution set to {camera.Width.Value}x{camera.Height.Value}")

        # Packet Size
        packet_size_node = camera.GetTLNodeMap().GetNode("GevSCPSPacketSize")
        if packet_size_node and packet_size_node.IsWritable():
            packet_size_node.SetValue(9000)
            print("Packet size set to 9000.")
        else:
            print("Packet size node not available or not writable.")

        # Frame Rate
        frame_rate_node = camera.GetNodeMap().GetNode("AcquisitionFrameRate")
        if frame_rate_node and frame_rate_node.IsWritable():
            camera.AcquisitionFrameRateEnable.SetValue(True)
            frame_rate_node.SetValue(fps)
            print(f"Frame rate set to {fps} FPS.")
        else:
            print("Frame rate node not available or not writable.")

        # Exposure Time
        min_exposure_time = 30000
        exposure_time_node = camera.GetNodeMap().GetNode("ExposureTime")
        if exposure_time_node and exposure_time_node.IsWritable():
            camera.ExposureTime.SetValue(min_exposure_time)
            print(f"Exposure time set to {min_exposure_time} µs.")
        else:
            print("Exposure time node not available or not writable.")

    except Exception as e:
        print(f"Configuration error: {e}")


def stream_and_record_camera(index, filename, fps=30):
    """
    Streams and records video from the specified Basler camera.
    """
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if index >= len(devices):
        print(f"Camera index {index} out of range.")
        return

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[index]))
    try:
        camera.Open()
        serial_number = camera.GetDeviceInfo().GetSerialNumber()
        print(f"\nStreaming & recording from Camera Serial Number: {serial_number}")

        configure_camera(camera, fps=fps)
        cv2.namedWindow(f"Camera Feed - {serial_number}", cv2.WINDOW_NORMAL)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # VideoWriter setup
        video_writer = None
        print(f"Recording will be saved as {filename}")

        while camera.IsGrabbing():
            if cv2.waitKey(1) == 27:  # ESC key pressed
                print(f"Stopping streaming for Camera {serial_number}.")
                break

            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                img = grab_result.Array
                height, width = img.shape[:2]

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), True)
                    print(f"VideoWriter initialized for {filename}: {width}x{height}, {fps} FPS.")

                cv2.imshow(f"Camera Feed - {serial_number}", img)
                video_writer.write(img)
            grab_result.Release()

    finally:
        camera.StopGrabbing()
        if video_writer:
            video_writer.release()
            print(f"Recording saved as {filename}")
        cv2.destroyAllWindows()
        camera.Close()
        print(f"Camera {serial_number} closed.\n")


def main():
    devices = list_available_cameras()
    if len(devices) < 2:
        print("Need at least 2 cameras for simultaneous streaming and recording.")
        return

    # Filenames for both cameras
    filename_cam1 = input("Enter output filename for Camera 1 (with .mp4 extension): ").strip()
    filename_cam2 = input("Enter output filename for Camera 2 (with .mp4 extension): ").strip()

    if not filename_cam1.endswith(".mp4"):
        filename_cam1 += ".mp4"
    if not filename_cam2.endswith(".mp4"):
        filename_cam2 += ".mp4"

    # Processes for each camera
    p1 = Process(target=stream_and_record_camera, args=(0, filename_cam1))
    p2 = Process(target=stream_and_record_camera, args=(1, filename_cam2))

    print("\nStarting simultaneous streaming and recording for both cameras...\n")
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("\nSimultaneous streaming and recording completed successfully.")


if __name__ == "__main__":
    main()

