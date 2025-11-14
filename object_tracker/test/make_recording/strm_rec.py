from pypylon import pylon
import cv2


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
        # Set pixel format to BGR8
        if 'BGR8' in camera.PixelFormat.Symbolics:
            camera.PixelFormat.SetValue('BGR8')
            print("Pixel format set to BGR8.")
        else:
            print("BGR8 pixel format not supported by this camera.")

        # Set resolution
        camera.Width.Value = 1900
        camera.Height.Value = 1200
        print(f"Resolution set to {camera.Width.Value}x{camera.Height.Value}")

        # Set packet size in transport layer
        if camera.GetTLNodeMap().GetNode("GevSCPSPacketSize").IsWritable():
            camera.GetTLNodeMap().GetNode("GevSCPSPacketSize").SetValue(9000)
            print("Packet size set to 900 in transport layer.")
        else:
            print("Warning: Could not set packet size in transport layer.")

        # Set acquisition frame rate
        if camera.GetNodeMap().GetNode("AcquisitionFrameRateEnable").IsWritable():
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(fps)
            print(f"Frame rate set to {fps} FPS.")
        else:
            print("Warning: Could not set FPS.")

        # Set acquisition mode to continuous
        camera.AcquisitionMode.SetValue('Continuous')
        print("Acquisition mode set to Continuous.")

        # Set exposure time (ensure it's not less than 40000 µs)
        min_exposure_time = 40000  # in microseconds
        if camera.ExposureTime.Min <= min_exposure_time <= camera.ExposureTime.Max:
            camera.ExposureTime.SetValue(min_exposure_time)
            print(f"Exposure time set to {min_exposure_time} µs.")
        else:
            print(f"Warning: Desired exposure time of {min_exposure_time} µs is out of range.")

        # Enable auto exposure
        if camera.ExposureAuto.Symbolics:
            camera.ExposureAuto.SetValue('Continuous')
            print("Auto exposure enabled.")
        else:
            print("Auto exposure not supported by this camera.")

        # Enable auto gain
        if camera.GainAuto.Symbolics:
            camera.GainAuto.SetValue('Continuous')
            print("Auto gain enabled.")
        else:
            print("Auto gain not supported by this camera.")

        # Enable auto white balance
        if camera.BalanceWhiteAuto.Symbolics:
            camera.BalanceWhiteAuto.SetValue('Continuous')
            print("Auto white balance enabled.")
        else:
            print("Auto white balance not supported by this camera.")

        # Adjust brightness
        if 'BslBrightness' in camera.GetNodeMap().GetNodeNames():
            camera.BslBrightness.SetValue(0.2)  # Adjust value as needed (-1 to 1)
            print("Brightness adjusted.")
        else:
            print("Brightness adjustment not supported by this camera.")

        # Adjust contrast
        if 'BslContrast' in camera.GetNodeMap().GetNodeNames():
            camera.BslContrast.SetValue(0.2)  # Adjust value as needed (-1 to 1)
            print("Contrast adjusted.")
        else:
            print("Contrast adjustment not supported by this camera.")

    except Exception as e:
        print(f"Configuration error: {e}")



def stream_and_record_camera(camera, record=False, output_filename='output.mp4', fps=30):
    """
    Streams video from the specified Basler camera and records if specified.
    """
    try:
        camera.Open()
        serial_number = camera.GetDeviceInfo().GetSerialNumber()
        print(f"\nStreaming from Camera Serial Number: {serial_number}")

        configure_camera(camera, fps=fps)

        # Setup window for streaming
        cv2.namedWindow(f"Camera Feed - {serial_number}", cv2.WINDOW_NORMAL)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # Video writer setup (if recording)
        video_writer = None
        if record:
            print(f"Recording enabled. Video will be saved as {output_filename}")

        print("Press 'Esc' to stop streaming and recording...\n")

        while camera.IsGrabbing() and cv2.waitKey(1) != 27:
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                img = grab_result.Array
                height, width = img.shape[:2]

                # Initialize VideoWriter when recording is enabled
                if record and video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), True)
                    if video_writer.isOpened():
                        print(f"VideoWriter initialized: {width}x{height}, {fps} FPS.")
                    else:
                        print("Error: Failed to initialize VideoWriter. Recording disabled.")
                        record = False

                # Show streaming window
                cv2.imshow(f"Camera Feed - {serial_number}", img)

                # Write frame if recording
                if record and video_writer is not None:
                    video_writer.write(img)

            grab_result.Release()

    finally:
        # Stop and release resources
        camera.StopGrabbing()
        if video_writer is not None:
            video_writer.release()
            print(f"Recording saved as {output_filename}")
        cv2.destroyAllWindows()
        camera.Close()
        print(f"Camera {serial_number} closed.\n")


def main():
    devices = list_available_cameras()
    if not devices:
        return

    try:
        # Camera selection by user
        choice = int(input(f"\nEnter the camera index (0-{len(devices) - 1}) you want to stream: "))
        if 0 <= choice < len(devices):
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[choice]))

            # Recording option
            record_option = input("Do you want to record the video? (yes/no): ").strip().lower()
            if record_option == 'yes':
                filename = input("Enter the output video filename (with .mp4 extension): ").strip()
                if not filename.endswith(".mp4"):
                    filename += ".mp4"
                stream_and_record_camera(camera, record=True, output_filename=filename)
            else:
                stream_and_record_camera(camera, record=False)

        else:
            print("Invalid selection. Exiting.")

    except ValueError:
        print("Invalid input. Please enter a valid integer index.")


if __name__ == "__main__":
    main()

