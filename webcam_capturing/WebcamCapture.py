import cv2
import numpy as np
import torch
import time


class WebcamCapture:
    def __init__(self, frame_count=142, target_fps=30):
        """
        Initialize the WebcamCapture class.

        Parameters:
            frame_count (int): The number of frames to capture in each cycle.
            target_fps (int): The target frame rate for capturing video.
            save_frames (bool): Whether to save the first, middle, and last frames as images.
        """
        self.frame_count = frame_count
        self.target_fps = target_fps
        self.frame_interval = 1 / target_fps

    def _process_and_save_frames(self, frames):
        """
        Process the frames and optionally save the first, middle, and last frame.
        """
        cv2.imwrite("first_frame.jpg", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            "middle_frame.jpg",
            cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite("last_frame.jpg", cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

    def check_manually_webcam_frame_rate(self):
        """
        Estimate the actual frame rate of the webcam.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        num_frames = 60
        start = time.time()

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

        end = time.time()
        seconds = end - start
        fps = num_frames / seconds

        print("Estimated frames per second: {0}".format(fps))

        cap.release()

    def start_capture(self, save_test_frames=False):
        """
        Capture frames from the webcam, process them, and return as a tensor.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None

        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Actual frames per second: {actual_fps}")

        frames = []
        last_frame_time = time.time()
        last_time = time.time()

        while True:
            current_time = time.time()
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame. Exiting ...")
                    break

                cv2.imshow("Webcam Feed", frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

                if len(frames) > self.frame_count:
                    frames.pop(0)

                last_frame_time = current_time

                if time.time() - last_time >= 1:
                    frames_to_process = frames[-self.frame_count :]
                    frames_tensor = torch.tensor(np.array(frames_to_process))
                    print("Tensor shape:", frames_tensor.shape)
                    last_time = time.time()

                    if save_test_frames:
                        self._process_and_save_frames(frames)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        return torch.tensor(np.array(frames[-self.frame_count :]))


if __name__ == "__main__":
    webcam_capture = WebcamCapture()
    webcam_capture.check_manually_webcam_frame_rate()
    frames_tensor = webcam_capture.start_capture(save_test_frames=True)
    print("Tensor shape:", frames_tensor.shape)
