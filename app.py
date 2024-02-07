import cv2
import numpy as np
import torch
import time
import mediapipe as mp
from models.LSTM import LSTMGestureClassifier
import os


class WebcamCapture:
    def __init__(self, frame_count=60, target_fps=30, width=640, height=480, show_landmarks=False):
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
        self.width = width
        self.height = height
        self.model = LSTMGestureClassifier(63, 128, 2, 2)
        model_path = os.path.join(".", "lstm_models_det", "lstm_det_local_epoch_4.pth")
        self.model.load_state_dict(torch.load(model_path))

        if show_landmarks:
            self.show_landmarks = show_landmarks
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
            self.mp_drawing = mp.solutions.drawing_utils
        


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
            ret, _ = cap.read()
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

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Actual frames per second: {actual_fps}")

        frames = []
        video_landmarks = []
        last_frame_time = time.time()
        last_time = time.time()

        while True:
            current_time = time.time()
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame. Exiting ...")
                    break

                ###### Draw landmarks ######
                frame_landmarks = []
                if self.show_landmarks:
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
                    results = self.hands.process(frame)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                            frame_landmarks.append(landmarks)
                    else:
                        frame_landmarks.append(np.zeros((21, 3)))

                    # landmarks_in = torch.tensor(video_landmarks).float()
                    # #print(landmarks_in.shape)
                    # out = self.model(landmarks_in)

                    # print(torch.argmax(out, 1))

                cv2.imshow("Webcam Feed", frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                video_landmarks.append(frame_landmarks)


                if len(frames) > self.frame_count:
                    frames.pop(0)

                last_frame_time = current_time

                if time.time() - last_time >= 1:
                    frames_to_process = frames[-self.frame_count :]
                    frames_tensor = torch.tensor(np.array(frames_to_process)).unsqueeze(0).float()
                    landmarks_to_process = video_landmarks[-self.frame_count:]
                    landmarks_to_process_tensor = torch.tensor(np.array(landmarks_to_process)).float().unsqueeze(0).reshape(1, frames_tensor.shape[1], -1)
                    with torch.no_grad():
                        # print(landmarks_to_process_tensor.shape)
                        out = self.model(landmarks_to_process_tensor)
                        # print(out)

                        y = torch.argmax(out, 1).item()
                        if y == 1:
                            print("No Gestures detected!")
                        else:
                            print("Gesture detected!! ! !")

                    # print("Tensor shape:", frames_tensor.shape)
                    last_time = time.time()

                    if save_test_frames:
                        self._process_and_save_frames(frames)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        return torch.tensor(np.array(frames[-self.frame_count :]))


if __name__ == "__main__":
    webcam_capture = WebcamCapture(60, show_landmarks=True)
    frames_tensor = webcam_capture.start_capture(save_test_frames=False)