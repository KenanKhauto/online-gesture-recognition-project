import cv2
import numpy as np
import torch
import time

import cv2
import time

def check_webcam_frame_rate():
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
    fps  = num_frames / seconds

    print("Estimated frames per second : {0}".format(fps))

    cap.release()

check_webcam_frame_rate()


def capture_and_process_video(frame_count=142):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    target_fps = 30
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual FPS: {actual_fps}")

    frames = []
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        cv2.imshow('Webcam Feed', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

        if len(frames) > frame_count:
            frames.pop(0)

        if time.time() - last_time >= 1:
            frames_to_process = frames[-frame_count:]
            frames_tensor = torch.tensor(np.array(frames_to_process))
            print("Tensor shape:", frames_tensor.shape)

            # Save some frames for inspection
            cv2.imwrite("first_frame.jpg", cv2.cvtColor(frames_to_process[0], cv2.COLOR_RGB2BGR))
            cv2.imwrite("middle_frame.jpg", cv2.cvtColor(frames_to_process[len(frames_to_process)//2], cv2.COLOR_RGB2BGR))
            cv2.imwrite("last_frame.jpg", cv2.cvtColor(frames_to_process[-1], cv2.COLOR_RGB2BGR))

            last_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_and_process_video()
