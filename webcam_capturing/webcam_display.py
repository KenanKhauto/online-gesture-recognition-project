import cv2
import numpy as np
import torch

def capture_video(frame_count=142):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frames = []
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        cv2.imshow('Recording...', frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Padding if frames are less than required
    while len(frames) < frame_count:
        frames.append(frames[-1])

    cap.release()
    cv2.destroyAllWindows()

    # Convert to NumPy array, then to PyTorch tensor
    frames_np = np.array(frames)
    frames_tensor = torch.tensor(frames_np)
    
    return frames_tensor

frames_tensor = capture_video()
print(frames_tensor.shape)
