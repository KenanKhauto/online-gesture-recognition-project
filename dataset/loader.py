
"""
custom dataloader for online hand gesture recognition
"""

__author__ = "7592047, Kenan Khauto"

import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class GestureDataset(Dataset):
    def __init__(self, frame_folders, label_file, transform=None, sample_duration = 142):
        """

        Parameter:
            frame_folders: Path to the directory where frame folders are stored.
            label_file: Path to the label .txt file.
            transform: Optional transforms to be applied on a sample.
        """
        self.frame_folders = frame_folders
        self.labels = self.parse_labels(label_file)
        self.transform = transform
        self.sample_duration = sample_duration

    def parse_labels(self, label_file):
        """
        Parse the label file and return a structure containing
        the mappings of video name, start frame, end frame, label, label id and the number of frames.
        
        Parameters:
            label_file: .txt file that contains the labels
        
        Returns:
            list of tuples: (folder_name, start_frame, end_frame, label, label id, number of frames)
        """
        labels = []
        with open(label_file, 'r') as file:
            for line in file:
                folder_name, label, id, start, end, number_frames = line.split(",")
                labels.append((folder_name, int(start), int(end), label, int(id) - 1, int(number_frames)))
        return labels

    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        allows the dataset to be indexed, so that each call to dataset[idx] returns 
        the idx-th sample from the dataset.

        Parameters:
            idx: index of the datapoint
        
        Returns:
            a tuple of frames tensor and label id
        """
        folder_name, start, end, label, id, n_frames = self.labels[idx]
        frames = self.load_frames(folder_name, start, end)
        
        transformed_frames = []
        for frame in frames:
            if self.transform:
                # Apply transform to each frame
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)

        # Stack the frames back into a single tensor
        if self.transform:
            frames_tensor = torch.stack(transformed_frames)
            return frames_tensor, id, label
        return np.array(frames), id, label

    def load_frames(self, folder_name, start, end):
        """
        Load and return frames from the specified folder, between start and end frames.

        Parameters:
            folder_name: video name that contain all the frames of a video
            start: starting frame
            end: ending frame
        """

        frames = []
        folder_path = os.path.join(self.frame_folders, folder_name)
        total_frames = end - start + 1
        step = max(1, total_frames // self.sample_duration)
        for frame_idx in range(start, end + 1, step):  # skip frames if more than fixed frame count
            frame_path = os.path.join(folder_path, f"{folder_name}_{str(frame_idx).zfill(6)}.jpg")
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            frames.append(frame)
            if len(frames) == self.sample_duration:
                break

        while len(frames) < self.sample_duration:
            frames.append(np.zeros_like(frames[0]))  # apply padding to gestures with smaller frames number
        return frames


if __name__ == "__main__":
    file = os.path.join(".", "IPN_Hand","annotations-20231128T085307Z-001", "annotations", "Annot_TrainList.txt")
    frame_folders = os.path.join(".", "IPN_Hand", "frames")
    dataset = GestureDataset(frame_folders, file)
    # test_sample = dataset.get_labels()[0]
    # frames = dataset.load_frames(test_sample[0], test_sample[1], test_sample[2])


    frames_tensor = dataset[4][0]
    h = frames_tensor.shape[1]
    w = frames_tensor.shape[2]

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))
    print(frames_tensor.shape[0])
    for i in range(frames_tensor.shape[0]):
        frame = frames_tensor[i].numpy()
        # print(f"shape of frame {i}: {frame.shape}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)

        cv2.imshow("Video Test", frame)

        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
        # out.write(frame)
    cv2.destroyAllWindows()
        
    # out.release()
