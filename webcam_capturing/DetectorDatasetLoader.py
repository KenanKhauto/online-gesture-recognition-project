
"""
custom dataloader for online hand gesture recognition
"""

import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import h5py

class GestureDataset(Dataset):
    def __init__(self, hdf5_path, label_file, transform=None, sample_duration = 142):
        """

        Parameter:
            frame_folders: Path to the directory where frame folders are stored.
            label_file: Path to the label .txt file.
            transform: Optional transforms to be applied on a sample.
        """
        self.hdf5_path = hdf5_path
        # self.frame_folders = frame_folders
        self.labels = self.parse_labels(label_file)
        self.transform = transform
        self.sample_duration = sample_duration
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')

    def __del__(self):
        self.hdf5_file.close()
    
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
                labels.append((folder_name, int(start), int(end), False if label == "D0X" else True, int(id) - 1, int(number_frames)))
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
        Load and return frames from the HDF5 file, between start and end frames.

        Parameters:
            folder_name: Name of the group in the HDF5 file corresponding to the video.
            start: Starting frame number.
            end: Ending frame number.
        """

        frames = []
        total_frames = end - start + 1
        step = max(1, total_frames // self.sample_duration)
        # Ensure the video group exists in the HDF5 file
        if folder_name in self.hdf5_file:
            video_group = self.hdf5_file[folder_name]

            for frame_idx in range(start, end + 1, step):  # Skip frames if more than fixed frame count
                # Construct the dataset name for each frame
                frame_dataset_name = f"{folder_name}_{folder_name}_{str(frame_idx).zfill(6)}"  # Assuming frame datasets are named by their indices

                if frame_dataset_name in video_group:
                    frame_data = video_group[frame_dataset_name]
                    frame = np.array(frame_data)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    if len(frames) == self.sample_duration:
                        break

        # Apply padding to gestures with fewer frames than self.sample_duration
        while len(frames) < self.sample_duration and len(frames) > 0:
            frames.append(np.zeros_like(frames[0]))

        return frames


if __name__ == "__main__":
    file = os.path.join(".", "IPN_Hand","annotations-20231128T085307Z-001", "annotations", "Annot_TrainList.txt")
    # frame_folders = os.path.join(".", "IPN_Hand", "frames")
    hdf5_path = os.path.join(".", "IPN_Hand", "hand_gestures.h5")
    dataset = GestureDataset(hdf5_path, file)
    # test_sample = dataset.get_labels()[0]
    # frames = dataset.load_frames(test_sample[0], test_sample[1], test_sample[2])


    frames_tensor = dataset[4][0]
    h = frames_tensor.shape[1]
    w = frames_tensor.shape[2]

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))
    print(frames_tensor.shape[0])
    for i in range(frames_tensor.shape[0]):
        frame = frames_tensor[i]
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
