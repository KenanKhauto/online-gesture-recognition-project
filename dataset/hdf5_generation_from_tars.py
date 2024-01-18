import h5py
import cv2
import os
import glob
import tarfile
import numpy as np

def store_frames_to_hdf5(path_patterns, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf_file:
        tar_files = glob.glob(path_patterns)
        for t, tar_file in enumerate(tar_files):
            with tarfile.open(tar_file) as tar:
                files = sum(1 for member in tar if member.isfile())
                counter = 0
                for member in tar:
                    if member.isfile():
                        _, video_folder, frame_file = member.name.split("/")
                        content = tar.extractfile(member).read()
                        data = np.frombuffer(content, dtype=np.uint8)
                        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

                        # Check if the frame is read correctly
                        if frame is not None:
                            frame_dataset_name = f"{video_folder}_{frame_file.split('.')[0]}"
                            video_group = hdf_file.require_group(video_folder)
                            video_group.create_dataset(frame_dataset_name, data=frame)
                            # print(f"{frame_dataset_name} created!")
                        else:
                            print(f"Failed to read frame: {member.name}")
                        counter += 1
                        print(f"Frame: {member.name} done! Tarfile {t+1}/{len(tar_files)} File {counter}/{files}")

path_patterns = "/scratch/vihps/vihps15/data/IPN_Hand/frames*.tgz"
hdf5_path = "/scratch/vihps/vihps15/data/IPN_Hand/hand_gestures.h5"
store_frames_to_hdf5(path_patterns, hdf5_path)