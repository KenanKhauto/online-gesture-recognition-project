import h5py
import cv2
import os

def store_frames_to_hdf5(frame_folders, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf_file:
        len_vids = len(os.listdir(frame_folders))
        counter = 0
        for video_folder in os.listdir(frame_folders):
            video_path = os.path.join(frame_folders, video_folder)
            if os.path.isdir(video_path):
                video_group = hdf_file.create_group(video_folder)
                for frame_file in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame_file)
                    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

                    # Check if the frame is read correctly
                    if frame is not None:
                        frame_dataset_name = f"{video_folder}_{frame_file.split('.')[0]}"
                        video_group.create_dataset(frame_dataset_name, data=frame)
                        #print(f"{frame_dataset_name} created!")
                    else:
                        print(f"Failed to read frame: {frame_path}")
            counter += 1
            print(f"Video: {video_path} done! {counter}/{len_vids}")

frame_folders = os.path.join(".", "IPN_Hand", "frames")
hdf5_path = os.path.join(".", "IPN_Hand", "hand_gestures.h5")
store_frames_to_hdf5(frame_folders, hdf5_path)
