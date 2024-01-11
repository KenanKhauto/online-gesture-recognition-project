
import h5py
import os
import numpy as np 


hdf5_path = os.path.join(".", "IPN_Hand", "hand_gestures.h5")
hdf5_file = h5py.File(hdf5_path, 'r')

x = None
for item in hdf5_file:
    video_group = hdf5_file[item]
    n = 0
    for frame in video_group:
        print(frame)
        arr = np.array(video_group[frame])
        n+=1
        if n == 2:
            break
    break
