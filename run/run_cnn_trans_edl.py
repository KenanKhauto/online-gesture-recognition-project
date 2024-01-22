import os

frames_path = "/scratch/vihps/vihps15/data/IPN_Hand/hand_gestures.h5"
ann_train = "/scratch/vihps/vihps15/data/IPN_Hand/annotations/Annot_TrainList.txt"
ann_test = "/scratch/vihps/vihps15/data/IPN_Hand/annotations/Annot_TestList.txt"
save_path = "/scratch/vihps/vihps15/models/cnn_trans_edl.pth"

os.system(f"python train_cnn_trans_edl.py {frames_path} {ann_train} {ann_test} {save_path}")


