import os

frames_path = "/scratch/vihps/vihps14/IPN_Hand/hand_gestures.h5"
ann_train = "/scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TrainList.txt"
ann_test = "/scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TestList.txt"
save_path = "/scratch/vihps/vihps14/IPN_Hand/models/3dresnet_v2.pth"

os.system(f"python ./train/train_3dresnet.py {frames_path} {ann_train} {ann_test} {save_path}")
