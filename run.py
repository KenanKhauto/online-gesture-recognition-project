import os

frames_path = "/scratch/vihps/vihps14/IPN_Hand/frames"
ann_train = "/scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TrainList.txt"
ann_test = "/scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TestList.txt"
save_path = "/scratch/vihps/vihps14/IPN_Hand/models/cnn_trans.pth"

os.system(f"python train.py {frames_path} {ann_train} {ann_test} {save_path}")


