import os

frames_path = "/scratch/vihps/vihps14/IPN_Hand/hand_gestures.h5"
ann_train = "/scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TrainList.txt"
ann_test = "/scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TestList.txt"
save_path = "/scratch/vihps/vihps14/IPN_Hand/models/cnn_trans.pth"
process_id = int(os.environ.get("SLURM_PROCID"))

os.system(f"python train_cnn_trans.py {frames_path} {ann_train} {ann_test} {save_path} {process_id}")


