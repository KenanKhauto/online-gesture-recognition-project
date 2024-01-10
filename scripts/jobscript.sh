#!/bin/bash

#SBATCH --job-name=gesture_training_cnn_trans

#SBATCH --partition=gpu
#SBATCH --nodes=1                        # Run all processes on a single node
#SBATCH --ntasks=1                       # Run a single task

#SBATCH --gres=gpu:8                         # Number of GPUs to use
#SBATCH --time=10:00:00                  # Time limit hrs:min:sec

#SBATCH --output=console_output/training_%j.txt

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kenan.khauto@outlook.com

export PYTHONPATH=/scratch/vihps/vihps14/env/lib/python3.9/site-packages:$PYTHONPATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/vihps/vihps14/env/

cd ~/project/online-gesture-recognition-project/

python train.py /scratch/vihps/vihps14/IPN_Hand/frames /scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TrainList.txt /scratch/vihps/vihps14/IPN_Hand/annotations/Annot_TestList.txt /scratch/vihps/vihps14/IPN_Hand/models/cnn_trans.pth
