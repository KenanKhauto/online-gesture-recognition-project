#!/bin/bash

#SBATCH --job-name=detector_training

#SBATCH --partition=gpu
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                              

#SBATCH --gres=gpu:1                            # Number of GPUs to use
#SBATCH --time=48:00:00                         # Time limit hrs:min:sec

#SBATCH --output=console_output/detector_training_%j.txt

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kenan.khauto@outlook.com

export PYTHONPATH=/scratch/vihps/vihps14/env/lib/python3.9/site-packages:$PYTHONPATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/vihps/vihps14/env/

cd ~/project/online-gesture-recognition-project/

srun python ./run/run_detector.py