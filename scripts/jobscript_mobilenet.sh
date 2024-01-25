#!/bin/bash

#SBATCH --job-name=mobilenet

#SBATCH --partition=gpu
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                              # Run a single task

#SBATCH --gres=gpu:1                            # Number of GPUs to use
#SBATCH --time=72:00:00                  # Time limit hrs:min:sec

#SBATCH --output=/scratch/vihps/vihps15/logs/training_%j.out
#SBATCH --error=/scratch/vihps/vihps15/logs/training_%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=

/home/vihps/vihps15/miniconda3/bin/conda activate /scratch/vihps/vihps15/envs/torch

cd ~/online-gesture-recognition-project/

srun /scratch/vihps/vihps15/envs/torch/bin/python ./run/run_mobilenet.py