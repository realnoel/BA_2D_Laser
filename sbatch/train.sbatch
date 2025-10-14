#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=train
#SBATCH --cpus-per-task=16
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=truncate

module load stack/2024-06 python_cuda/3.11.6

cd /cluster/home/nkompalla/Bachelorarbeit/ba_code

python src/lightning.py