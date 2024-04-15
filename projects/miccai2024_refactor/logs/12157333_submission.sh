#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=ProstNFound
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/fs01/home/harmanan/medAI/projects/miccai2024_refactor/logs/%j_0_log.out
#SBATCH --qos=m2
#SBATCH --signal=USR2@90
#SBATCH --time=8:00:00
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /fs01/home/harmanan/medAI/projects/miccai2024_refactor/logs/%j_%t_log.out /h/harmanan/anaconda3/envs/medai/bin/python3 -u -m submitit.core._submit /fs01/home/harmanan/medAI/projects/miccai2024_refactor/logs
