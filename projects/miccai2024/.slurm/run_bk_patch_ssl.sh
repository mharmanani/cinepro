#!/bin/bash

#SBATCH -J SSL-BK-PATCH
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBSTCH --partition=a40
#SBATCH --time 7:59:00
#SBATCH --export=ALL
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
#SBATCH --qos=m2

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240


CENTER=PMCC

# Set environment variables for training
export TQDM_MININTERVAL=60

# Create experiment directory

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo "Sending SIGINT to child process"
  scancel $SLURM_JOB_ID --signal=SIGINT
  wait $child_pid
  echo "Job step terminated gracefully"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

cd /h/harmanan/medAI/projects/miccai2024

/h/harmanan/anaconda3/envs/medai/bin/python bk_patch_ssl.py \
    --batch_size 32 \
    --seed 0 \
    --inv_threshold 0.4 \
    --save_weights_path=/h/harmanan/medAI/ckpt/bk_patch_ssl_$SLURM_JOB_ID.pth \
    --checkpoint_path=/checkpoint/$USER/$SLURM_JOB_ID/checkpoint.pt &

child_pid=$!
wait $child_pid
