#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/%j_output.out
#SBATCH --error=./logs/%j_error.err
#SBATCH --mail-user=274236@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train model
python train_pl.py > logs/train.log
# Eval Model
python eval_pl.py > logs/test.log

