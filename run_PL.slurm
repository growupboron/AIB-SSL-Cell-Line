#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --account=ai4bio2023
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/%j_output.out
#SBATCH --error=./logs/%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train model
python train_pl.py > logs/train_PL.log
# Eval Model
python eval_pl.py > logs/test_PL.log

