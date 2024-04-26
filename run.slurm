#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --account=ai4bio2023
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/serial/%j_output.out
#SBATCH --error=./logs/serial/%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train model
python -u train.py > logs/serial/train.log
# Eval Model
python -u eval.py > logs/serial/test.log

#partition can be all_usr_prod or all_serial