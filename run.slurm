#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/%j_output.out
#SBATCH --error=./logs/%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train model
python model.py
# Eval Model
python eval.py

#partition can be all_usr_prod or all_serial
