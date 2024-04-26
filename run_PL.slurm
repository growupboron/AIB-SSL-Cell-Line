#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/parallel/%j_output.out
#SBATCH --error=./logs/parallel/%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train model
python -u train_pl.py > logs/parallel/train_PL.log
# Eval Model
python -u eval_pl.py > logs/parallel/test_PL.log

#partition can be all_usr_prod or all_serial