#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/parallel_noPL/%j_output.out
#SBATCH --error=./logs/parallel_noPL/%j_error.err
#SBATCH --mail-user=274236@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train model
python -m torch.distributed.launch --nproc_per_node=2 train.py > logs/parallel_noPL/train.log
# Eval Model
python -m torch.distributed.launch --nproc_per_node=2 eval.py > logs/parallel_noPL/test.log

#partition can be all_usr_prod or all_serial