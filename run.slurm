#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=08:00:00
#SBATCH --job-name="fine-grained"
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/parallel_noPL/%j_output.out
#SBATCH --error=./logs/parallel_noPL/%j_error.err
#SBATCH --mail-user=274236@studenti.unimore.it
#SBATCH --mail-type=ALL
#SBATCH --constraint="gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G"

source AIB_env/bin/activate

# Train model
python -u -m torch.distributed.run \
    --nproc_per_node=3 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12345 \
     train.py >> logs/parallel_noPL/train.log
# Eval Model
python -u eval.py >> logs/parallel_noPL/test.log

#partition can be all_usr_prod or all_serial