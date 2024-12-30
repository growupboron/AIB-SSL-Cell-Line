#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=04:00:00
#SBATCH --job-name="fine-grained"
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --ntasks-per-node=1         # Number of tasks (GPUs) per node
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --output=./logs/crop/%j_output.out
#SBATCH --error=./logs/crop/%j_error.err
#SBATCH --mail-user=274236@studenti.unimore.it
#SBATCH --mail-type=ALL
#SBATCH --constraint="gpu_A40_48G"

# Activate your environment
source /work/cvcs_2023_group23/AIB_new/AIB_env/bin/activate
cd /work/cvcs_2023_group23/AIB_new/AIB

# Set the number of threads
export OMP_NUM_THREADS=1

# Updated NCCL variables
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Suppress Albumentations update warning
export NO_ALBUMENTATIONS_UPDATE=1

MASTER_PORT_EVAL=29505  # Fixed port since only one GPU

# Eval model
echo "Starting evaluation with MASTER_PORT=${MASTER_PORT_EVAL}"
torchrun --nproc_per_node=1  \
         --master_addr=localhost \
         --master_port=${MASTER_PORT_EVAL} \
         eval_crop_optimized.py >> logs/crop/test.log

echo "All processes completed."