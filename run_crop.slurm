#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=12:00:00
#SBATCH --job-name="fine-grained"
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH --ntasks-per-node=2         # Number of tasks (GPUs) per node
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --output=./logs/crop/%j_output.out
#SBATCH --error=./logs/crop/%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
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

# Set LOCAL_RANK for Slurm tasks
export LOCAL_RANK=$SLURM_LOCALID

# Set the master address (usually the hostname)
export MASTER_ADDR=$(hostname)

# Ensure MASTER_PORT is set correctly
MASTER_PORT=$(shuf -i 20000-65000 -n 1)
if [ -z "$MASTER_PORT" ]; then
  MASTER_PORT=29500
fi

# Explicitly set other ports if needed
MASTER_PORT_TRAIN=${MASTER_PORT_TRAIN:-$MASTER_PORT}
# MASTER_PORT_LOSS_K2=${MASTER_PORT_LOSS_K2:-$(($MASTER_PORT + 1))}
# MASTER_PORT_LOSS_K3=${MASTER_PORT_LOSS_K3:-$(($MASTER_PORT + 2))}
# MASTER_PORT_LOSS_K4=${MASTER_PORT_LOSS_K4:-$(($MASTER_PORT + 3))}
# MASTER_PORT_LOSS_K5=${MASTER_PORT_LOSS_K5:-$(($MASTER_PORT + 4))}
MASTER_PORT_EVAL=${MASTER_PORT_EVAL:-$(($MASTER_PORT + 5))}

# echo "Testing Loss function"

# # Test loss for K=2
# echo "Starting loss test for K=2 with MASTER_PORT=${MASTER_PORT_LOSS_K2}"
# torchrun --nproc_per_node=2 \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT_LOSS_K2} \
#          loss_crop.py --K=2 >> logs/crop/loss_k2.log

# # Test loss for K=3
# echo "Starting loss test for K=3 with MASTER_PORT=${MASTER_PORT_LOSS_K3}"
# torchrun --nproc_per_node=2 \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT_LOSS_K3} \
#          loss_crop.py --K=3 >> logs/crop/loss_k3.log

# # Test loss for K=4
# echo "Starting loss test for K=4 with MASTER_PORT=${MASTER_PORT_LOSS_K4}"
# torchrun --nproc_per_node=2 \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT_LOSS_K4} \
#          loss_crop.py --K=4 >> logs/crop/loss_k4.log

# # Test loss for K=5
# echo "Starting loss test for K=5 with MASTER_PORT=${MASTER_PORT_LOSS_K5}"
# torchrun --nproc_per_node=2 \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT_LOSS_K5} \
#          loss_crop.py --K=5 >> logs/crop/loss_k5.log

# Train model
echo "Starting training with MASTER_PORT=${MASTER_PORT_TRAIN}"
torchrun --nproc_per_node=2 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT_TRAIN} \
         train_crop.py #>> logs/crop/train.log

# Eval model
echo "Starting evaluation with MASTER_PORT=${MASTER_PORT_EVAL}"
torchrun --nproc_per_node=2 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT_EVAL} \
         eval_crop.py >> logs/crop/test.log

echo "All processes completed."
