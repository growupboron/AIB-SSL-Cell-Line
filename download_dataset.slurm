#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=04:00:00
#SBATCH --job-name="dataset"
#SBATCH --output=./logs/misc/%j_output.out
#SBATCH --error=./logs/misc/%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# source /work/cvcs_2023_group23/AIB_new/AIB_env/bin/activate
# cd /work/cvcs_2023_group23/AIB_new/AIB

## WILDS Dataset
#python3 download_dataset.py >> logs/misc/dataset.log

## RxRx1-Wilds Cell-Level Dataset: https://zenodo.org/records/7272553
# wget https://zenodo.org/records/7272553/files/rxrx1_cells.zip >> logs/misc/dataset.log
rm -rf /work/cvcs_2023_group23/AIB_new/AIB/rxrx1_cells/ >> logs/misc/dataset.log #If failed in between
unzip rxrx1_cells.zip >> logs/misc/dataset.log
# Metadata files from https://github.com/microsoft/batch-effects-normalization/tree/main/misc
# wget https://raw.githubusercontent.com/microsoft/batch-effects-normalization/refs/heads/main/misc/summary_rxrx1.csv
# wget https://raw.githubusercontent.com/microsoft/batch-effects-normalization/refs/heads/main/misc/metadata.csv