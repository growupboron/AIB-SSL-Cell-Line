# AIB-Project-rxrx1
Course Project: AI for Bioinformatics

This repository contains implementations for training and evaluating models using the [RxRx1-Wilds Cell-Level Dataset](https://www.rxrx.ai/rxrx1#Download). The project is structured into two main workflows: 
- standard processing
- crop-based processing (which also uses [cell level dataset](https://zenodo.org/records/7272553)).

## Quickstart
1. Download dataset
   ```
   sbatch download_dataset.slurm
   ```
2. Train and eval model
   ```
   sbatch run.slurm #standard processing
   sbatch run_crop.slurm #crop-based processing
   ```

## Project Structure 
### Python Scripts 
| **Script** | **Description** |
| --- | --- |
| `download_dataset.py` |	Script to download the RxRx1-Wilds dataset automatically.|
| `dataset.py` |	Dataset class for standard processing.|
| `train.py` | Training script for the standard workflow. | 
| `eval.py` |	Evaluation script for standard models, including feature extraction and classification.|
| `dataset_crop.py` |	Dataset class for crop-based processing.|
| `train_crop.py` | Training script for the crop-based workflow with advanced augmentations. | 
| `eval_crop.py`|	Evaluation script for crop-based models with feature visualization (PCA, t-SNE).|
| `loss_crop.py`|	Implements NTXent loss for contrastive learning with distributed support.|

### SLURM Scripts 
| **Script** | **Description** |
| --- | --- |
| `download_dataset.slurm` | Download the dataset |
| `run.slurm` | SLURM job script for standard training. |
| `run_crop.slurm` | SLURM job script for crop-based training. |
