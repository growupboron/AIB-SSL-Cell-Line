import torch
import torchvision
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from dataset_crop import RxRx1WildsCellDataset  
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.modules.batchnorm import SyncBatchNorm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loss_crop import *
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_debug.log")
    ]
)
logger = logging.getLogger(__name__)

BATCH = 96

def setup():
    try:
        logger.debug("Setting up distributed environment")
        dist.init_process_group(backend='nccl', init_method='env://')
        logger.info("Distributed process group initialized")
    except Exception as e:
        logger.exception("Failed to set up distributed environment")
        raise e

def cleanup():
    try:
        logger.debug("Cleaning up distributed process group")
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")
    except Exception as e:
        logger.exception("Failed to clean up distributed environment")

def get_data_loaders():
    logger.debug("Starting to get data loaders")
    try:
        generator = torch.Generator().manual_seed(42)  # For reproducible results
        logger.debug("Data generator initialized with seed 42")

        # Define Albumentations transformation pipeline for basic preprocessing
        basic_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.0232, 0.0618, 0.0403), std=(0.0266, 0.0484, 0.0210)),
        ])

        # Initialize dataset with Albumentations transforms
        dataset = RxRx1WildsCellDataset(
            img_dir="rxrx1_cells",
            summary_file="rxrx1_cells/summary_rxrx1.csv",
            subset="train",
            transform=basic_transform,
            metadata_file="data/rxrx1_v1.0/metadata.csv",
            num_img=3,
            mode="random",
            include_labels=True  
        )
        total_size = len(dataset)
        logger.info(f"Total dataset size: {total_size}")

        # Split dataset
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
        logger.info(f"Dataset split into train: {train_size}, val: {val_size}, test: {test_size}")

        # Create samplers
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
        test_sampler = DistributedSampler(test_set)
        logger.debug("Distributed samplers created for train, val, and test sets")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=BATCH, 
            shuffle=False, 
            sampler=train_sampler, 
            num_workers=6, 
            drop_last=True
        )
        eval_loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=BATCH, 
            shuffle=False, 
            sampler=test_sampler, 
            num_workers=6, 
            drop_last=True
        )
        logger.info("Data loaders for training and evaluation created successfully")

        return train_loader, eval_loader
    except Exception as e:
        logger.exception("Failed to get data loaders")
        raise e

def nt_xent_loss(z, tau=0.5):
    try:
        logger.debug("Calculating NT-Xent loss")
        N = z.size(0) // 2
        device = z.device
        cosine_sim = torch.nn.CosineSimilarity(dim=2)
        z_expanded = z.unsqueeze(1).repeat(1, 2*N, 1)
        z_tiled = z.repeat(2*N, 1).view(2*N, 2*N, -1)
        sim = cosine_sim(z_expanded, z_tiled) / tau
        mask = torch.eye(2*N, device=device).bool()
        sim.masked_fill_(mask, float('-inf'))
        sim_softmax = F.softmax(sim, dim=1)
        labels = torch.arange(2*N, device=device)
        labels = (labels + N) % (2*N)
        pos_probs = sim_softmax.gather(1, labels.view(-1, 1)).squeeze()
        loss = -torch.log(pos_probs + 1e-9).mean()
        logger.debug(f"NT-Xent loss calculated: {loss.item()}")
        return loss
    except Exception as e:
        logger.exception("Failed to calculate NT-Xent loss")
        raise e

class SimCLREncoder(nn.Module):
    def __init__(self, base_model, out_features):
        super(SimCLREncoder, self).__init__()
        logger.debug("Initializing SimCLREncoder")
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_features)
        )
        logger.info("SimCLREncoder initialized successfully")

    def forward(self, x):
        try:
            logger.debug(f"SimCLREncoder forward pass started with input shape: {x.shape}")
            x = self.base(x)
            logger.debug(f"After base model: {x.shape}")
            x = torch.flatten(x, 1)
            logger.debug(f"After flattening: {x.shape}")
            x = self.projection_head(x)
            logger.debug(f"After projection head: {x.shape}")
            return x
        except Exception as e:
            logger.exception("Error during forward pass of SimCLREncoder")
            raise e

def get_simclr_augmentation_pipeline(limit=5):
    logger.debug(f"Creating SimCLR augmentation pipeline with limit={limit}")
    try:
        augs = [
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.0233, 0.0623, 0.0407), std=(0.0267, 0.0481, 0.0209)),
            ]),
            A.Compose([
                A.VerticalFlip(p=0.5),
                A.RandomGamma(p=0.2),
                A.Normalize(mean=(0.0233, 0.0623, 0.0407), std=(0.0267, 0.0481, 0.0209)),
            ]),
            A.Compose([
                A.Rotate(limit=30, p=0.5), 
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
                A.Normalize(mean=(0.0233, 0.0623, 0.0407), std=(0.0267, 0.0481, 0.0209)),
            ]),
            A.Compose([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                A.Normalize(mean=(0.0233, 0.0623, 0.0407), std=(0.0267, 0.0481, 0.0209)),
            ]),
            A.Compose([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),
                A.Normalize(mean=(0.0233, 0.0623, 0.0407), std=(0.0267, 0.0481, 0.0209)),
            ]),
        ]
        logger.info(f"Created {len(augs)} augmentation pipelines")
        return augs[:limit]
    except Exception as e:
        logger.exception("Failed to create SimCLR augmentation pipeline")
        raise e

def save_checkpoint(state, epoch, base_dir="./checkpoints/crop", filename="checkpoint_{epoch}.pth.tar"):
    try:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, filename.format(epoch=epoch))
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved at '{filepath}' for epoch {epoch}")
    except Exception as e:
        logger.exception(f"Failed to save checkpoint for epoch {epoch}")

def load_checkpoint(checkpoint_dir, model, optimizer):
    logger.debug(f"Attempting to load checkpoint from directory: {checkpoint_dir}")
    try:
        checkpoints = [chkpt for chkpt in os.listdir(checkpoint_dir) if chkpt.endswith('.pth.tar')]
        if not checkpoints:
            logger.warning(f"No checkpoints found at '{checkpoint_dir}', starting from scratch")
            return 0
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        logger.info(f"Loading checkpoint '{latest_checkpoint_path}'")
        checkpoint = torch.load(latest_checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint '{latest_checkpoint_path}' (epoch {epoch})")
        return epoch
    except Exception as e:
        logger.exception("Error loading checkpoint, starting from epoch 0")
        return 0

def train(rank, world_size, epochs, start_epoch, train_loader, simclr_model, optimizer, scheduler, augmentations, temperature, checkpoint_dir):
    logger.debug(f"Training started on rank {rank} with world_size {world_size}")
    try:
        device = torch.device(f'cuda:{rank}')
        logger.info(f"Device set to {device}")

        writer = SummaryWriter(log_dir=f'./tb_logs/crop_{rank}')
        logger.info(f"TensorBoard SummaryWriter initialized at './tb_logs/crop_{rank}'")

        K = len(augmentations)
        logger.info(f"K is {K}")
        loss_fn = NTXent(tau=temperature, gpu=rank, multiplier=K, distributed=True)
        logger.debug("NT-Xent loss function initialized")

        simclr_model.train()
        logger.info("SimCLR model set to training mode")

        for epoch in range(start_epoch, epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")
            total_loss = 0
            train_loader.sampler.set_epoch(epoch)
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for batch_idx, (imgs, _, _) in enumerate(train_loader):
                    try:
                        optimizer.zero_grad()
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Optimizer gradients zeroed")

                        # Reshape imgs to [batch_size * num_img, C, H, W]
                        batch_size, num_img, C, H, W = imgs.shape
                        imgs = imgs.view(-1, C, H, W)  # Shape: [batch_size * num_img, C, H, W]

                        # Ensure images are in [0, 255] range and uint8 type
                        imgs = imgs * 255.0  # If imgs are in [0, 1] range
                        imgs = imgs.to(torch.uint8)

                        # Apply augmentations
                        views = []
                        for i in range(K):
                            # Apply the i-th augmentation to all images in the batch
                            augmented = [
                                augmentations[i](image=img.permute(1, 2, 0).cpu().numpy())['image']
                                for img in imgs
                            ]
                            # Convert augmented images back to tensors and reshape to [C, H, W]
                            augmented = torch.stack([
                                torch.tensor(item).permute(2, 0, 1).to(device)
                                for item in augmented
                            ])
                            views.append(augmented)
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Augmentations applied")

                        # Concatenate all augmented views
                        z = torch.cat(views, dim=0)  # Shape: [K * batch_size * num_img, C, H, W]
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Concatenated views shape: {z.shape}")

                        # Forward pass through the model
                        z = simclr_model(z)
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Model output shape: {z.shape}")

                        # Compute loss
                        loss, acc, _map = loss_fn(z, get_map=True)
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss.item()}, Acc={acc.mean().item()}, MAP={_map.mean().item()}")

                        # Backward pass and optimization
                        loss.backward()

                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Backward pass completed")

                        torch.nn.utils.clip_grad_norm_(simclr_model.parameters(), max_norm=1.0)
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Gradient clipping applied")

                        optimizer.step()
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Optimizer step completed")

                        # Update metrics and logs
                        total_loss += loss.item()
                        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
                        writer.add_scalar('Loss/Acc', acc.mean().item(), epoch * len(train_loader) + batch_idx)

                        pbar.set_postfix({'Loss': total_loss / (batch_idx + 1)})
                        pbar.update(1)

                        if batch_idx % 32 == 0:
                            logger.info(
                                f"Epoch: [{epoch+1}/{epochs}] "
                                f"Batch: {batch_idx+1}/{len(train_loader)} "
                                f"Loss: {total_loss / (batch_idx + 1):.4f} "
                                f"Accuracy: {acc.mean().item():.4f} "
                                f"MAP: {_map.mean().item():.4f}"
                            )

                        if (epoch + 1) % 5 == 0 and dist.get_rank() == 0:
                            save_checkpoint({
                                'epoch': epoch + 1,
                                'state_dict': simclr_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                            }, epoch + 1, base_dir=checkpoint_dir)
                            logger.info(f"Checkpoint saved at epoch {epoch + 1}")

                        scheduler.step()
                        logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Scheduler stepped")
                    except Exception as batch_e:
                        logger.exception(f"Error during training at epoch {epoch+1}, batch {batch_idx+1}")
                        raise batch_e

            logger.info(f"Epoch {epoch + 1}/{epochs} completed with average loss {total_loss / len(train_loader):.4f}")

        writer.close()
        logger.info("TensorBoard SummaryWriter closed")
    except Exception as e:
        logger.exception("An error occurred during training")
    finally:
        cleanup()

@record
def main():
    logger.debug("Main function started")
    try:
        torch.backends.cudnn.deterministic = True
        logger.info("Set torch.backends.cudnn.deterministic=True for reproducibility")

        checkpoint_dir = "./checkpoints/crop"
        os.makedirs(name=checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint directory '{checkpoint_dir}' is ready")

        # Retrieve local_rank and set device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        logger.debug(f"Local rank obtained from environment: {local_rank}")
        logger.info(f"CUDA device set to {local_rank}")

        # Set up distributed training
        setup()

        # Create SimCLR augmentation pipelines
        augmentations = get_simclr_augmentation_pipeline(limit=5)

        # Loading dataset with Albumentations transforms
        logger.debug("Loading data loaders")
        train_loader, eval_loader = get_data_loaders()
        logger.info("Data loaders loaded successfully")

        # Initialize model
        logger.debug("Initializing base model (ResNet18)")
        base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        logger.debug("Base model loaded")

        logger.debug("Initializing SimCLREncoder")
        simclr_model = SimCLREncoder(base_model, out_features=1139)
        simclr_model = SyncBatchNorm.convert_sync_batchnorm(simclr_model)
        simclr_model.to(device)
        logger.info("SimCLREncoder initialized and moved to device")

        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")

        simclr_model = DDP(simclr_model, device_ids=[local_rank])
        logger.info("Model wrapped with DistributedDataParallel")

        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=1e-4, weight_decay=0.005)
        logger.info("Optimizer (AdamW) initialized")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1)
        logger.info("Scheduler (CosineAnnealingWarmRestarts) initialized")

        # Load checkpoint if available
        start_epoch = load_checkpoint(checkpoint_dir, simclr_model, optimizer)
        logger.info(f"Starting training from epoch {start_epoch}")

        temperature = 0.2
        logger.debug(f"Temperature for NT-Xent loss set to {temperature}")

        # Start training
        train(
            rank=local_rank,
            world_size=dist.get_world_size(),
            epochs=200,
            start_epoch=start_epoch,
            train_loader=train_loader,
            simclr_model=simclr_model,
            optimizer=optimizer,
            scheduler=scheduler,
            augmentations=augmentations,
            temperature=temperature,
            checkpoint_dir=checkpoint_dir
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.exception("An error occurred in the main function")
        raise e

if __name__ == "__main__":
    main()