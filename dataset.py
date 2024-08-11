
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import random_split

# Set the seed for reproducibility
seed = 42
generator = torch.Generator().manual_seed(seed)

class RxRx1Dataset(Dataset):
    """
    Custom Dataset for loading RxRx1 microscopy images
    """
    def __init__(self, metadata_csv, root_dir, transform=None):
        self.metadata_df = pd.read_csv(os.path.join(root_dir, metadata_csv))
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        img_path = os.path.join(self.root_dir, f"images/{row['experiment']}/Plate{row['plate']}/{row['well']}_s{row['site']}.png")
        image = Image.open(img_path)
        image = image.convert('RGB')  # Convert to RGB if needed

        if self.transform:
            image = self.transform(image)

        cell_type_id = {"HUVEC": 0, "HEPG2": 1, "RPE": 2, "U2OS": 3}[row['cell_type']]
        sirna_id = row['sirna_id']
        return image, cell_type_id, sirna_id

class RxRx1DataModule(pl.LightningDataModule):
    """
    Data Module for RxRx1 Dataset, compatible with PyTorch Lightning
    """
    def __init__(self, batch_size=32, root_dir="data/rxrx1_v1.0", metadata_csv="metadata.csv"):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.metadata_csv = os.path.join(root_dir, metadata_csv)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        full_dataset = RxRx1Dataset(self.metadata_csv, self.root_dir, transform=self.transform)         
        # Calculate sizes of splits
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        #self.train_set, self.val_set, self.test_set = random_split(full_dataset, [train_size, val_size, test_size])
        # Split the dataset with the given generator of seed 42
        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        if stage in ['fit', None]:
                self.train_dataset = self.train_set
        if stage in ["test", None]:
                self.test_dataset = self.test_set
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)