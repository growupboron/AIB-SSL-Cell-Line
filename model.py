# Load dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import os

dataset = get_dataset(dataset="rxrx1", download=False)
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    ),
)
eval_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)
eval_loader = get_eval_loader(loader="standard", dataset=eval_data, batch_size=16)

def nt_xent_loss(z, tau=0.5):
    """
    Computes the NT-Xent loss for a batch of embeddings using cosine similarity,
    leveraging torch.nn.CosineSimilarity for efficient computation.

    Args:
    - z: Concatenated embeddings of shape (2N, D) where N is the batch size and D is the embedding dimension.
         This should contain pairs of embeddings (z_i, z_j) for positive samples concatenated along the first dimension.
    - tau: The temperature scaling parameter.

    Returns:
    - The mean NT-Xent loss for the input batch.
    """
    N = z.size(0) // 2
    device = z.device
    cosine_sim = torch.nn.CosineSimilarity(dim=2)

    # Expand z to (2N, 2N, D) by unsqueezing and repeating to compute all-vs-all cosine similarities
    z_expanded = z.unsqueeze(1).repeat(1, 2*N, 1)
    z_tiled = z.repeat(2*N, 1).view(2*N, 2*N, -1)

    # Compute cosine similarity
    sim = cosine_sim(z_expanded, z_tiled) / tau

    # Mask to zero out similarities with themselves and compute softmax
    mask = torch.eye(2*N, device=device).bool()
    sim.masked_fill_(mask, float('-inf'))
    sim_softmax = F.softmax(sim, dim=1)

    # Create labels
    labels = torch.arange(2*N, device=device)
    labels = (labels + N) % (2*N)  # Shift labels to match positive pairs

    # Gather the softmax probabilities of positive pairs
    pos_probs = sim_softmax.gather(1, labels.view(-1, 1)).squeeze()

    # Compute the NT-Xent loss
    loss = -torch.log(pos_probs + 1e-9).mean()

    return loss

class SimCLREncoder(nn.Module):
    def __init__(self, base_model, out_features):
        super(SimCLREncoder, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove original fc layer
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.projection_head(x)
        return x

# Define the augmentation pipeline
def get_simclr_augmentation_pipeline(input_size=256, type=1):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]) if type == 1 else transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=180),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        # transforms.GaussianBlur(kernel_size=int(0.1 * input_size)),
        transforms.ToTensor(),
    ])

# Augmentation pipeline
augmentation_pipeline1 = get_simclr_augmentation_pipeline()
augmentation_pipeline2 = get_simclr_augmentation_pipeline(type=2)

# Training setup
base_model = torchvision.models.resnet50(pretrained=True)
simclr_model = SimCLREncoder(base_model, out_features=4)
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=3e-4)

# import pandas as pd
# import numpy as np

# df = pd.read_csv("/content/data/rxrx1_v1.0/metadata.csv")
# df.head()

# df.info()

# print(df["cell_type"].value_counts()) # these are the 4 classes I think

classes_to_id = {
    0:"HUVEC",
    1:"HEPG2",
    2:"RPE",
    3:"U2OS"
}

# print(df["sirna"].value_counts())

# print(df["sirna_id"].value_counts())

#!export CUDA_LAUNCH_BLOCKING=1
# SSL training
epochs = 20
device = torch.device("cuda")
simclr_model = simclr_model.to(device)
temperature = 0.5
weight_decay = 1e-4
# optimizer = torch.optim.ASGD(simclr_model.parameters(), lr=0.001, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=1e-4)

def save_checkpoint(state, filename="simclr_checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0

checkpoint_path = "simclr_checkpoint.pth.tar"
start_epoch = load_checkpoint(checkpoint_path, simclr_model, optimizer)

simclr_model.train()
for epoch in range(start_epoch, epochs):
    total_loss = 0
    for batch_idx, (images, _, _) in enumerate(train_loader):
        optimizer.zero_grad()

        images_i = torch.stack([augmentation_pipeline1(img).to(device) for img in images])
        images_j = torch.stack([augmentation_pipeline2(img).to(device) for img in images])

        z_i = simclr_model(images_i).to(device)
        z_j = simclr_model(images_j).to(device)

        z = torch.cat((z_i, z_j), dim=0)
        loss = nt_xent_loss(z, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss / (batch_idx + 1)}')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': simclr_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)