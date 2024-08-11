import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from dataset import RxRx1Dataset
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from train import SimCLREncoder, nt_xent_loss
from torch.utils.data import random_split


def get_data_loaders():
    generator = torch.Generator().manual_seed(42)
    dataset = RxRx1Dataset(metadata_csv='metadata.csv', root_dir='/work/cvcs_2023_group23/AIB/data/rxrx1_v1.0', transform=transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))]))
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True, num_workers=4)
    return train_loader, eval_loader


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    try:
        checkpoints = [chkpt for chkpt in os.listdir(checkpoint_dir) if chkpt.endswith('.pth.tar')]
        if not checkpoints:
            print(f"No checkpoints found at '{checkpoint_dir}', starting from scratch")
            return 0
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint '{latest_checkpoint_path}'")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{latest_checkpoint_path}' (epoch {epoch})")
        return epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

def extract_features_and_labels(model, loader, verbose_every=500):
    print("Starting feature extraction...")
    model.eval()
    features = []
    labels = []
    device = next(model.parameters()).device
    batch_count = len(loader)
    with torch.no_grad():
        for i, (images, label, _) in enumerate(loader):
            if i % verbose_every == 0:
                print(f"Processing batch {i+1}/{batch_count}...")
            images = images.to(device)
            output = model.base(images)
            output = torch.flatten(output, start_dim=1)
            features.append(output)
            labels.append(label.to(device))
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    print("Feature extraction completed. Processed total batches:", batch_count)
    return features, labels

def evaluate_simclr(model, train_loader, test_loader, train_features, train_labels, test_features, test_labels):
    print("Model evaluation started.")
    model.eval()
    classifier = LogisticRegression(multi_class="multinomial", max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Test Accuracy: {accuracy}')

if __name__ == "__main__":
    checkpoint_dir = "./checkpoints/parallel"
    train_loader, eval_loader = get_data_loaders()

    base_model = torchvision.models.resnet50(pretrained=True)
    simclr_model = SimCLREncoder(base_model, out_features=4)
    optimizer = torch.optim.SGD(simclr_model.parameters(), lr=1e-3, momentum=0.8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr_model = simclr_model.to(device)

    start_epoch = load_checkpoint(checkpoint_dir, simclr_model, optimizer)

    print(f'Starting feature extraction for train set')
    train_features, train_labels = extract_features_and_labels(simclr_model, train_loader)

    print(f'Starting feature extraction for test set')
    test_features, test_labels = extract_features_and_labels(simclr_model, eval_loader)

    evaluate_simclr(simclr_model, train_loader, eval_loader, train_features, train_labels, test_features, test_labels)
