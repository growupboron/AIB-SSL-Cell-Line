import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from dataset_crop import RxRx1WildsCellDataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import random_split
import argparse
import logging
from train_crop import SimCLREncoder, load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get data loaders
def get_data_loaders():
    generator = torch.Generator().manual_seed(42)

    # Define Albumentations transformation pipeline for basic preprocessing
    basic_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.0232, 0.0618, 0.0403), std=(0.0266, 0.0484, 0.0210)),
    ])

    dataset = RxRx1WildsCellDataset(
        img_dir="rxrx1_cells/",
        summary_file="rxrx1_cells/summary_rxrx1.csv",
        subset="train",
        transform=basic_transform,
        metadata_file="data/rxrx1_v1.0/metadata.csv",
        num_img=3,
        mode="random",
        include_labels=True
    )
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Create data loaders without DistributedSampler for simplicity
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=128, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    eval_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    return train_loader, eval_loader

# Function to extract features and labels
def extract_features_and_labels(model, loader, device, save_dir, split_name):
    logger.info(f"Starting feature extraction for {split_name} set...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    feature_list = []
    label_list = []

    with torch.no_grad():
        for i, (images, coarse, fine_grained) in enumerate(loader):
            if i % 10 == 0:
                logger.info(f"Processing batch {i+1}/{len(loader)}...")

            # Move images to device
            images = images.to(device)
            batch_size, num_img, C, H, W = images.shape
            images = images.view(-1, C, H, W)

            # Forward pass through the model's base
            output = model.module.base(images)
            output = torch.flatten(output, start_dim=1)
            output = output.view(batch_size, num_img, -1)
            aggregated_features = output.mean(dim=1)

            # Move to CPU and convert to NumPy
            features_cpu = aggregated_features.cpu().numpy()
            labels_cpu = fine_grained.cpu().numpy()

            # Save features and labels to disk
            np.save(os.path.join(save_dir, f'{split_name}_features_batch_{i}.npy'), features_cpu)
            np.save(os.path.join(save_dir, f'{split_name}_labels_batch_{i}.npy'), labels_cpu)

            # Optionally, clear cache
            del images, output, aggregated_features, features_cpu, labels_cpu
            torch.cuda.empty_cache()

    logger.info(f"Feature extraction for {split_name} set completed.")

def load_and_concatenate_features(save_dir, split_name):
    feature_files = sorted([f for f in os.listdir(save_dir) if f.startswith(f'{split_name}_features_batch_')])
    label_files = sorted([f for f in os.listdir(save_dir) if f.startswith(f'{split_name}_labels_batch_')])

    features_list = [np.load(os.path.join(save_dir, f)) for f in feature_files]
    labels_list = [np.load(os.path.join(save_dir, f)) for f in label_files]

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels

# Function to perform PCA and t-SNE for visualization
def plot_feature_space(features, labels, save_dir):
    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2, random_state=42)

    features_pca = pca.fit_transform(features)
    features_tsne = tsne.fit_transform(features_pca)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=labels, palette="deep", legend="full", alpha=0.7)
    plt.title("t-SNE of Extracted Features")
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'tsne_plot.png')
    plt.savefig(plot_path)
    plt.close()

# Function to evaluate the model using the selected classifier
def evaluate_model(train_features, train_labels, test_features, test_labels, save_dir, classifier_type):
    if classifier_type == "logistic":
        classifier = LogisticRegression(multi_class="multinomial", max_iter=1000, verbose=1, n_jobs=-1)
    elif classifier_type == "knn":
        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='macro')
    cm = confusion_matrix(test_labels, predictions)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Save the confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a SimCLR model with different classifiers.")
    parser.add_argument("--classifier", type=str, choices=["logistic", "knn"], default="logistic", help="Classifier type to use for evaluation.")
    args = parser.parse_args()

    checkpoint_dir = "./checkpoints/crop"
    results_dir = "./results/crop"

    # Initialize distributed processing
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    train_loader, eval_loader = get_data_loaders()

    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    simclr_model = SimCLREncoder(base_model, out_features=1139)
    simclr_model = nn.SyncBatchNorm.convert_sync_batchnorm(simclr_model)
    simclr_model.to(device)
    simclr_model = DistributedDataParallel(simclr_model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(simclr_model.parameters(), lr=3e-4, weight_decay=0.01)
    start_epoch = load_checkpoint(checkpoint_dir, simclr_model, optimizer)

    logger.info("Starting feature extraction for train set")
    extract_features_and_labels(simclr_model, train_loader, device, results_dir, 'train')

    logger.info("Starting feature extraction for test set")
    extract_features_and_labels(simclr_model, eval_loader, device, results_dir, 'test')

    # Load features from disk
    train_features, train_labels = load_and_concatenate_features(results_dir, 'train')
    test_features, test_labels = load_and_concatenate_features(results_dir, 'test')

    logger.info("Plotting feature space")
    plot_feature_space(test_features, test_labels, results_dir)

    logger.info(f"Evaluating model performance with {args.classifier} classifier")
    evaluate_model(train_features, train_labels, test_features, test_labels, results_dir, args.classifier)

    if dist.is_initialized():
        logger.info("Destroying distributed process group")
        dist.destroy_process_group()
