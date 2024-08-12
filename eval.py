import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from dataset import RxRx1Dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from train import SimCLREncoder, load_checkpoint
from torch.utils.data import random_split

# Function to get data loaders
def get_data_loaders():
    generator = torch.Generator().manual_seed(42)
    dataset = RxRx1Dataset(metadata_csv='metadata.csv', root_dir='/work/cvcs_2023_group23/AIB/data/rxrx1_v1.0', transform=transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ]))
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
    return train_loader, eval_loader

# Function to extract features and labels
def extract_features_and_labels(model, loader, device, verbose_every=500):
    print("Starting feature extraction...")
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, (images, label, _) in enumerate(loader):
            if i % verbose_every == 0:
                print(f"Processing batch {i+1}/{len(loader)}...")
            images = images.to(device)
            output = model.base(images)
            output = torch.flatten(output, start_dim=1)
            features.append(output.cpu())  # Move to CPU to conserve GPU memory
            labels.append(label.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    print("Feature extraction completed.")
    return features, labels

# Function to perform PCA and t-SNE for visualization
def plot_feature_space(features, labels):
    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2, random_state=42)

    features_pca = pca.fit_transform(features)
    features_tsne = tsne.fit_transform(features_pca)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=labels, palette="deep", legend="full", alpha=0.7)
    plt.title("t-SNE of Extracted Features")
    plt.show()

# Function to evaluate the model using Logistic Regression
def evaluate_model(train_features, train_labels, test_features, test_labels):
    classifier = LogisticRegression(multi_class="multinomial", max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='macro')
    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print("Classification Report:\n", report)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

if __name__ == "__main__":
    checkpoint_dir = "./checkpoints/parallel_noPL"
    train_loader, eval_loader = get_data_loaders()

    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    simclr_model = SimCLREncoder(base_model, out_features=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr_model = simclr_model.to(device)

    start_epoch = load_checkpoint(checkpoint_dir, simclr_model)

    print(f"Starting feature extraction for train set")
    train_features, train_labels = extract_features_and_labels(simclr_model, train_loader, device)

    print(f"Starting feature extraction for test set")
    test_features, test_labels = extract_features_and_labels(simclr_model, eval_loader, device)

    print("Plotting feature space")
    plot_feature_space(test_features.numpy(), test_labels.numpy())

    print("Evaluating model performance")
    evaluate_model(train_features.numpy(), train_labels.numpy(), test_features.numpy(), test_labels.numpy())