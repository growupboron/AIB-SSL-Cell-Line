import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from train_pl import SimCLRModule
from data import RxRx1DataModule
import pytorch_lightning as pl

def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for (images, cell_type, sirna_ids) in dataloader:
            images = images.to(model.device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(cell_type.cpu().numpy())
    
    return np.concatenate(features), np.concatenate(labels)

def plot_confusion_matrix(conf_mat, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    fig.savefig("tmp.png", dpi=fig.dpi)

def main():
    pl.seed_everything(42)
    # Load the trained model
    model = SimCLRModule.load_from_checkpoint(checkpoint_path="checkpoints/parallel/last.ckpt")
    model.eval()
    model.freeze()
    
    # Prepare the data module
    data_module = RxRx1DataModule(batch_size=30)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    
    # Extract features and labels
    features, labels = extract_features(model, test_loader)
    
    # Train logistic regression classifier
    classifier = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(features, labels)
    
    # Evaluate classifier
    predictions = classifier.predict(features)
    print(classification_report(labels, predictions))
    
    # Plot confusion matrix
    conf_mat = confusion_matrix(labels, predictions)
    class_names = ["HUVEC", "HEPG2", "RPE", "U2OS"]
    plot_confusion_matrix(conf_mat, class_names)

if __name__ == "__main__":
    main()
