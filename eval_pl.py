import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from train_pl import SimCLRModule
from dataset import RxRx1DataModule

def plot_confusion_matrix(conf_mat, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    fig.savefig("tb_logs/eval/confusion_matrix/tmp.png", dpi=fig.dpi)

def main():
    pl.seed_everything(42)

    # Setup logger and checkpoint callback
    logger = TensorBoardLogger("tb_logs", name="eval")
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", save_top_k=1, verbose=True, monitor='val_loss', mode='min')

    # Initialize the data module
    data_module = RxRx1DataModule(batch_size=512)
    #data_module.setup(stage='fit')
    data_module.setup(stage='test')

    # Load the trained model
    model = SimCLRModule.load_from_checkpoint(checkpoint_path="checkpoints/parallel/last.ckpt")
    model.eval()
    model.freeze()

    # Configure the Trainer

    trainer = pl.Trainer(
        devices=4,  # Specify the number of GPUs
        strategy='ddp',
        callbacks=[checkpoint_callback],
        logger=logger,
        precision='16-mixed',
    )

    print("Starting feature extraction...")
    output, cell_type_ids, sirna_ids = trainer.predict(model, dataloaders=data_module.test_dataloader())[0]
    print(output.shape)
    print(cell_type_ids.shape)
    print("Features extracted.")

    # Reshaping it for fit
    features = output.reshape(-1, 1)
    labels = cell_type_ids.reshape(-1, 1)
    
    print("Starting logistic regression training...")
    classifier = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
    #classifier = LogisticRegression(max_iter=2, verbose=1, n_jobs=-1)
    classifier.fit(features, labels)
    print("Logistic regression model trained.")

    print("Evaluating classifier...")
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)  # Get probabilities for top-k accuracy

    print("Classification Report:")
    print(classification_report(labels, predictions))

    print("Top-3 Accuracy:")
    top3_accuracy = top_k_accuracy_score(labels, probabilities, k=3)
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}")

    print("Top-1 Accuracy:")
    top1_accuracy = top_k_accuracy_score(labels, probabilities, k=1)
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}")

    print("Generating confusion matrix plot...")
    conf_mat = confusion_matrix(labels, predictions)
    class_names = ["HUVEC", "HEPG2", "RPE", "U2OS"]
    plot_confusion_matrix(conf_mat, class_names)
    print("Plot generated. Check the 'tmp.png' file.")

if __name__ == "__main__":
    main()