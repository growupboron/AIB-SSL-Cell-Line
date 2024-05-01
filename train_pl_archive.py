# Training

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import RxRx1DataModule

class SaveLatestCheckpointCallback(Callback):
    def __init__(self, file_path):
        """
        Args:
        file_path (str): Path to save the checkpoint.
        """
        self.file_path = file_path

    def on_train_epoch_end(self, trainer, pl_module):
        """ Called when the train epoch ends. """
        trainer.save_checkpoint(self.file_path)

class SimCLREncoder(nn.Module):
    def __init__(self, base_model, out_features=4):
        super(SimCLREncoder, self).__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])  # Remove the original fc layer
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection_head(x)
        return x

def nt_xent_loss(z, tau=0.5):
    N = z.size(0) // 2
    device = z.device
    print(device)
    cosine_sim = nn.CosineSimilarity(dim=2)

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

    return loss

class SimCLRModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, tau=0.5):
        super(SimCLRModule, self).__init__()
        self.save_hyperparameters()
        base_model = models.resnet18(pretrained=True)
        self.model = SimCLREncoder(base_model, out_features=4)
        self.tau = tau
        self.learning_rate = learning_rate
        self.t1 = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((256, 256)),
            transforms.RandomCrop(175),
            transforms.ToTensor(),
        ])
        self.t2 = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((256, 256)),
            transforms.RandomRotation(120),
            transforms.ToTensor()
        ])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, cell, sirna = batch
        #images_i, images_j = torch.stack([self.t1(image).half().cuda() for image in images]), torch.stack([self.t2(image).half().cuda() for image in images])
        images_i = torch.stack([self.t1(image).to(self.device, dtype=torch.float16) for image in images])
        images_j = torch.stack([self.t2(image).to(self.device, dtype=torch.float16) for image in images])
        z_i = self(images_i)
        z_j = self(images_j)
        z = torch.cat([z_i, z_j], dim=0)
        loss = nt_xent_loss(z, self.tau)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     images, cell_type_ids, sirna_ids = batch  
    #     images = images.to(self.device)
    #     outputs = self(images)
    #     return outputs, cell_type_ids, sirna_ids

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, cell_type_ids, sirna_ids = batch
        images = images.to(self.device)
        logits = self(images)  # Logits: the raw output values from the model
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        predicted_labels = torch.argmax(probabilities, dim=1)  # Get the predicted labels
        return predicted_labels, cell_type_ids, sirna_ids

def main():
    pl.seed_everything(42)
    data_module = RxRx1DataModule(batch_size=512)
    model = SimCLRModule(learning_rate=3e-4, tau=0.5)

    # Path to the latest checkpoint
    checkpoint_path = "./checkpoints/parallel/last.ckpt"

    # Check if the last checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from the last checkpoint: {checkpoint_path}")
        resume_from_checkpoint = checkpoint_path
    else:
        print("No checkpoint found, starting training from scratch")
        resume_from_checkpoint = None

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/parallel/",
        monitor="train_loss",
        save_top_k=3,
        mode="min"
    )
    latest_checkpoint_callback = SaveLatestCheckpointCallback(checkpoint_path)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("tb_logs/", name="parallel")

    trainer = pl.Trainer(
        max_epochs=25,
        devices=4,  # Specify the number of GPUs
        strategy='ddp',
        callbacks=[checkpoint_callback, latest_checkpoint_callback, lr_monitor],
        logger=logger,
        precision='16-mixed',
    )
    trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)

if __name__ == "__main__":
    main()