import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

classes_to_id = {
    0:"HUVEC",
    1:"HEPG2",
    2:"RPE",
    3:"U2OS"
}

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
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * input_size)),
        transforms.ToTensor(),
    ])

# Augmentation pipeline
augmentation_pipeline1 = get_simclr_augmentation_pipeline()
augmentation_pipeline2 = get_simclr_augmentation_pipeline(type=2)

# Training setup
base_model = torchvision.models.resnet50(pretrained=True)
simclr_model = SimCLREncoder(base_model, out_features=4)
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=3e-4)

epochs = 20
device = torch.device("cuda")
simclr_model = simclr_model.to(device)
temperature = 0.5
weight_decay = 1e-4
# optimizer = torch.optim.ASGD(simclr_model.parameters(), lr=0.001, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=1e-4)

def load_checkpoint(checkpoint_dir, model, optimizer):
    try:
        # Load the latest checkpoint
        checkpoints = [chkpt for chkpt in os.listdir(checkpoint_dir) if chkpt.endswith('.pth.tar')]
        if not checkpoints:
            print("No checkpoints found at '{}', starting from scratch".format(checkpoint_dir))
            return 0
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint '{latest_checkpoint_path}'")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{latest_checkpoint_path}' (epoch {epoch})")
        return epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

checkpoint_dir = "./checkpoints/serial"

def extract_features_and_labels(model, loader, verbose_every=500):
    print("Starting feature extraction...")
    model.eval()  # Ensure the model is in evaluation mode
    features = []
    labels = []
    device = next(model.parameters()).device  # Get the device of the model
    batch_count = len(loader)
    with torch.no_grad():
        for i, (images, label, _) in enumerate(loader):
            if i % verbose_every == 0:  # Conditionally print progress
                print(f"Processing batch {i+1}/{batch_count}...")
            images = images.to(device)  # Move images to the device
            output = model.base(images)
            output = torch.flatten(output, start_dim=1)  # Keep on GPU
            features.append(output)
            labels.append(label.to(device))  # Keep labels on GPU if used later

    # Concatenate all features and labels at once, then move to CPU if necessary
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    print("Feature extraction completed. Processed total batches:", batch_count)
    return features, labels

def evaluate_simclr(model, train_loader, test_loader, train_features, train_labels, test_features, test_labels):
    print("Model evaluation started.")
    model.eval()
    #print("Extracting features and labels from training data...")
    #train_features, train_labels = extract_features_and_labels(model, train_loader)
    #print("Extracting features and labels from test data...")
    #test_features, test_labels = extract_features_and_labels(model, test_loader)

    # Train a classifier on the representations
    print("Training the logistic regression classifier...")
    classifier = LogisticRegression(multi_class="multinomial", max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(train_features, train_labels)

    # Predict on the test set
    print("Predicting on the test set...")
    predictions = classifier.predict(test_features)

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Test Accuracy: {accuracy}')

print(f'Starting extract_features_and_labels for train')
features_extracted_train_features, features_extracted_train_labels = extract_features_and_labels(simclr_model,train_loader)

print(f'Starting extract_features_and_labels for test')
features_extracted_eval_features, features_extracted_eval_labels = extract_features_and_labels(simclr_model,eval_loader)

evaluate_simclr(simclr_model,train_loader,eval_loader,features_extracted_train_features,features_extracted_train_labels,features_extracted_eval_features, features_extracted_eval_labels)