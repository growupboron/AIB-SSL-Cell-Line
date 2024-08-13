import torch
import torchvision
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from dataset import RxRx1Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.modules.batchnorm import SyncBatchNorm
from timm.models.vision_transformer import vit_base_patch16_224

BATCH = 256

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_data_loaders():
    generator = torch.Generator().manual_seed(42)
    dataset = RxRx1Dataset(metadata_csv='metadata.csv', root_dir='/work/cvcs_2023_group23/AIB/data/rxrx1_v1.0', transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_sampler, val_sampler, test_sampler = DistributedSampler(train_set), DistributedSampler(val_set), DistributedSampler(test_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=False, sampler=train_sampler, num_workers=6, drop_last=True)
    eval_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH, shuffle=False, sampler=test_sampler, num_workers=6, drop_last=True)
    return train_loader, eval_loader

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MaskedAutoencoderViT, self).__init__()
        
        # Vision Transformer (ViT) Encoder
        self.encoder = vit_base_patch16_224(pretrained=False)
        
        # Decoder specifics for MAE
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 196, decoder_embed_dim))
        self.decoder = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(decoder_embed_dim * mlp_ratio, patch_size * patch_size * in_chans),
        )
        self.norm = norm_layer(embed_dim)
    
    def forward(self, x, mask_ratio=0.75):
        # Encode
        encoded_tokens = self.encoder.patch_embed(x)
        # Masking logic
        N, L, D = encoded_tokens.shape  # batch, length, embedding dimension
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        masked_tokens = torch.gather(encoded_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Add positional embeddings to the encoded tokens
        encoded_tokens += self.encoder.pos_embed[:, :L]
        masked_tokens += self.decoder_pos_embed[:, :masked_tokens.size(1)]
        
        # Decode
        decoded_tokens = self.decoder(masked_tokens)
        return decoded_tokens, ids_restore

def mae_loss(pred, target, ids_restore):
    target = target.view(target.size(0), -1)
    pred = pred.view(pred.size(0), -1)
    loss = (pred - target).pow(2).mean(dim=-1)  # mean squared error (MSE)
    return loss.mean()

def save_checkpoint(state, epoch, base_dir="./checkpoints/mae", filename="checkpoint_{epoch}.pth.tar"):
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, filename.format(epoch=epoch))
    torch.save(state, filepath)

def load_checkpoint(checkpoint_dir, model, optimizer):
    try:
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

def train(rank, world_size, epochs, start_epoch, train_loader, mae_model, optimizer, scheduler, checkpoint_dir):
    
    device = torch.device(f'cuda:{rank}')
   
    writer = SummaryWriter(log_dir=f'./tb_logs/mae_{rank}')
    mae_model.train()

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch_idx, (images, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                masked_images = images.to(device)
                reconstructed_images, ids_restore = mae_model(masked_images)
                
                # Compute loss
                loss = mae_loss(reconstructed_images, masked_images, ids_restore)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
                pbar.set_postfix({'Loss': total_loss / (batch_idx + 1)})
                pbar.update(1)
                
                if batch_idx % 32 == 0:
                    print(f"Epoch: [{epoch}/{epochs}]\tBatch: {batch_idx+1}({round(100 * (batch_idx + 1)/BATCH, 1)}%)\tLoss: {total_loss / (batch_idx + 1)}")
                
                if (epoch + 1) % 5 == 0 and dist.get_rank() == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': mae_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch, base_dir=checkpoint_dir)
                
                scheduler.step()
    
    writer.close()
    cleanup()

@record
def main():
    torch.backends.cudnn.deterministic = True
    
    checkpoint_dir = "./checkpoints/mae"
    os.makedirs(name=checkpoint_dir, exist_ok=True)
    
    # Setting up multi-GPU
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    setup(rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    # loading dataset
    train_loader, eval_loader = get_data_loaders()
    
    # Define MAE Model
    mae_model = MaskedAutoencoderViT()
    mae_model = SyncBatchNorm.convert_sync_batchnorm(mae_model)
    mae_model.to(local_rank)
    
    torch.cuda.empty_cache()
    
    mae_model = DDP(mae_model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    start_epoch = load_checkpoint(checkpoint_dir, mae_model, optimizer)
    
    # Train the model
    train(local_rank, world_size, 50, start_epoch, train_loader, mae_model, optimizer, scheduler, checkpoint_dir)

if __name__ == "__main__":
    main()