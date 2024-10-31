import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import os
import torch.distributed as dist
import logging
import sys
import argparse

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("loss_functions_debug.log")
    ]
)
logger = logging.getLogger(__name__)

def gather(z):
    """
    Gathers tensors from all processes in the distributed group.

    Args:
        z (torch.Tensor): Tensor to gather.

    Returns:
        torch.Tensor: Gathered tensor from all processes.
    """
    try:
        world_size = dist.get_world_size()
        logger.debug(f"Gathering tensor from all {world_size} processes")
        gather_z = [torch.zeros_like(z) for _ in range(world_size)]
        logger.debug(f"Initialized list of zero tensors for gathering")
        gather_z = diffdist.functional.all_gather(gather_z, z)
        logger.debug(f"All_gather completed")
        gather_z = torch.cat(gather_z)
        logger.debug(f"Concatenated gathered tensors with shape {gather_z.shape}")
        return gather_z
    except Exception as e:
        logger.exception("Failed to gather tensors from all processes")
        raise e

def accuracy(logits, labels, k):
    """
    Computes the top-k accuracy of predictions.
    """
    try:
        logger.debug(f"Calculating top-{k} accuracy")
        topk = logits.topk(k, dim=1, largest=True, sorted=True)[1]  # Shape: [batch_size, k]
        correct = topk.eq(labels).any(dim=1).float()  # Shape: [batch_size]
        logger.debug(f"Accuracy calculated with shape {correct.shape}")
        return correct
    except Exception as e:
        logger.exception("Failed to calculate accuracy")
        raise e

def mean_cumulative_gain(logits, labels, k):
    """
    Computes the mean cumulative gain for the top-k predictions.
    """
    try:
        logger.debug(f"Calculating Mean Cumulative Gain for top-{k} predictions")
        topk = torch.sort(logits.topk(k, dim=1)[1], dim=1)[0]
        labels_sorted = torch.sort(labels, dim=1)[0]
        mcg = (topk == labels_sorted).float().mean(dim=1)
        logger.debug(f"Mean Cumulative Gain calculated with shape {mcg.shape}")
        return mcg
    except Exception as e:
        logger.exception("Failed to calculate Mean Cumulative Gain")
        raise e

def mean_average_precision(logits, labels, k):
    """
    Computes the Mean Average Precision (MAP) for the top-k predictions.
    """
    try:
        logger.debug(f"Calculating Mean Average Precision for top-{k} predictions")
        topk = logits.topk(k, dim=1, largest=True, sorted=True)[1]
        labels_expanded = labels
        relevant = topk.eq(labels_expanded)
        ranks = torch.nonzero(relevant, as_tuple=False)
        if ranks.numel() == 0:
            map_score = torch.tensor(0.0, device=logits.device)
            logger.debug(f"No relevant labels found in top-{k} predictions")
            return map_score
        ranks = ranks[:, 1].float() + 1
        precision_at_k = 1.0 / ranks
        map_score = precision_at_k.mean()
        logger.debug(f"Mean Average Precision calculated with value {map_score.item()}")
        return map_score
    except Exception as e:
        logger.exception("Failed to calculate Mean Average Precision")
        raise e

class NTXent(nn.Module):
    """
    Generalized Contrastive loss with distributed data parallel support for K positive views.
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.
        logger.debug(f"Initialized NTXent with tau={self.tau}, multiplier={self.multiplier}, distributed={self.distributed}")

    def forward(self, z, get_map=False):
        try:
            logger.debug("Forward pass started for NTXent loss")
            n = z.shape[0]
            logger.debug(f"Input tensor shape: {z.shape}")
            assert n % self.multiplier == 0, f"Batch size {n} is not divisible by multiplier {self.multiplier}"

            z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
            logger.debug(f"Normalized input tensor with tau={self.tau}")

            if self.distributed:
                logger.debug("Distributed flag is True. Gathering tensors from all processes")
                z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
                z_list = diffdist.functional.all_gather(z_list, z)
                z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
                logger.debug("All_gather and chunking completed")
                z_sorted = []
                for m in range(self.multiplier):
                    for i in range(dist.get_world_size()):
                        z_sorted.append(z_list[i * self.multiplier + m])
                z = torch.cat(z_sorted, dim=0)
                n = z.shape[0]
                logger.debug(f"Concatenated sorted tensors with new shape {z.shape}")

            logits = z @ z.t()
            logger.debug(f"Logits computed with shape {logits.shape}")
            logits[torch.arange(n, device=z.device), torch.arange(n, device=z.device)] = -self.LARGE_NUMBER
            logger.debug("Diagonal of logits set to LARGE_NUMBER to exclude self-comparisons")

            logprob = F.log_softmax(logits, dim=1)
            logger.debug("LogSoftmax applied to logits")

            m = self.multiplier
            labels = (torch.repeat_interleave(torch.arange(n, device=z.device), m - 1) +
                     torch.tile(torch.arange(1, m, device=z.device) * n // m, (n,))) % n
            labels = labels.long()
            logger.debug(f"Labels for positive pairs created with shape {labels.shape}")

            loss = -logprob[torch.repeat_interleave(torch.arange(n, device=z.device), m - 1), labels].sum() / n / (m - 1) / self.norm
            logger.debug(f"Loss calculated: {loss.item()}")

            pred = logprob.clone()
            pred[torch.arange(n, device=z.device), torch.arange(n, device=z.device)] = -self.LARGE_NUMBER
            acc = accuracy(pred, labels.view(n, m - 1), k=m - 1)

            logger.debug(f"Accuracy computed with mean: {acc.mean().item()}")

            if get_map:
                _map = mean_average_precision(pred, labels.view(n, m - 1), k=m - 1)
                logger.debug(f"Mean Average Precision computed with value: {_map.item()}")
                return loss, acc, _map

            return loss, acc
        except Exception as e:
            logger.exception("Failed during forward pass of NTXent loss")
            raise e

def test_ntxent_loss(K, batch_size, feature_dim, tau=0.5):
    try:
        logger.info(f"Starting NTXent loss test with K={K}, batch_size={batch_size}, feature_dim={feature_dim}, tau={tau}")

        # Get local rank and set device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        logger.debug(f"Local rank obtained from environment: {local_rank}")

        # Initialize distributed process group
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        logger.info(f"Distributed process group initialized on rank {local_rank} with world_size={world_size}")

        ntxent_loss = NTXent(tau=tau, multiplier=K, gpu=local_rank, distributed=True)
        logger.debug("NTXent loss module initialized")

        z_unique = torch.randn(batch_size, feature_dim, device=device)
        z = z_unique.repeat(K, 1)
        logger.debug(f"Random feature vectors with known positives generated with shape {z.shape} on GPU {local_rank}")

        loss, acc, _map = ntxent_loss(z, get_map=True)
        logger.info(f"NTXent loss computed: {loss.item():.4f}")
        logger.info(f"Accuracy: {acc.mean().item():.4f}")
        logger.info(f"Mean Average Precision: {_map.mean().item():.4f}")

        if local_rank == 0:  # Print only from the main process
            print(f"Testing NTXent Loss with K={K}:")
            print(f"Loss: {loss.item():.4f}")
            print(f"Accuracy: {acc.mean().item():.4f}")
            print(f"Mean Average Precision: {_map.mean().item():.4f}")
            print("-" * 40)
    except Exception as e:
        logger.exception(f"Failed to test NTXent loss with K={K}")
        raise e
    finally:
        if dist.is_initialized():
            logger.debug("Destroying distributed process group")
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NTXent Loss")
    parser.add_argument('--K', type=int, required=True, help='Number of positive views per sample (multiplier)')
    args = parser.parse_args()

    # Example parameters
    #batch_size = 256  # Number of samples per batch
    batch_size = 128  # Number of samples per batch
    feature_dim = 256  # Feature dimensionality
    tau = 0.2  # Temperature parameter

    # Test NTXent loss with specified K value
    test_ntxent_loss(K=args.K, batch_size=batch_size, feature_dim=feature_dim, tau=tau)
