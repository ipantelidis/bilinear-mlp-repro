"""
train.py — Training utilities for BilinearVAE.

Provides:
    vae_loss         ELBO = BCE reconstruction + β·KL
    train_epoch      Single training epoch
    evaluate         Compute loss on a data loader without gradients
    save_checkpoint  Save model weights + epoch to disk
    load_checkpoint  Load checkpoint into model in-place
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def vae_loss(
    recon_x: torch.Tensor,
    x:       torch.Tensor,
    mu:      torch.Tensor,
    logvar:  torch.Tensor,
    beta:    float = 1.0,
) -> torch.Tensor:
    """ELBO loss (sum over pixels and batch)."""
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl


def train_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    str,
    beta:      float = 1.0,
) -> float:
    """Run one epoch; return mean loss per sample."""
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: str,
    beta:   float = 1.0,
) -> float:
    """Compute mean loss per sample without updating weights."""
    model.eval()
    total = 0.0
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        recon, mu, logvar = model(x)
        total += vae_loss(recon, x, mu, logvar, beta).item()
    return total / len(loader.dataset)


def save_checkpoint(model: torch.nn.Module, path: str, epoch: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict()}, path)


def load_checkpoint(
    model:  torch.nn.Module,
    path:   str,
    device: str = "cpu",
) -> dict:
    """Load checkpoint into model in-place; return the full checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded {path}  (epoch {ckpt['epoch']})")
    return ckpt
