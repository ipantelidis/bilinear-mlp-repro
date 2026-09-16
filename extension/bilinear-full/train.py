"""train.py — Checkpoint loading utility for bilinear-full."""
import torch
import torch.nn.functional as F


def elbo_loss(recon, x, mu, logvar, beta=1.0):
    recon_l = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_l    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_l + beta * kl_l, recon_l, kl_l


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device); model.eval()
    print(f"Loaded {path}  (epoch {ckpt['epoch']})")
    return ckpt
