"""
train.py — Loss function and training loop for the bilinear VAE extension.

The ELBO loss supports a β weight on the KL term:

    L(x; β) = E[log p(x|z)]  −  β · KL(q(z|x) || p(z))

    β = 1  →  standard VAE (Kingma & Welling, 2014)
    β > 1  →  β-VAE        (Higgins et al., 2017)

Gaussian input noise is applied during training following Pearce et al. (2025),
which found it improves eigenfeature interpretability by suppressing spurious
high-frequency interactions.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def elbo_loss(recon: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0, recon_loss: str = "bce"):
    """
    Compute the β-weighted ELBO loss (sum over pixels and batch).

    Args:
        recon      : reconstructed input  (any shape, must match x)
        x          : original clean input (any shape)
        mu         : posterior mean,      shape (batch, d_latent)
        logvar     : posterior log-variance, shape (batch, d_latent)
        beta       : KL weight (1 = standard VAE, >1 = β-VAE)
        recon_loss : "bce" for MNIST/Fashion-MNIST (binary-like pixels),
                     "mse" for CIFAR-10 (continuous RGB pixels).
                     BCE assumes a Bernoulli decoder; MSE assumes Gaussian.

    Returns:
        total, recon_l, kl_l  (all scalars)
    """
    if recon_loss == "bce":
        # Bernoulli decoder — works well for near-binary images (MNIST)
        recon_l = F.binary_cross_entropy(recon, x, reduction="sum")
    else:
        # Gaussian decoder — better for natural RGB images (CIFAR-10).
        # Flatten all spatial/channel dims so the sum is over the same
        # number of elements regardless of image shape.
        recon_l = F.mse_loss(recon, x, reduction="none").flatten(1).sum(dim=1).sum()

    # Analytical KL between N(μ, σ²) and N(0, I)
    kl_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_l + beta * kl_l
    return total, recon_l, kl_l


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device,
              beta: float = 1.0, noise_std: float = 0.0,
              recon_loss: str = "bce", training: bool = True):
    """
    Run one pass over the dataset.

    Args:
        model     : VAE model
        loader    : DataLoader
        optimizer : only used when training=True
        device    : torch device
        beta      : KL weight
        noise_std : standard deviation of Gaussian input noise (0 = no noise)
        training  : True for train mode, False for eval mode

    Returns:
        dict with mean per-sample losses: "total", "recon", "kl"
    """
    model.train(training)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0

    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)

            # Add Gaussian noise to input during training only.
            # The model reconstructs the clean x, so noise acts as regularisation.
            if training and noise_std > 0:
                x_in = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
            else:
                x_in = x

            recon, mu, logvar = model(x_in)
            loss, recon_l, kl_l = elbo_loss(recon, x, mu, logvar, beta, recon_loss)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            totals["total"] += loss.item()
            totals["recon"] += recon_l.item()
            totals["kl"]    += kl_l.item()
            n += batch_size

    # Return mean per-sample loss
    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, test_loader, *,
          epochs: int = 30,
          lr: float = 1e-3,
          weight_decay: float = 0.01,
          beta: float = 1.0,
          noise_std: float = 0.3,
          recon_loss: str = "bce",
          device: str = "cpu",
          checkpoint_dir: str = "extension/checkpoints",
          run_name: str = "model"):
    """
    Train a VAE and save the best checkpoint (lowest test loss).

    Args:
        model          : VAE model (VanillaVAE or BilinearVAE)
        train_loader   : training DataLoader
        test_loader    : test DataLoader
        epochs         : number of training epochs
        lr             : learning rate
        weight_decay   : AdamW weight decay
        beta           : KL weight (1 = standard VAE, >1 = β-VAE)
        noise_std      : Gaussian input noise std — set 0 for CIFAR-10
        recon_loss     : "bce" for MNIST (Bernoulli decoder),
                         "mse" for CIFAR-10 (Gaussian decoder)
        device         : "cpu" or "cuda"
        checkpoint_dir : directory to save checkpoints and training log
        run_name       : used as filename prefix for checkpoint and log

    Returns:
        history : list of dicts, one per epoch, with train/test losses
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    # Cosine annealing smoothly decays LR to near-zero by the final epoch,
    # preventing the noisy test loss oscillations seen with a flat LR.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history   = []
    best_loss = float("inf")

    print(f"\nTraining  : {run_name}")
    print(f"β={beta}  noise_std={noise_std}  recon_loss={recon_loss}  "
          f"epochs={epochs}  lr={lr}  wd={weight_decay}")
    print(f"{'Epoch':>5}  {'Train':>10}  {'Test':>10}  {'KL':>8}  {'LR':>8}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        train_log = run_epoch(model, train_loader, optimizer, device,
                              beta=beta, noise_std=noise_std,
                              recon_loss=recon_loss, training=True)
        test_log  = run_epoch(model, test_loader,  optimizer, device,
                              beta=beta, noise_std=0.0,
                              recon_loss=recon_loss, training=False)

        # Step the scheduler once per epoch (after both train and eval passes)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_total": train_log["total"],
            "train_recon": train_log["recon"],
            "train_kl":    train_log["kl"],
            "test_total":  test_log["total"],
            "test_recon":  test_log["recon"],
            "test_kl":     test_log["kl"],
            "lr":          scheduler.get_last_lr()[0],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5}  {train_log['total']:>10.2f}  "
                  f"{test_log['total']:>10.2f}  {test_log['kl']:>8.2f}  "
                  f"{scheduler.get_last_lr()[0]:>8.2e}")

        # Save best checkpoint
        if test_log["total"] < best_loss:
            best_loss = test_log["total"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "history":     history,
                "config": {
                    "beta":         beta,
                    "noise_std":    noise_std,
                    "recon_loss":   recon_loss,
                    "lr":           lr,
                    "weight_decay": weight_decay,
                    "run_name":     run_name,
                },
            }, ckpt_dir / f"{run_name}.pt")

    # Save training history as JSON for easy plotting
    with open(ckpt_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test loss : {best_loss:.2f}")
    print(f"Checkpoint saved → {ckpt_dir / run_name}.pt")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model, path: str, device: str = "cpu"):
    """Load a saved checkpoint into model (in-place). Returns the checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
    return ckpt
