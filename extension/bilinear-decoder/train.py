"""
train.py — Loss function and training loop for the bilinear decoder VAE.

    L(x; β) = E[log p(x|z)]  −  β · KL(q(z|x) || p(z))

Gaussian input noise is applied during training following Pearce et al. (2025).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def elbo_loss(recon, x, mu, logvar, beta=1.0):
    recon_l = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_l    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_l + beta * kl_l, recon_l, kl_l


def run_epoch(model, loader, optimizer, device,
              beta=1.0, noise_std=0.0, training=True):
    model.train(training)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            x_in = (x + noise_std * torch.randn_like(x)).clamp(0, 1) if (training and noise_std > 0) else x
            recon, mu, logvar = model(x_in)
            loss, recon_l, kl_l = elbo_loss(recon, x, mu, logvar, beta)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            totals["total"] += loss.item()
            totals["recon"] += recon_l.item()
            totals["kl"]    += kl_l.item()
            n += x.size(0)
    return {k: v / n for k, v in totals.items()}


def train(model, train_loader, test_loader, *,
          epochs=30, lr=1e-3, weight_decay=0.01, beta=1.0,
          noise_std=0.3, device="cpu",
          checkpoint_dir="checkpoints", run_name="model"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history   = []
    best_loss = float("inf")

    print(f"\nTraining {run_name}  β={beta}  noise={noise_std}  epochs={epochs}")
    print(f"{'Epoch':>5}  {'Train':>10}  {'Test':>10}  {'KL':>8}")
    print("-" * 40)

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, beta=beta, noise_std=noise_std, training=True)
        te = run_epoch(model, test_loader,  optimizer, device, beta=beta, noise_std=0.0,       training=False)
        scheduler.step()
        history.append({"epoch": epoch, "train_total": tr["total"], "test_total": te["total"], "test_kl": te["kl"]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5}  {tr['total']:>10.2f}  {te['total']:>10.2f}  {te['kl']:>8.2f}")
        if te["total"] < best_loss:
            best_loss = te["total"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "history": history},
                       ckpt_dir / f"{run_name}.pt")

    with open(ckpt_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest test loss: {best_loss:.2f}  →  {ckpt_dir / run_name}.pt")
    return history


def load_checkpoint(model, path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device); model.eval()
    print(f"Loaded {path}  (epoch {ckpt['epoch']})")
    return ckpt
