"""
train.py — Training utilities for BilinearVAE.

Provides:
    vae_loss         ELBO = BCE reconstruction + β·KL
    train_epoch      Single training epoch
    evaluate         Compute loss on a data loader without gradients
    save_checkpoint  Save model weights + epoch to disk
    load_checkpoint  Load checkpoint into model in-place

CLI (run from bilinear-encoder/) — retrains the shipped checkpoints from
scratch with the recipe used to produce them (epochs=30, lr=1e-3, wd=0.01,
β=1, noise_std=0.3, batch 128, AdamW + cosine LR, best-test checkpointing):

    python train.py --dataset mnist              # checkpoints/mnist/model.pt
    python train.py --dataset fashion_mnist      # checkpoints/fashion_mnist/model.pt
    python train.py --dataset mnist --seed 3     # checkpoints/mnist/seeds/seed3.pt

Existing checkpoint files are never overwritten (the run is refused).
Optional overrides, mainly for smoke tests: --epochs N --ckpt-dir DIR.
"""

import argparse
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI — retrain the shipped checkpoints from scratch
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST", "kmnist": "KMNIST"}


def _build_loaders(dataset: str, data_dir, batch_size: int = 128):
    """Standard loaders: torchvision, flattened [0,1] images."""
    from torchvision import datasets, transforms

    cls = getattr(datasets, _DATASETS[dataset])
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = cls(str(data_dir), train=True,  download=True, transform=tfm)
    te = cls(str(data_dir), train=False, download=True, transform=tfm)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=512, shuffle=False,
                       num_workers=2, pin_memory=True))


def _run_epoch(model, loader, optimizer, device,
               beta=1.0, noise_std=0.0, training=True):
    """One pass with optional Gaussian input noise (regularisation during
    training only; the model reconstructs the clean x)."""
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            x_in = ((x + noise_std * torch.randn_like(x)).clamp(0, 1)
                    if training and noise_std > 0 else x)
            recon, mu, logvar = model(x_in)
            loss = vae_loss(recon, x, mu, logvar, beta)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader.dataset)


def _cli():
    from models import BilinearVAE

    ap = argparse.ArgumentParser(
        description="Retrain a BilinearVAE checkpoint from scratch "
                    "(existing files are never overwritten).")
    ap.add_argument("--dataset", choices=sorted(_DATASETS), default="mnist")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed run → checkpoints/<dataset>/seeds/seedN.pt "
                         "(omit for the main model.pt)")
    ap.add_argument("--d-latent", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ckpt-dir", default=None,
                    help="override the checkpoint directory (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here.parents[1] / "data"                 # <repo>/data

    if a.seed is not None:
        ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else here / f"checkpoints/{a.dataset}/seeds"
        run_name = f"seed{a.seed}"
        torch.manual_seed(a.seed)
    else:
        ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else here / f"checkpoints/{a.dataset}"
        run_name = "model"

    out = ckpt_dir / f"{run_name}.pt"
    if out.exists():
        print(f"REFUSED: {out} already exists — delete it first to retrain.")
        return

    model = BilinearVAE(d_latent=a.d_latent).to(a.device)
    tr, te = _build_loaders(a.dataset, data_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    best = float("inf")
    print(f"\nTraining {run_name} ({a.dataset})  β=1.0  noise=0.3  epochs={a.epochs}")
    print(f"{'Epoch':>5}  {'Train':>10}  {'Test':>10}")
    print("-" * 30)
    for epoch in range(1, a.epochs + 1):
        tr_l = _run_epoch(model, tr, optimizer, a.device,
                          beta=1.0, noise_std=0.3, training=True)
        te_l = _run_epoch(model, te, optimizer, a.device,
                          beta=1.0, noise_std=0.0, training=False)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5}  {tr_l:>10.2f}  {te_l:>10.2f}")
        if te_l < best:
            best = te_l
            save_checkpoint(model, str(out), epoch)
    print(f"\nBest test loss: {best:.2f}  →  {out}")


if __name__ == "__main__":
    _cli()
