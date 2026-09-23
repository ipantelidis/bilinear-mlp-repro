"""
train.py — Loss function and training loop for the bilinear decoder VAE.

    L(x; β) = E[log p(x|z)]  −  β · KL(q(z|x) || p(z))

Gaussian input noise is applied during training following Pearce et al. (2025).

CLI (run from bilinear-decoder/) — retrains the shipped checkpoints from
scratch with the recipe used to produce them (epochs=30, lr=1e-3, wd=0.01,
β=1, noise_std=0.3, batch 128, AdamW + cosine LR, best-test checkpointing):

    python train.py --dataset mnist                      # checkpoints/mnist/model.pt
    python train.py --dataset fashion_mnist              # checkpoints/fashion_mnist/model.pt
    python train.py --dataset mnist --seed 3             # checkpoints/mnist/seeds/seed3.pt
    python train.py --dataset mnist --d-latent 20 --seed 0
                       # checkpoints/mnist/latent_sweep/d20_seed0.pt

Existing checkpoint files are never overwritten (the run is refused).
Optional overrides, mainly for smoke tests: --epochs N --ckpt-dir DIR.
"""

import argparse
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI — retrain the shipped checkpoints from scratch
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST", "kmnist": "KMNIST"}


def _build_loaders(dataset, data_dir, batch_size=128, generator=None):
    """Standard loaders: torchvision, flattened [0,1] images."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    cls = getattr(datasets, _DATASETS[dataset])
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = cls(str(data_dir), train=True,  download=True, transform=tfm)
    te = cls(str(data_dir), train=False, download=True, transform=tfm)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       generator=generator, num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=512, shuffle=False,
                       num_workers=2, pin_memory=True))


def _cli():
    from models import DecBilinearVAE

    ap = argparse.ArgumentParser(
        description="Retrain a DecBilinearVAE checkpoint from scratch "
                    "(existing files are never overwritten).")
    ap.add_argument("--dataset", choices=sorted(_DATASETS), default="mnist")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed run → checkpoints/<dataset>/seeds/seedN.pt "
                         "(omit for the main model.pt)")
    ap.add_argument("--d-latent", type=int, default=10,
                    help="latent dimensionality; values ≠ 10 go to "
                         "checkpoints/mnist/latent_sweep/d{D}_seed{S}.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ckpt-dir", default=None,
                    help="override the checkpoint directory (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here.parents[1] / "data"                 # <repo>/data

    generator = None
    if a.d_latent != 10:
        # Latent-dimension sweep runs (exp16 protocol): mnist only, seeded
        # DataLoader shuffling, d{D}_seed{S} naming.
        if a.dataset != "mnist":
            ap.error("--d-latent sweeps are defined for --dataset mnist only")
        seed = a.seed if a.seed is not None else 0
        ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else here / "checkpoints/mnist/latent_sweep"
        run_name = f"d{a.d_latent}_seed{seed}"
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
    elif a.seed is not None:
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

    model = DecBilinearVAE(d_latent=a.d_latent)
    tr, te = _build_loaders(a.dataset, data_dir, generator=generator)
    train(model, tr, te, epochs=a.epochs, lr=1e-3, weight_decay=0.01,
          beta=1.0, noise_std=0.3, device=a.device,
          checkpoint_dir=str(ckpt_dir), run_name=run_name)


if __name__ == "__main__":
    _cli()
