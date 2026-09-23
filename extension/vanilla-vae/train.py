"""Train the VanillaVAE baseline used across the extension experiments.

This is the ordinary MLP VAE against which the bilinear variants are
compared (decoder exp12/exp17, full exp05/exp12).  The provided checkpoints
(`checkpoints/mnist/model.pt` and `checkpoints/mnist/seeds/seed{1,2,3}.pt`)
were trained with exactly this recipe, which mirrors the one used for the
bilinear models: AdamW (lr 1e-3, weight decay 0.01), cosine annealing,
batch 128, 30 epochs, Gaussian input noise 0.3, BCE reconstruction, β = 1,
best-validation checkpointing.

Usage:
    python train.py                # main checkpoint (seed 0)
    python train.py --seed 2      # a seed run → checkpoints/mnist/seeds/seed2.pt

Existing checkpoint files are never overwritten.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

EPOCHS, LR, WD, BATCH, NOISE, BETA = 30, 1e-3, 0.01, 128, 0.3, 1.0


class VanillaVAE(nn.Module):
    """Standard VAE: MLP encoder and MLP decoder, no bilinear layers.

    Encoder: x(784) → Linear(256) → ReLU → Linear(512) → ReLU → μ, log σ² (10)
    Decoder: z(10)  → Linear(256) → ReLU → Linear(784) → Sigmoid

    The decoder submodule uses attribute names fc1/fc2 so that its state-dict
    keys (decoder.fc1.*, decoder.fc2.*) match the shipped checkpoints and the
    inline loaders in the experiment scripts.
    """

    def __init__(self, d_input=784, d_embed=256, d_hidden=512, d_latent=10,
                 d_dec=256):
        super().__init__()
        self.d_latent  = d_latent
        self.enc_fc1   = nn.Linear(d_input,  d_embed)
        self.enc_fc2   = nn.Linear(d_embed,  d_hidden)
        self.fc_mu     = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        class _Dec(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(d_latent, d_dec)
                self.fc2 = nn.Linear(d_dec, d_input)

            def forward(self, z):
                return torch.sigmoid(self.fc2(F.relu(self.fc1(z))))

        self.decoder = _Dec()

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterise(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def elbo(recon, x, mu, logvar, beta):
    """Per-sample negative ELBO: BCE reconstruction + β·KL to N(0, I)."""
    bce = F.binary_cross_entropy(recon.clamp(1e-6, 1 - 1e-6), x,
                                 reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce + beta * kl, bce, kl


def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Lambda(lambda t: t.view(-1))])
    train = datasets.MNIST(str(DATA), train=True,  download=True, transform=tf)
    test  = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=BATCH, shuffle=True,  num_workers=2),
            DataLoader(test,  batch_size=512,   shuffle=False, num_workers=2))


def main(seed):
    out = (HERE / "checkpoints/mnist/model.pt" if seed == 0
           else HERE / f"checkpoints/mnist/seeds/seed{seed}.pt")
    if out.exists():
        print(f"{out} exists — not overwriting.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = VanillaVAE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    train_loader, test_loader = get_loaders()

    history, best = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            noisy = (x + NOISE * torch.randn_like(x)).clamp(0, 1)
            recon, mu, logvar = model(noisy)
            loss, _, _ = elbo(recon, x, mu, logvar, BETA)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item() * x.size(0)
        sched.step()

        model.eval()
        te = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(DEVICE)
                recon, mu, logvar = model(x)
                loss, _, _ = elbo(recon, x, mu, logvar, BETA)
                te += loss.item() * x.size(0)
        tr /= len(train_loader.dataset)
        te /= len(test_loader.dataset)
        history.append({"epoch": epoch, "train": tr, "test": te})
        print(f"epoch {epoch:3d}  train {tr:9.3f}  test {te:9.3f}")

        if te < best:
            best = te
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "test_loss": te}, out)

    with open(out.with_name(out.stem + "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"best test loss {best:.3f} → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
