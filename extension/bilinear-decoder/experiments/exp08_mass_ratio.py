"""
Exp 08 — Positive/Negative Eigenvalue Mass Ratio

The decoder Q_dec has both positive (generative) and negative (suppressive)
eigenvalues.  The ratio pos_mass/neg_mass quantifies how much of the decoder's
response to each class direction is generative vs. suppressive.

Also computes this for all 784 pixel directions (p* = e_i) and maps the spatial
distribution of generative fraction across the image.

Figures saved:
    figures/mnist/exp08_mass_ratio_classes.png
    figures/mnist/exp08_mass_spatial.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def _mass_ratio(vals):
    pos = vals[vals > 0].sum().item()
    neg = vals[vals < 0].abs().sum().item()
    return pos / (neg + 1e-8)


def main():
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}

    # ── Per-class mass ratios ─────────────────────────────────────────────
    ratios = {}
    print(f"{'Class':<8} {'mass_ratio':>12}  eigenvalue_counts")
    with torch.no_grad():
        for c in range(10):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, _ = decompose(Q)
            r = _mass_ratio(vals)
            ratios[c] = r
            n_pos = int((vals > 0).sum()); n_neg = int((vals < 0).sum())
            print(f"  d{c}      {r:>12.4f}  (+{n_pos} / -{n_neg})")

    cmap = plt.get_cmap("tab10")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars = ax1.bar(range(10), [ratios[c] for c in range(10)],
                   color=[cmap(c) for c in range(10)], edgecolor="white")
    for bar, c in zip(bars, range(10)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{ratios[c]:.3f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([f"d{c}" for c in range(10)])
    ax1.set_ylabel("Positive / negative eigenvalue mass", fontsize=11)
    ax1.set_title("Exp 08 — Decoder mass ratio per class", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.tight_layout()
    save_fig(fig1, "figures/mnist/exp08_mass_ratio_classes.png")

    # ── Spatial mass map (all pixel directions) ───────────────────────────
    spatial_ratio = np.zeros(784)
    with torch.no_grad():
        for i in range(784):
            p = torch.zeros(784); p[i] = 1.0
            Q = get_decoder_interaction_matrix(model, p)
            vals, _ = decompose(Q)
            spatial_ratio[i] = _mass_ratio(vals)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    im = ax2.imshow(spatial_ratio.reshape(28, 28), cmap="RdYlGn", aspect="equal")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("Exp 08 — Spatial generative fraction\n(pos/neg mass per pixel direction)",
                  fontsize=10)
    ax2.axis("off")
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp08_mass_spatial.png")


if __name__ == "__main__":
    main()
