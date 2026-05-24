"""
Exp 04 — Decoder Eigenspectrum

For each class mean direction p* = mean_c, plot the full eigenvalue spectrum
of Q_dec (10 values).  Also overlays all 10 spectra in a single panel.

Key finding: Q_dec consistently has only 2-3 positive eigenvalues (rank-2
generative subspace) and a dominant negative eigenvalue.

Figures saved:
    figures/mnist/exp04_eigenspectra.png
    figures/mnist/exp04_spectra_overlay.png
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

    all_vals = {}
    print(f"{'Class':<8} {'n_pos':>6} {'λ_max':>10} {'λ_min':>10}")
    with torch.no_grad():
        for c in range(10):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, _ = decompose(Q)
            all_vals[c] = vals
            n_pos = int((vals > 0).sum())
            print(f"  d{c}      {n_pos:>6}  {vals.max():>10.3f}  {vals.min():>10.3f}")

    # Figure 1: individual bar charts
    fig1, axes = plt.subplots(2, 5, figsize=(14, 5.5),
                               gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(range(10)):
        ax = axes[i // 5, i % 5]
        vals = all_vals[c].numpy()
        colors = ["steelblue" if v > 0 else "firebrick" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"d{c}", fontsize=9)
        ax.set_xlabel("Eigvec index (|λ| descending)", fontsize=7)
        ax.set_ylabel("λ", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig1.suptitle("Exp 04 — Decoder eigenspectra per class  (p* = class mean image)",
                  fontsize=11, y=1.02)
    save_fig(fig1, "figures/mnist/exp04_eigenspectra.png")

    # Figure 2: all spectra overlaid
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for c in range(10):
        vals = all_vals[c].numpy()
        ax2.plot(range(len(vals)), vals, "o-", color=cmap(c), linewidth=1.5,
                 markersize=4, label=f"d{c}", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_xlabel("Eigvec index (|λ| descending)", fontsize=11)
    ax2.set_ylabel("Eigenvalue λ",                  fontsize=11)
    ax2.set_title("Exp 04 — All 10 decoder eigenspectra overlaid",  fontsize=11)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp04_spectra_overlay.png")


if __name__ == "__main__":
    main()
