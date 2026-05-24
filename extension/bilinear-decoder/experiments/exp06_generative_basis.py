"""
Exp 06 — Complete Generative Basis Per Class

For each digit class, Q_dec has exactly 2-3 positive eigenvectors.
Decode ALL positive eigenvectors — this is the complete weight-based generative
vocabulary of the decoder for each class direction.

Figure saved:
    figures/mnist/exp06_generative_basis.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"
MAX_POS = 4   # columns: up to 4 positive eigvecs per class


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
    scale = mean_lat_norm(model, loader)

    # Layout: rows = classes, cols = mean + up to MAX_POS decoded +eigvecs
    n_cols = 1 + MAX_POS
    fig, axes = plt.subplots(10, n_cols, figsize=(n_cols * 1.8, 10 * 1.8),
                              gridspec_kw={"hspace": 0.05, "wspace": 0.04})

    print(f"{'Class':<8} {'n_pos':>6}  eigenvalues")
    with torch.no_grad():
        for c in range(10):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            n_pos   = len(pos_idx)
            print(f"  d{c}      {n_pos:>6}  {[round(float(vals[i]),2) for i in pos_idx]}")

            # Col 0: mean image
            axes[c, 0].imshow(mean_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes[c, 0].set_ylabel(f"d{c}", fontsize=9, rotation=0, labelpad=14, va="center")
            axes[c, 0].axis("off")

            # Cols 1…: decoded positive eigvecs
            for k in range(MAX_POS):
                ax = axes[c, k + 1]
                if k < n_pos:
                    v   = vecs[pos_idx[k]] * scale
                    img = model.decode(v.unsqueeze(0)).squeeze(0)
                    ax.imshow(img.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
                    ax.set_xlabel(f"λ={vals[pos_idx[k]]:.1f}", fontsize=7)
                else:
                    ax.set_facecolor("#f0f0f0")
                ax.axis("off")

    for k, title in enumerate(["mean image"] + [f"+eig {k+1}" for k in range(MAX_POS)]):
        axes[0, k].set_title(title, fontsize=8)

    fig.suptitle("Exp 06 — Complete generative basis (all positive eigvecs decoded)",
                 fontsize=11, y=1.005)
    save_fig(fig, "figures/mnist/exp06_generative_basis.png")


if __name__ == "__main__":
    main()
