"""
Experiment 06 — Cross-Suppression Map
=======================================
For each class c, the top negative eigenvector is the pixel pattern that most
suppresses class-c activation.  We ask: does suppressing c look like activating
some other class d?  High cross-suppression means the encoder has learned an
implicit boundary between c and d.

    suppress_sim(c, d) = |cos(top_neg_c, top_pos_d)|

Outputs: figures/mnist/exp06_negative_grid.png
         figures/mnist/exp06_cross_suppression.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.insert(0, str(Path(__file__).parent.parent))

from models    import BilinearVAE
from train     import load_checkpoint
from analysis  import interaction_matrix, decompose, class_means
from visualize import plot_heatmap

# ── Constants ────────────────────────────────────────────────────────────────
CKPT   = Path("checkpoints/mnist/model.pt")
OUT_GRID = Path("figures/mnist/exp06_negative_grid.png")
OUT_MAP  = Path("figures/mnist/exp06_cross_suppression.png")
DEVICE = "cpu"
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_eigenvectors(model, means: dict) -> tuple[dict, dict]:
    """Return (pos_vecs, neg_vecs) dicts mapping class → top eigenvector."""
    pos_vecs, neg_vecs = {}, {}
    for c, direction in means.items():
        Q = interaction_matrix(model, direction)
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        neg_idx = (vals < 0).nonzero(as_tuple=True)[0]
        pos_vecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(784)
        neg_vecs[c] = vecs[neg_idx[0]] if len(neg_idx) else torch.zeros(784)
    return pos_vecs, neg_vecs


def plot_negative_grid(neg_vecs: dict, means: dict) -> None:
    labels = sorted(neg_vecs.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.5),
                             gridspec_kw={"wspace": 0.05})
    for col, c in enumerate(labels):
        img  = neg_vecs[c].view(28, 28).numpy()
        vmax = max(abs(img).max(), 1e-8)
        axes[col].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[col].set_title(f"d{c}", fontsize=9)
        axes[col].axis("off")
    axes[0].set_ylabel("Top −eig", fontsize=8, labelpad=4)
    fig.suptitle("Exp 06 — Top negative eigenvector per class (what suppresses it)",
                 fontsize=11, y=1.04)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GRID, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT_GRID}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    model = BilinearVAE()
    load_checkpoint(model, str(CKPT), DEVICE)
    print("Running Exp 06: Cross-Suppression Map...")

    means = class_means(model, loader, DEVICE)
    pos_vecs, neg_vecs = compute_eigenvectors(model, means)
    labels = sorted(means.keys())

    # Cross-suppression matrix
    n   = len(labels)
    mat = np.zeros((n, n))
    for i, c in enumerate(labels):
        for j, d in enumerate(labels):
            cos = float(torch.dot(neg_vecs[c], pos_vecs[d]) /
                        (neg_vecs[c].norm() * pos_vecs[d].norm() + 1e-10))
            mat[i, j] = abs(cos)

    # Strongest suppression pair per class (excluding self)
    print("\n  Strongest suppression pairs:")
    for i, c in enumerate(labels):
        row = mat[i].copy(); row[i] = 0
        d = labels[int(row.argmax())]
        print(f"    suppressing {c} → most activates {d}  ({mat[i, labels.index(d)]:.3f})")

    plot_negative_grid(neg_vecs, means)
    tick_labels = [str(l) for l in labels]
    plot_heatmap(
        matrix=mat,
        row_labels=tick_labels,
        col_labels=tick_labels,
        title="Exp 06 — Cross-suppression  |cos(neg_c, pos_d)|",
        out_path=OUT_MAP,
    )
    print("Done.")
