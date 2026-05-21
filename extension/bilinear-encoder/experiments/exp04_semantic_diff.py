"""
Experiment 04 — Semantic Difference Directions
===============================================
Set μ* = mean_A − mean_B for visually confusable pairs.
Positive eigenvectors show what A has that B does not;
negative eigenvectors show the reverse.

Output: figures/mnist/exp04_semantic_diff.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.insert(0, str(Path(__file__).parent.parent))

from models   import BilinearVAE
from train    import load_checkpoint
from analysis import interaction_matrix, decompose, class_means

# ── Constants ────────────────────────────────────────────────────────────────
CKPT   = Path("checkpoints/mnist/model.pt")
OUT    = Path("figures/mnist/exp04_semantic_diff.png")
DEVICE = "cpu"
PAIRS  = [(4, 9), (1, 7), (0, 6), (3, 5)]   # (A, B) — mean_A − mean_B
N_SHOW = 4                                    # top eigenvectors per row
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run(model, means: dict) -> None:
    n_pairs = len(PAIRS)
    fig, axes = plt.subplots(
        2 * N_SHOW, n_pairs,
        figsize=(2.5 * n_pairs, 2.0 * 2 * N_SHOW),
        gridspec_kw={"hspace": 0.05, "wspace": 0.08},
    )

    for col, (a, b) in enumerate(PAIRS):
        direction = means[a] - means[b]
        direction = direction / (direction.norm() + 1e-10)
        Q = interaction_matrix(model, direction)
        vals, vecs = decompose(Q)

        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        neg_idx = (vals < 0).nonzero(as_tuple=True)[0]

        print(f"  ({a}→{b})  top pos λ: {[round(float(vals[i]),3) for i in pos_idx[:N_SHOW]]}")
        print(f"         top neg λ: {[round(float(vals[i]),3) for i in neg_idx[:N_SHOW]]}")

        vmax = vecs[:N_SHOW + 1].abs().max().item() + 1e-8

        for row in range(N_SHOW):
            # Top half: positive eigenvectors (what activates A over B)
            ax_pos = axes[row, col]
            if row < len(pos_idx):
                img = vecs[pos_idx[row]].view(28, 28).numpy()
                ax_pos.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax_pos.set_xlabel(f"λ={float(vals[pos_idx[row]]):.3f}", fontsize=7)
            ax_pos.axis("off")

            # Bottom half: negative eigenvectors (what activates B over A)
            ax_neg = axes[N_SHOW + row, col]
            if row < len(neg_idx):
                img = vecs[neg_idx[row]].view(28, 28).numpy()
                ax_neg.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax_neg.set_xlabel(f"λ={float(vals[neg_idx[row]]):.3f}", fontsize=7)
            ax_neg.axis("off")

        axes[0, col].set_title(f"mean{a} − mean{b}", fontsize=10)

    axes[0, 0].set_ylabel(f"Activates {PAIRS[0][0]}\n(λ > 0)", fontsize=9, labelpad=4)
    axes[N_SHOW, 0].set_ylabel(f"Activates {PAIRS[0][1]}\n(λ < 0)", fontsize=9, labelpad=4)

    # Dashed separator between positive and negative rows
    fig.add_artist(plt.Line2D([0, 1], [0.5, 0.5], transform=fig.transFigure,
                               color="gray", linestyle="--", linewidth=0.8))
    fig.suptitle("Exp 04 — Semantic difference directions  (μ* = mean_A − mean_B)",
                 fontsize=11, y=1.01)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    model = BilinearVAE()
    load_checkpoint(model, str(CKPT), DEVICE)
    print("Running Exp 04: Semantic Difference Directions...")

    means = class_means(model, loader, DEVICE)
    run(model, means)
    print("Done.")
