"""
Experiment 03 — Cross-Class Eigenvector Similarity
====================================================
For every pair of classes (A, B), compute |cos(v1_A, v1_B)| between their top
positive eigenvectors (μ* = mean_c).  High similarity means the encoder uses
similar pixel patterns to represent both classes — mirroring visual confusability.

Output: figures/mnist/exp03_cross_class.png
"""

import sys
from pathlib import Path
import numpy as np
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
OUT    = Path("figures/mnist/exp03_cross_class.png")
DEVICE = "cpu"
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def top_pos_eigenvector(model, direction: torch.Tensor) -> torch.Tensor:
    """Return the top positive eigenvector for a given latent direction."""
    Q = interaction_matrix(model, direction)
    vals, vecs = decompose(Q)
    pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
    return vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(784)


@torch.no_grad()
def build_similarity_matrix(model, loader) -> tuple[list, np.ndarray]:
    means  = class_means(model, loader, DEVICE)
    labels = sorted(means.keys())
    top_vecs = {c: top_pos_eigenvector(model, means[c]) for c in labels}

    n   = len(labels)
    mat = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            v_a, v_b = top_vecs[a], top_vecs[b]
            cos = float(torch.dot(v_a, v_b) /
                        (v_a.norm() * v_b.norm() + 1e-10))
            mat[i, j] = abs(cos)

    return labels, mat


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    model = BilinearVAE()
    load_checkpoint(model, str(CKPT), DEVICE)
    print("Running Exp 03: Cross-Class Similarity...")

    labels, mat = build_similarity_matrix(model, loader)
    tick_labels = [str(l) for l in labels]

    plot_heatmap(
        matrix=mat,
        row_labels=tick_labels,
        col_labels=tick_labels,
        title="Exp 03 — Cross-class similarity  |cos(v1_A, v1_B)|",
        out_path=OUT,
    )

    # Print most similar pairs
    print("\n  Top-5 most similar pairs:")
    pairs = [(mat[i, j], labels[i], labels[j])
             for i in range(len(labels)) for j in range(i + 1, len(labels))]
    for sim, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"    ({a}, {b}): {sim:.3f}")
    print("Done.")
