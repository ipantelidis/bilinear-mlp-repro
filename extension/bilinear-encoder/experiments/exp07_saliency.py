"""
Experiment 07 — Weight-Based Saliency Maps
===========================================
The gradient of the quadratic form f(x) = x^T Q x with respect to x is:

    s(x) = |∇_x f| = |2 Q x|

This gives per-pixel sensitivity without any backpropagation — just a
matrix-vector multiply.  A rank-k approximation uses only the top-k eigenvectors.

Outputs: figures/mnist/exp07_saliency_grid.png
         figures/mnist/exp07_saliency_rank.png
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
CKPT       = Path("checkpoints/mnist/model.pt")
OUT_GRID   = Path("figures/mnist/exp07_saliency_grid.png")
OUT_RANK   = Path("figures/mnist/exp07_saliency_rank.png")
DEVICE     = "cpu"
N_EXAMPLES = 3     # real images shown per class
RANKS      = [1, 3, 10]   # ranks shown in the progression figure
# ─────────────────────────────────────────────────────────────────────────────


def saliency_full(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Full saliency: |2 Q x|, shape (784,)."""
    return (2.0 * Q @ x).abs()


def saliency_rank_k(vals: torch.Tensor, vecs: torch.Tensor,
                    x: torch.Tensor, k: int) -> torch.Tensor:
    """Rank-k saliency: |2 Σ_{i<k} λ_i (v_i·x) v_i|, shape (784,)."""
    top_v = vecs[:k]; top_λ = vals[:k]
    grad  = 2.0 * (top_λ * (top_v @ x)) @ top_v
    return grad.abs()


@torch.no_grad()
def collect_examples(loader, n_per_class=N_EXAMPLES) -> dict:
    buckets: dict[int, list] = {}
    for x, y in loader:
        x = x.view(x.size(0), -1)
        for i, lbl in enumerate(y.tolist()):
            if len(buckets.get(lbl, [])) < n_per_class:
                buckets.setdefault(lbl, []).append(x[i])
        if all(len(v) >= n_per_class for v in buckets.values()):
            break
    return {c: torch.stack(imgs) for c, imgs in buckets.items()}


@torch.no_grad()
def plot_saliency_grid(model, means, examples) -> None:
    classes = sorted(means.keys())
    n_cls   = len(classes)
    n_rows  = N_EXAMPLES + 3   # examples + full_sal + rank3_sal + top_eig
    fig, axes = plt.subplots(n_rows, n_cls,
                             figsize=(1.7 * n_cls, 1.7 * n_rows),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    row_labels = [f"ex {i+1}" for i in range(N_EXAMPLES)] + \
                 ["full saliency", "rank-3 saliency", "top eigvec"]

    for col, c in enumerate(classes):
        Q = interaction_matrix(model, means[c])
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        v1  = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(784)
        imgs = examples[c]

        # Real images
        for i in range(N_EXAMPLES):
            axes[i, col].imshow(imgs[i].view(28, 28).numpy(), cmap="gray_r",
                                vmin=0, vmax=1)
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(f"digit {c}", fontsize=9)

        # Full saliency (mean over examples)
        sal_full = torch.stack([saliency_full(Q, x) for x in imgs]).mean(0)
        s = sal_full.view(28, 28).numpy()
        axes[N_EXAMPLES, col].imshow(s, cmap="hot", vmin=0, vmax=s.max() + 1e-8)
        axes[N_EXAMPLES, col].axis("off")

        # Rank-3 saliency
        sal_r3 = torch.stack([saliency_rank_k(vals, vecs, x, 3) for x in imgs]).mean(0)
        s3 = sal_r3.view(28, 28).numpy()
        axes[N_EXAMPLES + 1, col].imshow(s3, cmap="hot", vmin=0, vmax=s3.max() + 1e-8)
        axes[N_EXAMPLES + 1, col].axis("off")

        # Top eigenvector
        eig = v1.view(28, 28).numpy()
        vmax = max(abs(eig).max(), 1e-8)
        axes[N_EXAMPLES + 2, col].imshow(eig, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[N_EXAMPLES + 2, col].axis("off")

    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=8, rotation=90,
                                labelpad=4, va="center")

    fig.suptitle("Exp 07 — Weight-based saliency maps", fontsize=11, y=1.01)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GRID, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT_GRID}")


@torch.no_grad()
def plot_rank_progression(model, means, examples) -> None:
    classes   = sorted(means.keys())
    col_lbls  = ["original"] + [f"rank-{k}" for k in RANKS] + ["full"]
    n_cols    = len(col_lbls)
    fig, axes = plt.subplots(len(classes), n_cols,
                             figsize=(1.7 * n_cols, 1.7 * len(classes)),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})

    for row, c in enumerate(classes):
        Q = interaction_matrix(model, means[c])
        vals, vecs = decompose(Q)
        x = examples[c][0]

        axes[row, 0].imshow(x.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"d{c}", fontsize=9, rotation=0, labelpad=14, va="center")

        for ci, k in enumerate(RANKS, start=1):
            s = saliency_rank_k(vals, vecs, x, k).view(28, 28).numpy()
            axes[row, ci].imshow(s, cmap="hot", vmin=0, vmax=s.max() + 1e-8)
            axes[row, ci].axis("off")

        s_full = saliency_full(Q, x).view(28, 28).numpy()
        axes[row, -1].imshow(s_full, cmap="hot", vmin=0, vmax=s_full.max() + 1e-8)
        axes[row, -1].axis("off")

    for ci, lbl in enumerate(col_lbls):
        axes[0, ci].set_title(lbl, fontsize=9)

    fig.suptitle("Exp 07 — Saliency rank progression", fontsize=11, y=1.01)
    OUT_RANK.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_RANK, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT_RANK}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    model = BilinearVAE()
    load_checkpoint(model, str(CKPT), DEVICE)
    print("Running Exp 07: Weight-Based Saliency Maps...")

    means    = class_means(model, loader, DEVICE)
    examples = collect_examples(loader)
    plot_saliency_grid(model, means, examples)
    plot_rank_progression(model, means, examples)
    print("Done.")
