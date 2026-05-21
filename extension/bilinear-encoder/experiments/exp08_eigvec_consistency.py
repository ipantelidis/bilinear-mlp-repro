"""
Experiment 08 — Eigenvector Consistency Across Seeds
======================================================
Direct analog of Pearce et al. Figure 5A.

For each seed, compute class-mean directions from that seed's own encoder
(necessary because VAE latent spaces are not identified up to rotation across
seeds).  Then compare top eigenvectors — which live in pixel space and are
therefore directly comparable — across all C(5,2) = 10 seed pairs.

Pearce et al. report cosine similarity 0.80–0.90 for classifiers.

Outputs: figures/mnist/exp08_consistency_curve.png
"""

import sys, json
from pathlib import Path
from itertools import combinations
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
SEEDS_DIR = Path("checkpoints/mnist/seeds")
OUT       = Path("figures/mnist/exp08_consistency_curve.png")
DEVICE    = "cpu"
N_SEEDS   = 5
N_RANKS   = 10   # eigenvector ranks to plot
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def top_k_pos_eigenvectors(model, means: dict, n_ranks: int) -> dict:
    """
    For each class, return a (n_ranks, 784) tensor of top positive eigenvectors.
    Uses each seed's own class means so latent-space rotation doesn't matter.
    """
    result = {}
    for c, direction in means.items():
        Q = interaction_matrix(model, direction)
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        rows = []
        for k in range(n_ranks):
            rows.append(vecs[pos_idx[k]] if k < len(pos_idx) else torch.zeros(784))
        result[c] = torch.stack(rows)   # (n_ranks, 784)
    return result


def abs_cos(v_a: torch.Tensor, v_b: torch.Tensor) -> float:
    return float(abs(torch.dot(v_a, v_b) /
                     (v_a.norm() * v_b.norm() + 1e-8)))


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    print(f"Loading {N_SEEDS} seed models...")
    models, seed_eigvecs = [], []
    for i in range(N_SEEDS):
        m = BilinearVAE()
        load_checkpoint(m, str(SEEDS_DIR / f"seed{i}.pt"), DEVICE)
        m.eval()
        # Each seed's own class means → pixel-space eigenvectors
        means_i = class_means(m, loader, DEVICE)
        seed_eigvecs.append(top_k_pos_eigenvectors(m, means_i, N_RANKS))

    classes = sorted(seed_eigvecs[0].keys())
    pairs   = list(combinations(range(N_SEEDS), 2))

    # For each class and rank, mean |cos sim| across all 10 seed pairs
    sim_by_class = {}
    for c in classes:
        rank_sims = np.zeros((len(pairs), N_RANKS))
        for pi, (i, j) in enumerate(pairs):
            for k in range(N_RANKS):
                rank_sims[pi, k] = abs_cos(seed_eigvecs[i][c][k],
                                           seed_eigvecs[j][c][k])
        sim_by_class[c] = rank_sims.mean(axis=0)

    # Print summary
    rank1_vals = [sim_by_class[c][0] for c in classes]
    overall = float(np.mean(rank1_vals))
    print(f"\n  Rank-1 mean |cos sim| per class:")
    for c in classes:
        print(f"    digit {c}: {sim_by_class[c][0]:.3f}")
    print(f"\n  Overall rank-1: {overall:.3f} ± {np.std(rank1_vals):.3f}")
    print(f"  Pearce et al. classifier range: 0.80–0.90")

    # Figure
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ranks = np.arange(1, N_RANKS + 1)

    for c in classes:
        ax.plot(ranks, sim_by_class[c], color=cmap(c), alpha=0.35, linewidth=1.0)

    mean_curve = np.stack(list(sim_by_class.values())).mean(0)
    ax.plot(ranks, mean_curve, color="black", linewidth=2.5, label="Mean over classes")
    ax.axhspan(0.80, 0.90, alpha=0.10, color="steelblue",
               label="Pearce et al. range (classifiers)")
    ax.axhline(mean_curve[0], color="black", linestyle="--", linewidth=0.8,
               label=f"Our rank-1 mean ({mean_curve[0]:.3f})")

    ax.set_xlabel("Eigenvector rank", fontsize=11)
    ax.set_ylabel("|Cosine similarity| across seed pairs", fontsize=11)
    ax.set_title("Exp 08 — Eigenvector consistency across 5 training runs", fontsize=12)
    ax.set_xlim(1, N_RANKS); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {OUT}")
    print("Done.")
