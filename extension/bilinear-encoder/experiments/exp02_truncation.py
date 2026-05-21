"""
Experiment 02 — Truncation
===========================
For each class-mean direction μ* = mean_c, reconstruct Q using only the top-k
eigenvectors and measure Pearson r between the rank-k approximation and the true
encoder activation on test images, as a function of k.

Pearce et al. find rank-3 sufficient for classifiers.  We expect higher rank
for VAE encoders because they encode full distributional geometry.

Output: figures/mnist/exp02_truncation.png
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
CKPT      = Path("checkpoints/mnist/model.pt")
OUT       = Path("figures/mnist/exp02_truncation.png")
DEVICE    = "cpu"
K_MAX     = 20      # maximum rank to sweep
N_SAMPLES = 2000    # test images for correlation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def truncation_curves(model, loader) -> dict:
    """Return {class → array of Pearson r values for rank 1..K_MAX}."""
    means = class_means(model, loader, DEVICE)

    # Collect N_SAMPLES flattened images
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.view(x.size(0), -1))
        ys.append(y)
        if sum(t.size(0) for t in xs) >= N_SAMPLES:
            break
    xs = torch.cat(xs)[:N_SAMPLES]

    curves = {}
    for label, direction in means.items():
        mu, _ = model.encode(xs)
        a_true = (mu @ direction).numpy()

        Q = interaction_matrix(model, direction)
        vals, vecs = decompose(Q)

        corrs = []
        for k in range(1, K_MAX + 1):
            # Rank-k approximation: Q_k = Σ_{i<k} λ_i v_i v_i^T
            top_vecs = vecs[:k]                            # (k, 784)
            top_vals = vals[:k]                            # (k,)
            a_approx = ((xs @ top_vecs.T) ** 2 * top_vals).sum(dim=1).numpy()
            r = float(np.corrcoef(a_true, a_approx)[0, 1])
            corrs.append(r)

        curves[label] = np.array(corrs)
        print(f"  digit {label}: r@3={corrs[2]:.3f}  r@{K_MAX}={corrs[-1]:.3f}")

    return curves


def plot(curves: dict) -> None:
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = np.arange(1, K_MAX + 1)

    for label, corr in sorted(curves.items()):
        ax.plot(ks, corr, "o-", markersize=4, linewidth=1.5,
                color=cmap(label), label=f"digit {label}")

    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8, label="r = 0.90")
    ax.set_xlabel("Rank k", fontsize=11)
    ax.set_ylabel("Pearson r  (true vs. rank-k approx)", fontsize=11)
    ax.set_title("Exp 02 — Truncation: how many eigenvectors are needed?", fontsize=12)
    ax.set_xlim(1, K_MAX); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

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
    print("Running Exp 02: Truncation...")
    curves = truncation_curves(model, loader)
    plot(curves)
    print("Done.")
