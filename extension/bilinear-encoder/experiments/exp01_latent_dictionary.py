"""
Experiment 01 — Latent Dictionary
==================================
For each latent dimension k, set μ* = e_k and compute which pixel patterns
activate (top positive eigenvector) or suppress (top negative eigenvector)
that dimension.  All analysis is from weights alone — no data needed.

Output: figures/mnist/exp01_latent_dictionary.png
"""

import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

from models   import BilinearVAE
from train    import load_checkpoint
from analysis import interaction_matrix, decompose

# ── Constants ────────────────────────────────────────────────────────────────
CKPT    = Path("checkpoints/mnist/model.pt")
OUT     = Path("figures/mnist/exp01_latent_dictionary.png")
DEVICE  = "cpu"
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np


def run(model: BilinearVAE) -> None:
    d = model.d_latent
    pos_imgs, neg_imgs = [], []
    pos_vals, neg_vals = [], []

    for k in range(d):
        e_k = torch.zeros(d); e_k[k] = 1.0
        Q = interaction_matrix(model, e_k)
        vals, vecs = decompose(Q)

        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        neg_idx = (vals < 0).nonzero(as_tuple=True)[0]

        pos_imgs.append(vecs[pos_idx[0]].view(28, 28).numpy() if len(pos_idx) else np.zeros((28, 28)))
        neg_imgs.append(vecs[neg_idx[0]].view(28, 28).numpy() if len(neg_idx) else np.zeros((28, 28)))
        pos_vals.append(float(vals[pos_idx[0]]) if len(pos_idx) else 0.0)
        neg_vals.append(float(vals[neg_idx[0]]) if len(neg_idx) else 0.0)

    fig, axes = plt.subplots(2, d, figsize=(d * 1.4, 3.4),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.05})
    for k in range(d):
        for row, (img, lam) in enumerate([(pos_imgs[k], pos_vals[k]),
                                          (neg_imgs[k], neg_vals[k])]):
            vmax = max(abs(img).max(), 1e-8)
            axes[row, k].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, k].set_title(f"z{k}\nλ={lam:.2f}", fontsize=7)
            axes[row, k].axis("off")

    axes[0, 0].set_ylabel("Activates z_k\n(λ > 0)", fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("Suppresses z_k\n(λ < 0)", fontsize=8, labelpad=4)
    fig.suptitle("Exp 01 — Latent dictionary  (μ* = e_k, weights only)",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT}")


if __name__ == "__main__":
    model = BilinearVAE()
    load_checkpoint(model, str(CKPT), DEVICE)
    print("Running Exp 01: Latent Dictionary...")
    run(model)
    print("Done.")
