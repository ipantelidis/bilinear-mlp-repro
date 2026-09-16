"""
Exp 16 — Truncated Reconstruction (Rank-k Subspace Test)

Tests the rank-2 generative subspace hypothesis: if Q_dec consistently has only
2-3 meaningful positive eigenvalues, then reconstructing using only the top-k
eigenvectors should plateau at k=2 or k=3.

Protocol:
  1. For each class c, build Q_dec conditioned on the class mean image
  2. Sort eigenvectors by eigenvalue magnitude (abs)
  3. For k = 1, 2, 3, 5, 10 (= full latent dim): use only top-k eigenvectors
     z_k = sum of top-k |val_i| * vec_i  (scaled to match mean latent norm)
  4. Decode z_k and compute MSE to the class mean image
  5. Also compare to: z = class mean mu, z = full decoder output

Also shows the truncated images as a grid.

Figure saved:
    figures/mnist/exp16_truncated_recon.png
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, decompose, mean_lat_norm)
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"
K_VALS = [1, 2, 3, 5, 10]


def _truncated_decode(model, mean_img, k, scale):
    """
    Decode using z built from top-k eigenvectors of Q_dec sorted by eigenvalue magnitude.

    k=1: only the largest-|λ| eigvec
    k=2: top 2 largest-|λ| eigvecs, combined
    ...

    The eigvecs are combined as a weighted sum using sign(λ_i)*scale for each.
    This tests whether the dominant (largest |λ|) directions capture reconstruction quality.
    """
    Q = get_decoder_interaction_matrix(model, mean_img)
    vals, vecs = decompose(Q)  # sorted by |eigenvalue| descending; vecs[i] is eigvec i

    n_use = min(k, len(vals))
    # Build z as weighted combination: use eigvec_i scaled by sign(val_i)*scale/sqrt(n_use)
    z = torch.zeros(model.d_latent)
    for i in range(n_use):
        sign_val = 1.0 if vals[i] >= 0 else -1.0
        z = z + vecs[i] * sign_val * scale / (n_use ** 0.5)

    with torch.no_grad():
        img = model.decode(z.unsqueeze(0)).squeeze(0).detach()
    return img, z


def main():
    model = FullBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

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
    classes   = sorted(mean_imgs.keys())
    scale     = mean_lat_norm(model, loader)

    # Compute truncated reconstructions for each class and k
    recon_imgs = {}   # (c, k) -> image
    mse_table  = {}   # (c, k) -> mse

    for c in classes:
        recon_imgs[c] = {}
        mse_table[c]  = {}
        for k in K_VALS:
            img, _ = _truncated_decode(model, mean_imgs[c], k, scale)
            recon_imgs[c][k] = img
            mse_table[c][k]  = ((img - mean_imgs[c])**2).mean().item()

    # Also: full model decode (standard reconstruction)
    with torch.no_grad():
        for c in classes:
            recon, mu, _ = model(mean_imgs[c].unsqueeze(0))
            recon_imgs[c]["model"] = recon.squeeze(0).detach()
            mse_table[c]["model"]  = ((recon.squeeze(0) - mean_imgs[c])**2).mean().item()

    # Print table
    print(f"\nMSE to class-mean image by truncation rank:")
    k_labels = K_VALS + ["model"]
    print(f"{'Class':<8} " + "  ".join(f"{'k='+str(k):>8}" if k != "model" else f"{'full':>8}" for k in k_labels))
    print("-" * (8 + 10 * len(k_labels)))
    for c in classes:
        row = f"  d{c}     "
        for k in k_labels:
            row += f"  {mse_table[c][k]:>8.4f}"
        print(row)

    # Check n_positive eigenvalues per class
    n_pos_per_class = {}
    print(f"\nNumber of positive Q_dec eigenvalues per class:")
    for c in classes:
        Q = get_decoder_interaction_matrix(model, mean_imgs[c])
        vals, _ = decompose(Q)
        n_pos = int((vals > 0).sum())
        n_pos_per_class[c] = n_pos
        print(f"  d{c}: {n_pos} positive eigenvalues")

    with open("figures/mnist/exp16_results.json", "w") as f:
        json.dump({"mse_by_class_and_k": {str(c): {str(k): mse_table[c][k]
                                                    for k in mse_table[c]}
                                          for c in classes},
                   "mean_mse_by_k": {str(k): float(np.mean([mse_table[c][k] for c in classes]))
                                     for k in K_VALS + ["model"]},
                   "n_pos_per_class": {str(c): n for c, n in n_pos_per_class.items()}},
                  f, indent=2)

    # Figure 1: MSE curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: MSE vs k per class
    for c in classes:
        mse_vals = [mse_table[c][k] for k in K_VALS]
        axes[0].plot(K_VALS, mse_vals, "o-", label=f"d{c}", alpha=0.7)
    axes[0].set_xlabel("Top-k eigenvectors used")
    axes[0].set_ylabel("MSE to class-mean image")
    axes[0].set_title("Truncated Q_dec reconstruction quality\nvs number of eigenvectors", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, alpha=0.3)

    # Panel 2: mean MSE across classes at each k
    mean_mse = [np.mean([mse_table[c][k] for c in classes]) for k in K_VALS]
    full_mse = np.mean([mse_table[c]["model"] for c in classes])
    axes[1].plot(K_VALS, mean_mse, "o-", color="steelblue", linewidth=2, label="truncated eigvec")
    axes[1].axhline(full_mse, color="seagreen", linewidth=1.5, linestyle="--",
                    label=f"full model decode (MSE={full_mse:.4f})")
    axes[1].set_xlabel("Top-k eigenvectors used")
    axes[1].set_ylabel("Mean MSE across 10 classes")
    axes[1].set_title("Mean truncated MSE vs full model\n(rank saturation point)", fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Exp 16 — Truncated Q_dec reconstruction (rank-k subspace test)", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp16_truncated_recon.png")

    # Figure 2: image grid (k=1, 2, 3, 5, 10 vs mean image)
    n_k    = len(K_VALS)
    n_cls  = len(classes)
    fig2, axes2 = plt.subplots(n_k + 1, n_cls,
                                figsize=(2.0 * n_cls, 2.2 * (n_k + 1)),
                                gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    for ci, c in enumerate(classes):
        axes2[0, ci].imshow(mean_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes2[0, ci].axis("off")
        axes2[0, ci].set_title(f"d{c}", fontsize=9)
        for ri, k in enumerate(K_VALS):
            axes2[ri + 1, ci].imshow(recon_imgs[c][k].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes2[ri + 1, ci].axis("off")

    axes2[0, 0].set_ylabel("mean\nimage", fontsize=8, labelpad=4)
    for ri, k in enumerate(K_VALS):
        axes2[ri + 1, 0].set_ylabel(f"k={k}", fontsize=8, labelpad=4)

    fig2.suptitle("Exp 16 — Truncated Q_dec synthesis: top-k eigenvectors",
                  fontsize=10, y=1.01)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp16_truncated_images.png")


if __name__ == "__main__":
    main()
