"""
Exp 11 — Fashion-MNIST Decoder Analysis

Runs the three most important decoder experiments on Fashion-MNIST to test
whether the key findings from MNIST generalise:

  (a) Synthesis: decode top +eigvec for each class (like Exp 01)
  (b) Causal test: does decoding v*_c + encoding back give class c?
  (c) Cross-class similarity matrix of decoder eigenvectors

Key finding: decoder cross-class similarity is even higher on Fashion-MNIST
(≈ 0.910) than MNIST (≈ 0.842), confirming the near-universal generative
direction is a property of the bilinear decoder architecture, not MNIST.

Figures saved:
    figures/fashion_mnist/exp11_synthesis.png
    figures/fashion_mnist/exp11_crossclass.png
    figures/fashion_mnist/exp11_mass_ratio.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, compute_class_means, mean_lat_norm
from visualize import FMNIST_NAMES, similarity_heatmap, save_fig

CKPT = "checkpoints/fashion_mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def main():
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

    loader = DataLoader(
        datasets.FashionMNIST(DATA, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}

    lat_means = compute_class_means(model, loader)
    scale     = mean_lat_norm(model, loader)
    classes   = sorted(mean_imgs.keys())

    # ── (a) Synthesis ─────────────────────────────────────────────────────
    print("(a) Synthesis quality:")
    synth_imgs   = {}
    pos_neg_ratio = {}
    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            neg_idx = (vals < 0).nonzero(as_tuple=True)[0]
            n_pos   = len(pos_idx)
            pos_mass = vals[pos_idx].abs().sum().item() if n_pos else 0
            neg_mass = vals[neg_idx].abs().sum().item()
            ratio    = pos_mass / (neg_mass + 1e-8)
            pos_neg_ratio[c] = ratio

            v = vecs[pos_idx[0]] * scale if n_pos else torch.zeros(model.d_latent)
            synth_imgs[c] = model.decode(v.unsqueeze(0)).squeeze(0).detach()
            print(f"  {FMNIST_NAMES[c]:<12}: n_pos={n_pos}  mass_ratio={ratio:.3f}")

    # ── (b) Causal test ───────────────────────────────────────────────────
    print("\n(b) Causal generation test:")
    results = []
    with torch.no_grad():
        for c in classes:
            mu_back, _ = model.encode(synth_imgs[c].unsqueeze(0))
            mu_back    = mu_back.squeeze(0)
            dists   = {lbl: (mu_back - m).norm().item() for lbl, m in lat_means.items()}
            nearest = min(dists, key=dists.get)
            cos     = float(torch.cosine_similarity(mu_back.unsqueeze(0),
                                                     lat_means[c].unsqueeze(0)))
            correct = nearest == c
            results.append({"class": c, "nearest": nearest, "correct": correct, "cos": cos})
            status = "✓" if correct else f"→{FMNIST_NAMES[nearest]}"
            print(f"  {FMNIST_NAMES[c]:<12}: {status:<16}  cos={cos:.3f}")

    n_correct = sum(r["correct"] for r in results)
    print(f"\n  Result: {n_correct}/10 correct  (MNIST: 3/10)")

    # ── (c) Cross-class similarity ────────────────────────────────────────
    print("\n(c) Decoder cross-class similarity:")
    top_vecs = {}
    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx   = (vals > 0).nonzero(as_tuple=True)[0]
            top_vecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_latent)

    n   = len(classes)
    mat = np.zeros((n, n))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            va, vb = top_vecs[a], top_vecs[b]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))

    off_diag = mat[mat < 1.0]
    print(f"  mean={off_diag.mean():.3f}  min={off_diag.min():.3f}  max={off_diag.max():.3f}")
    print(f"  (MNIST decoder: mean=0.842)")

    # ── Figures ────────────────────────────────────────────────────────────
    tick_lbls = [FMNIST_NAMES[c][:6] for c in classes]

    # Figure 1: synthesis + causal
    fig1, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.0),
                               gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    for col, c in enumerate(classes):
        axes[0, col].imshow(mean_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[0, col].set_title(FMNIST_NAMES[c][:6], fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(synth_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        r     = results[col]
        tick  = "✓" if r["correct"] else f"→{FMNIST_NAMES[r['nearest']][:4]}"
        color = "darkgreen" if r["correct"] else "firebrick"
        axes[1, col].set_xlabel(tick, fontsize=7, color=color)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("mean image",              fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel(f"decoded +eig\n({n_correct}/10 ✓)", fontsize=8, labelpad=4)
    fig1.suptitle(f"Exp 11a — Fashion-MNIST synthesis + causal test ({n_correct}/10 correct)",
                  fontsize=11, y=1.01)
    save_fig(fig1, "figures/fashion_mnist/exp11_synthesis.png")

    # Figure 2: cross-class heatmap
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = similarity_heatmap(ax2, mat, tick_lbls,
                             title=f"Exp 11b — Fashion-MNIST decoder cross-class\n"
                                   f"(mean={off_diag.mean():.3f}, MNIST=0.842)")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    save_fig(fig2, "figures/fashion_mnist/exp11_crossclass.png")

    # Figure 3: mass ratio bar chart
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    cmap = plt.get_cmap("tab10")
    bars = ax3.bar(range(n), [pos_neg_ratio[c] for c in classes],
                   color=[cmap(c) for c in classes], edgecolor="white")
    for bar, c in zip(bars, classes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{pos_neg_ratio[c]:.3f}", ha="center", va="bottom", fontsize=7)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(tick_lbls, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Positive / negative eigenvalue mass", fontsize=11)
    ax3.set_title("Exp 11c — Fashion-MNIST decoder mass ratio per class", fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    save_fig(fig3, "figures/fashion_mnist/exp11_mass_ratio.png")


if __name__ == "__main__":
    main()
