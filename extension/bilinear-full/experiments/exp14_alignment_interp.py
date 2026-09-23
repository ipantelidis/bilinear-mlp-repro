"""
Exp 14 — Alignment Interpretation: Where does z_dec land in latent space?

The null result from exp07 (mean enc↔dec cos = 0.007) raises a question:
if the decoder's quadratic eigvec z_dec doesn't point toward the encoder's
preferred direction, where does it point?

This experiment decodes z_dec (the decoder's generative eigvec for each class)
and re-encodes the result, measuring:
  1. Does z_back = encode(decode(z_dec)) land near the class mean?
  2. Which class does it get nearest to (causal roundtrip)?
  3. How much of z_dec is explained by the class-mean subspace?
  4. What's the projection of z_dec onto the first 2 PCA axes of each class cluster?

This is the "round-trip" test: decoder → pixel space → encoder → does it come back
to the right class?

Figure saved:
    figures/mnist/exp14_alignment_interp.png
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, decompose,
                      compute_class_means, mean_lat_norm)
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")


def _cos(a, b):
    return float(torch.dot(a.float(), b.float()) / (a.float().norm() * b.float().norm() + 1e-8))


def _nearest_class(mu, class_means):
    dists = {c: (mu - m).norm().item() for c, m in class_means.items()}
    return min(dists, key=dists.get)


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
    class_means = compute_class_means(model, loader)
    scale = mean_lat_norm(model, loader)
    classes = sorted(mean_imgs.keys())

    # PCA on full latent distribution (for projection)
    all_mu = []
    with torch.no_grad():
        for x, _ in loader:
            mu, _ = model.encode(x)
            all_mu.append(mu)
    all_mu = torch.cat(all_mu)
    mu_c = all_mu - all_mu.mean(0)
    cov = (mu_c.T @ mu_c) / (len(all_mu) - 1)
    eigs_pca, vecs_pca = torch.linalg.eigh(cov)
    pca_axes = vecs_pca[:, -2:].T  # top 2 PCA axes, shape (2, 10)

    print(f"\n{'Class':<8} {'nearest (round-trip)':>22} {'cos(z_dec, class_mean)':>24} {'correct?':>10}")
    print("-" * 68)

    z_decs = {}; z_backs = {}; decoded_imgs = {}
    roundtrip_correct = 0
    per_class_rows = []

    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            z_dec = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            z_decs[c] = z_dec

            # Decode and re-encode
            x_dec = model.decode(z_dec.unsqueeze(0)).squeeze(0)
            decoded_imgs[c] = x_dec.detach()
            mu_back, _ = model.encode(x_dec.unsqueeze(0))
            z_backs[c] = mu_back.squeeze(0)

            nearest = _nearest_class(z_backs[c], class_means)
            cos_cm  = _cos(z_dec, class_means[c])
            correct = nearest == c
            roundtrip_correct += int(correct)
            per_class_rows.append({"class": c, "nearest": nearest,
                                   "correct": bool(correct),
                                   "cos_zdec_classmean": cos_cm})
            tick = "✓" if correct else f"→{nearest}"
            print(f"  d{c}      {tick:>22} {cos_cm:>24.3f} {str(correct):>10}")

    print(f"\n  Round-trip accuracy: {roundtrip_correct}/10")
    print(f"  (compare: direct causal accuracy was 6/10 in exp04)")

    with open("figures/mnist/exp14_results.json", "w") as f:
        json.dump({"roundtrip_accuracy": roundtrip_correct,
                   "per_class": per_class_rows}, f, indent=2)

    # PCA projections of z_dec vs class means
    z_dec_mat  = torch.stack([z_decs[c]   for c in classes])   # (10, 10)
    z_back_mat = torch.stack([z_backs[c]  for c in classes])   # (10, 10)
    cm_mat     = torch.stack([class_means[c] for c in classes]) # (10, 10)

    proj_dec   = (pca_axes @ z_dec_mat.T).T.numpy()   # (10, 2)
    proj_back  = (pca_axes @ z_back_mat.T).T.numpy()  # (10, 2)
    proj_means = (pca_axes @ cm_mat.T).T.numpy()      # (10, 2)

    # Figure — 3 columns via gridspec; the middle column is a nested 2×5 grid
    # of decoded images (fixes the old hardcoded add_axes overlap with panel 3)
    n = len(classes)
    fig = plt.figure(figsize=(16, 5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.3)
    ax_pca  = fig.add_subplot(gs[0])
    gs_imgs = gs[1].subgridspec(2, 5, hspace=0.35, wspace=0.06)
    img_axes = [fig.add_subplot(gs_imgs[r, c]) for r in range(2) for c in range(5)]
    ax_heat = fig.add_subplot(gs[2])

    # Panel 1: PCA scatter of z_dec vs class means vs z_back
    cmap = plt.cm.tab10
    for ci, c in enumerate(classes):
        col = cmap(ci / 10)
        ax_pca.scatter(proj_means[ci, 0], proj_means[ci, 1], c=[col],
                       marker="o", s=100, label=f"d{c}")
        ax_pca.scatter(proj_dec[ci, 0],   proj_dec[ci, 1],   c=[col],
                       marker="*", s=200, alpha=0.8)
        ax_pca.scatter(proj_back[ci, 0],  proj_back[ci, 1],  c=[col],
                       marker="^", s=80,  alpha=0.8)
        ax_pca.annotate(f"{c}", (proj_means[ci, 0], proj_means[ci, 1]),
                        fontsize=7, ha="center", va="bottom")
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="gray", linestyle="None",
                            label="class mean (o)"),
                    Line2D([0], [0], marker="*", color="gray", linestyle="None",
                            markersize=10, label="z_dec (★)"),
                    Line2D([0], [0], marker="^", color="gray", linestyle="None",
                            label="encode(decode(z_dec)) (▲)")]
    ax_pca.legend(handles=legend_elems, fontsize=8)
    ax_pca.set_xlabel("PCA 1"); ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("Latent PCA: z_dec vs class means vs round-trip", fontsize=9)
    ax_pca.grid(True, alpha=0.3)

    # Panel 2: decoded images (decode(z_dec)), 2×5 grid
    def _clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    for ci, c in enumerate(classes):
        ax = img_axes[ci]
        ax.imshow(decoded_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        _clean(ax)
        ax.set_title(f"d{c}", fontsize=8)
    img_axes[0].set_ylabel("decode(z_dec)", fontsize=8, labelpad=4)
    img_axes[5].set_ylabel("decode(z_dec)", fontsize=8, labelpad=4)

    # Panel 3: cosine of z_back to class means (heat map)
    cos_mat = np.zeros((n, n))
    for ci, c in enumerate(classes):
        for cj, cc in enumerate(classes):
            cos_mat[ci, cj] = _cos(z_backs[c], class_means[cc])
    im = ax_heat.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(n)); ax_heat.set_xticklabels([f"d{c}" for c in classes])
    ax_heat.set_yticks(range(n)); ax_heat.set_yticklabels([f"d{c}" for c in classes])
    ax_heat.set_xlabel("class mean (target)"); ax_heat.set_ylabel("z_dec source class")
    ax_heat.set_title("cos(encode(decode(z_dec_c)), class_mean_c')", fontsize=9)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.suptitle(f"Exp 14 — Alignment interpretation: round-trip accuracy {roundtrip_correct}/10\n"
                 f"z_dec → decode → encode → nearest class", fontsize=10, y=1.01)
    save_fig(fig, "figures/mnist/exp14_alignment_interp.png")


if __name__ == "__main__":
    main()
