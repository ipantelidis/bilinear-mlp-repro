"""
Exp 05 — Decoder Synthesis Consistency Across Seeds

For five independently trained seeds, synthesise class-c images using the
Exp 01 pipeline (decode top +eigvec of Q_dec for each class mean direction).
Measure pixel-space cosine similarity between all pairs of seeds.

Key finding: mean pairwise consistency ≈ 0.998 — the synthesised images are
essentially identical across seeds, confirming the decoder has a near-universal
generative direction that is stable across training runs.

Figure saved:
    figures/mnist/exp05_consistency.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import save_fig

SEEDS = [0, 1, 2, 3, 4]
DATA  = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def _synth_images(model, mean_imgs, scale):
    imgs = {}
    with torch.no_grad():
        for c, p_star in mean_imgs.items():
            Q = get_decoder_interaction_matrix(model, p_star)
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            v = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            imgs[c] = model.decode(v.unsqueeze(0)).squeeze(0).detach()
    return imgs


def main():
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

    # Synthesise one image per class per seed
    all_synth = {}   # seed → {class → tensor(784)}
    for s in SEEDS:
        ckpt_path = f"checkpoints/mnist/seeds/seed{s}.pt"
        m = DecBilinearVAE(); load_checkpoint(m, ckpt_path); m.eval()
        scale = mean_lat_norm(m, loader)
        all_synth[s] = _synth_images(m, mean_imgs, scale)

    # Pairwise pixel-space cosine similarity per class
    classes   = sorted(mean_imgs.keys())
    pairs     = list(combinations(SEEDS, 2))
    pair_lbls = [f"s{a}-s{b}" for a, b in pairs]

    cos_matrix = np.zeros((len(classes), len(pairs)))
    for ci, c in enumerate(classes):
        for pi, (a, b) in enumerate(pairs):
            va, vb = all_synth[a][c], all_synth[b][c]
            cos_matrix[ci, pi] = float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8))

    mean_per_class = cos_matrix.mean(axis=1)
    overall_mean   = cos_matrix.mean()

    print(f"\nPairwise pixel-space cosine similarity across seeds:")
    print(f"{'Class':<8} {'mean_cos':>10}")
    for c, m in zip(classes, mean_per_class):
        print(f"  d{c}      {m:.4f}")
    print(f"\nOverall mean: {overall_mean:.4f}")

    # Figure: heatmap (classes × pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(cos_matrix, cmap="YlGn", vmin=0.9, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"d{c}" for c in classes], fontsize=9)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_lbls, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Exp 05 — Decoder synthesis consistency across seeds\n"
                 f"Mean pairwise cosine = {overall_mean:.4f}", fontsize=10)
    for ci in range(len(classes)):
        for pi in range(len(pairs)):
            ax.text(pi, ci, f"{cos_matrix[ci, pi]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if cos_matrix[ci, pi] < 0.97 else "white")
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp05_consistency.png")


if __name__ == "__main__":
    main()
