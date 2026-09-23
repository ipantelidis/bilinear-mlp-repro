"""
Exp 06 — Cross-Seed Consistency

For 5 independently trained seeds, measure:
  (a) Decoder synthesis consistency: are the synthesised images stable across seeds?
      (same metric as bilinear-decoder exp05, expected ~0.993)
  (b) Encoder eigenvec consistency: are the encoder Q_in top eigvecs stable?
      (same metric as bilinear-encoder, expected ~0.8+)
  (c) Alignment consistency: is the enc↔dec cosine score stable across seeds?

Figure saved:
    figures/mnist/exp06_consistency.png
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_encoder_interaction_matrix, get_decoder_interaction_matrix,
                      decompose, mean_lat_norm)
from visualize import save_fig

SEEDS = [0, 1, 2, 3, 4]
DATA  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")


def _cos(a, b):
    return float(torch.dot(a.float(), b.float()) / (a.norm() * b.norm() + 1e-8))


def _get_dec_synth(model, mean_imgs, scale):
    imgs = {}
    with torch.no_grad():
        for c, p_star in mean_imgs.items():
            Q = get_decoder_interaction_matrix(model, p_star)
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            z = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            imgs[c] = model.decode(z.unsqueeze(0)).squeeze(0).detach()
    return imgs


def _get_enc_eigvec(model):
    vecs = {}
    with torch.no_grad():
        for k in range(model.d_latent):
            d = torch.zeros(model.d_latent); d[k] = 1.0
            Q = get_encoder_interaction_matrix(model, d)
            vals, v = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            vecs[k] = v[pos_idx[0]] if len(pos_idx) else torch.zeros(Q.shape[0])
    return vecs


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
    classes   = sorted(mean_imgs.keys())

    all_dec  = {}
    all_enc  = {}
    all_align = {}

    for s in SEEDS:
        m = FullBilinearVAE()
        load_checkpoint(m, f"checkpoints/mnist/seeds/seed{s}.pt")
        m.eval()
        scale = mean_lat_norm(m, loader)
        all_dec[s]   = _get_dec_synth(m, mean_imgs, scale)
        all_enc[s]   = _get_enc_eigvec(m)
        # alignment score per class
        all_align[s] = {c: _cos(all_enc[s][c], all_dec[s][c]) for c in classes}

    pairs = list(combinations(SEEDS, 2))

    # (a) Decoder synthesis consistency
    dec_cos = np.zeros((len(classes), len(pairs)))
    for ci, c in enumerate(classes):
        for pi, (a, b) in enumerate(pairs):
            dec_cos[ci, pi] = abs(_cos(all_dec[a][c], all_dec[b][c]))

    # (b) Encoder eigvec consistency
    enc_cos = np.zeros((len(classes), len(pairs)))
    for ci, c in enumerate(classes):
        for pi, (a, b) in enumerate(pairs):
            enc_cos[ci, pi] = abs(_cos(all_enc[a][c], all_enc[b][c]))

    # (c) Alignment score stability
    align_scores = np.array([[all_align[s][c] for c in classes] for s in SEEDS])

    print(f"\nDecoder synthesis consistency (mean over pairs):")
    print(f"  mean={dec_cos.mean():.4f}  (DecBilinearVAE alone: 0.993)")
    print(f"\nEncoder eigvec consistency:")
    print(f"  mean={enc_cos.mean():.4f}")
    print(f"\nAlignment score (enc↔dec) per seed, mean over classes:")
    for s in SEEDS:
        print(f"  seed{s}: {np.mean(list(all_align[s].values())):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pair_lbls = [f"s{a}-s{b}" for a, b in pairs]

    # Panel 1: decoder consistency heatmap
    im1 = axes[0].imshow(dec_cos, cmap="YlGn", vmin=0.8, vmax=1.0, aspect="auto")
    axes[0].set_yticks(range(len(classes))); axes[0].set_yticklabels([f"d{c}" for c in classes])
    axes[0].set_xticks(range(len(pairs))); axes[0].set_xticklabels(pair_lbls, rotation=45, ha="right", fontsize=7)
    axes[0].set_title(f"(a) Decoder synthesis\nconsistency  mean={dec_cos.mean():.3f}", fontsize=9)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: encoder eigvec consistency heatmap
    im2 = axes[1].imshow(enc_cos, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_yticks(range(len(classes))); axes[1].set_yticklabels([f"d{c}" for c in classes])
    axes[1].set_xticks(range(len(pairs))); axes[1].set_xticklabels(pair_lbls, rotation=45, ha="right", fontsize=7)
    axes[1].set_title(f"(b) Encoder eigvec\nconsistency  mean={enc_cos.mean():.3f}", fontsize=9)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: alignment score box per class
    axes[2].boxplot([align_scores[:, ci] for ci in range(len(classes))],
                    labels=[f"d{c}" for c in classes])
    axes[2].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[2].set_ylabel("enc ↔ dec alignment (cos)", fontsize=10)
    axes[2].set_title(f"(c) Alignment score stability\nacross seeds", fontsize=9)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 06 — Cross-seed consistency (full bilinear)", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp06_consistency.png")


if __name__ == "__main__":
    main()
