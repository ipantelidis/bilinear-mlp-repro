"""
Exp 02 — Decoder Eigenvectors: Is the Near-Universal Direction Preserved?

When the encoder is also bilinear, does the decoder's Q_dec still show a
near-universal generative direction across all class mean directions?

Key comparison:
  DecBilinearVAE (decoder only): mean ≈ 0.842
  FullBilinearVAE (this):        ?

Figures saved:
    figures/mnist/exp02_decoder_crossclass.png
    figures/mnist/exp02_decoder_synthesis.png
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
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import similarity_heatmap, save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")


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
    scale = mean_lat_norm(model, loader)
    classes = sorted(mean_imgs.keys())

    top_vecs = {}
    synth_imgs = {}
    print("Decoder synthesis:")
    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            n_pos   = len(pos_idx)
            v = vecs[pos_idx[0]] if n_pos else torch.zeros(model.d_latent)
            top_vecs[c]  = v
            synth_imgs[c] = model.decode((v * scale).unsqueeze(0)).squeeze(0).detach()
            print(f"  d{c}: n_pos={n_pos}")

    # Cross-class similarity
    n   = len(classes)
    mat = np.zeros((n, n))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            va, vb = top_vecs[a], top_vecs[b]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))

    off_diag = mat[mat < 1.0]
    print(f"\nDecoder cross-class similarity:")
    print(f"  mean={off_diag.mean():.3f}  min={off_diag.min():.3f}  max={off_diag.max():.3f}")
    print(f"  (DecBilinearVAE decoder-only: 0.842)")

    with open("figures/mnist/exp02_results.json", "w") as f:
        json.dump({"crossclass_cos_mean": float(off_diag.mean()),
                   "crossclass_cos_min":  float(off_diag.min()),
                   "crossclass_cos_max":  float(off_diag.max()),
                   "similarity_matrix":   mat.tolist()}, f, indent=2)

    # Figure 1: similarity heatmap
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    lbls = [f"d{c}" for c in classes]
    im = similarity_heatmap(ax1, mat, lbls,
                             title=f"Exp 02 — Decoder cross-class (full bilinear)\n"
                                   f"mean={off_diag.mean():.3f}  (decoder-only: 0.842)")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    fig1.tight_layout()
    save_fig(fig1, "figures/mnist/exp02_decoder_crossclass.png")

    # Figure 2: synthesised images
    fig2, axes = plt.subplots(2, n, figsize=(18, 4),
                               gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    for col, c in enumerate(classes):
        axes[0, col].imshow(mean_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[0, col].set_title(f"d{c}", fontsize=9); axes[0, col].axis("off")
        axes[1, col].imshow(synth_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Mean image",    fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("Decoded +eig",  fontsize=8, labelpad=4)
    fig2.suptitle(f"Exp 02 — Decoder synthesis (full bilinear)", fontsize=11, y=1.01)
    save_fig(fig2, "figures/mnist/exp02_decoder_synthesis.png")


if __name__ == "__main__":
    main()
