"""
Exp 03 — Encoder–Decoder Alignment

THE key experiment for the full bilinear model.  When both sides are bilinear,
we can ask: does the encoder learn to detect what the decoder generates?

For each class c:
  - Encoder eigenvec: top +eigvec of Q_in for direction e_c (pixel-space, 784-dim)
    → what pixel pattern most activates latent dimension c
  - Decoder synthesis: decode(top +eigvec of Q_dec for p*=mean_c) (pixel-space, 784-dim)
    → what image the decoder generates when steered toward class c

If these two pixel-space patterns are aligned (high cosine similarity), the
encoder and decoder have converged on a shared representation: the encoder is
sensitized to the patterns the decoder generates, and vice versa.

This is only measurable when both sides are bilinear.

Also computes: alignment between encoder eigenvec and mean class image,
and between decoder synthesis and mean class image.

Figures saved:
    figures/mnist/exp03_alignment_grid.png
    figures/mnist/exp03_alignment_scores.png
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_encoder_interaction_matrix, get_decoder_interaction_matrix,
                      decompose, mean_lat_norm)
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


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
    n = len(classes)

    enc_eigvecs  = {}   # pixel-space top +eigvec of Q_in for e_c
    dec_synthimgs = {}  # decoded top +eigvec of Q_dec for mean_c

    with torch.no_grad():
        for c in classes:
            # Encoder: direction = unit vector in latent dim c
            d = torch.zeros(model.d_latent); d[c] = 1.0
            Q_enc = get_encoder_interaction_matrix(model, d)
            v_enc, vecs_enc = decompose(Q_enc)
            pos_enc = (v_enc > 0).nonzero(as_tuple=True)[0]
            enc_eigvecs[c] = vecs_enc[pos_enc[0]] if len(pos_enc) else torch.zeros(784)

            # Decoder: direction = class mean pixel image
            Q_dec = get_decoder_interaction_matrix(model, mean_imgs[c])
            v_dec, vecs_dec = decompose(Q_dec)
            pos_dec = (v_dec > 0).nonzero(as_tuple=True)[0]
            z_dec   = vecs_dec[pos_dec[0]] * scale if len(pos_dec) else torch.zeros(model.d_latent)
            dec_synthimgs[c] = model.decode(z_dec.unsqueeze(0)).squeeze(0).detach()

    # Alignment scores
    enc_dec_cos  = [_cos(enc_eigvecs[c], dec_synthimgs[c]) for c in classes]
    enc_mean_cos = [_cos(enc_eigvecs[c], mean_imgs[c])     for c in classes]
    dec_mean_cos = [_cos(dec_synthimgs[c], mean_imgs[c])   for c in classes]

    print(f"{'Class':<8} {'enc↔dec':>10} {'enc↔mean':>10} {'dec↔mean':>10}")
    print("-" * 42)
    for c in classes:
        print(f"  d{c}      {enc_dec_cos[c]:>10.3f} {enc_mean_cos[c]:>10.3f} {dec_mean_cos[c]:>10.3f}")
    print(f"\n  Mean enc↔dec alignment: {np.mean(enc_dec_cos):.3f}")
    print(f"  Mean enc↔mean:          {np.mean(enc_mean_cos):.3f}")
    print(f"  Mean dec↔mean:          {np.mean(dec_mean_cos):.3f}")

    # Figure 1: grid [mean | enc eigvec | dec synth | difference] × 10 classes
    fig, axes = plt.subplots(n, 4, figsize=(4 * 2.0, n * 2.0),
                              gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    for row, c in enumerate(classes):
        diff = enc_eigvecs[c] - dec_synthimgs[c]
        vmax_diff = diff.abs().max().item() + 1e-8

        axes[row, 0].imshow(mean_imgs[c].view(28,28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[row, 1].imshow(enc_eigvecs[c].view(28,28).numpy(), cmap="RdBu_r",
                             vmin=-enc_eigvecs[c].abs().max().item(),
                             vmax= enc_eigvecs[c].abs().max().item())
        axes[row, 2].imshow(dec_synthimgs[c].view(28,28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[row, 3].imshow(diff.view(28,28).numpy(), cmap="RdBu_r",
                             vmin=-vmax_diff, vmax=vmax_diff)

        axes[row, 3].set_xlabel(f"cos={enc_dec_cos[c]:.3f}", fontsize=7)
        axes[row, 0].set_ylabel(f"d{c}", fontsize=9, rotation=0, labelpad=14, va="center")
        for ax in axes[row]: ax.axis("off")

    for col, title in enumerate(["mean image", "enc eigvec\n(Q_in +eig)",
                                  "dec synthesis\n(decode Q_dec +eig)", "difference"]):
        axes[0, col].set_title(title, fontsize=8)

    fig.suptitle(f"Exp 03 — Encoder–decoder alignment\n"
                 f"Mean cos(enc↔dec) = {np.mean(enc_dec_cos):.3f}", fontsize=11, y=1.01)
    save_fig(fig, "figures/mnist/exp03_alignment_grid.png")

    # Figure 2: alignment bar charts
    x   = np.arange(n)
    w   = 0.28
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x - w,   enc_dec_cos,  w, label="enc ↔ dec synthesis", color="steelblue",  alpha=0.85)
    ax2.bar(x,       enc_mean_cos, w, label="enc ↔ mean image",    color="seagreen",   alpha=0.85)
    ax2.bar(x + w,   dec_mean_cos, w, label="dec ↔ mean image",    color="darkorange", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels([f"d{c}" for c in classes])
    ax2.set_ylabel("Cosine similarity", fontsize=11)
    ax2.set_title(f"Exp 03 — Alignment scores  (mean enc↔dec = {np.mean(enc_dec_cos):.3f})",
                  fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp03_alignment_scores.png")


if __name__ == "__main__":
    main()
