"""
Exp 10 — Full Q_dec Eigenvalue Spectrum

For each of the 10 classes, shows the complete eigenvalue spectrum of Q_dec.
Key questions:
  1. How many positive eigenvalues does Q_dec have? (rank of generative subspace)
  2. Is the rank-2 structure preserved in the full bilinear model?
  3. How does trained vs random model differ?
  4. Does the dominant positive eigenvalue dominate (large gap)?

Also builds an encoder Q_in spectrum summary (top-20 eigenvalues, since Q_in is 784x784).

Figures saved:
    figures/mnist/exp10_dec_spectrum.png
    figures/mnist/exp10_enc_spectrum_top20.png
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
from analysis import (get_decoder_interaction_matrix, get_encoder_interaction_matrix,
                      decompose)
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")


def main():
    trained = FullBilinearVAE(); load_checkpoint(trained, CKPT); trained.eval()
    rand    = FullBilinearVAE(); rand.eval()

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
    n         = len(classes)

    # ---- Decoder Q_dec spectra ----
    dec_vals_tr = {}
    dec_vals_rn = {}
    for c in classes:
        vals_tr, _ = decompose(get_decoder_interaction_matrix(trained, mean_imgs[c]))
        vals_rn, _ = decompose(get_decoder_interaction_matrix(rand,    mean_imgs[c]))
        dec_vals_tr[c] = vals_tr.numpy()
        dec_vals_rn[c] = vals_rn.numpy()

    # ---- Encoder Q_in top-20 ----
    enc_vals_tr = {}
    enc_vals_rn = {}
    for c in classes:
        d = torch.zeros(trained.d_latent); d[c] = 1.0
        v_tr, _ = decompose(get_encoder_interaction_matrix(trained, d))
        v_rn, _ = decompose(get_encoder_interaction_matrix(rand,    d))
        enc_vals_tr[c] = v_tr[:20].numpy()
        enc_vals_rn[c] = v_rn[:20].numpy()

    # ---- Summary stats ----
    print(f"\nDecoder Q_dec spectrum summary:")
    print(f"{'Class':<8} {'n_pos (tr)':>10} {'top1 (tr)':>10} {'top2 (tr)':>10} {'top1 (rn)':>10}")
    print("-" * 52)
    for c in classes:
        v = dec_vals_tr[c]
        vr = dec_vals_rn[c]
        n_pos = int((v > 0).sum())
        print(f"  d{c}      {n_pos:>10}   {v[0]:>10.3f}   {v[1]:>10.3f}   {vr[0]:>10.3f}")

    with open("figures/mnist/exp10_results.json", "w") as f:
        json.dump({"per_class": {str(c): {
                       "n_pos_trained": int((dec_vals_tr[c] > 0).sum()),
                       "n_pos_random":  int((dec_vals_rn[c] > 0).sum()),
                       "dec_spectrum_trained": dec_vals_tr[c].tolist(),
                       "dec_spectrum_random":  dec_vals_rn[c].tolist(),
                       "enc_top20_trained":    enc_vals_tr[c].tolist()}
                   for c in classes}}, f, indent=2)

    # ---- Figure 1: decoder spectra ----
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6),
                              gridspec_kw={"hspace": 0.35})
    for ci, c in enumerate(classes):
        k   = trained.d_latent
        x_k = range(k)

        # Trained
        v  = dec_vals_tr[c]
        colors = ["steelblue" if val > 0 else "indianred" for val in v]
        axes[0, ci].bar(x_k, v, color=colors, edgecolor="white", linewidth=0.5)
        axes[0, ci].axhline(0, color="black", linewidth=0.6)
        axes[0, ci].set_title(f"d{c}", fontsize=9)
        if ci == 0:
            axes[0, ci].set_ylabel("eigenvalue", fontsize=8)
        axes[0, ci].set_xticks([])
        n_pos = int((v > 0).sum())
        axes[0, ci].text(0.5, 0.98, f"+:{n_pos}", transform=axes[0, ci].transAxes,
                          ha="center", va="top", fontsize=7)

        # Random
        vr = dec_vals_rn[c]
        colorsr = ["steelblue" if val > 0 else "indianred" for val in vr]
        axes[1, ci].bar(x_k, vr, color=colorsr, edgecolor="white", linewidth=0.5)
        axes[1, ci].axhline(0, color="black", linewidth=0.6)
        axes[1, ci].set_xticks([])
        n_pos_r = int((vr > 0).sum())
        axes[1, ci].text(0.5, 0.98, f"+:{n_pos_r}", transform=axes[1, ci].transAxes,
                          ha="center", va="top", fontsize=7)
        if ci == 0:
            axes[1, ci].set_ylabel("eigenvalue", fontsize=8)

    axes[0, 0].set_ylabel("Trained\neigenvalue", fontsize=8)
    axes[1, 0].set_ylabel("Random\neigenvalue", fontsize=8)
    fig.suptitle("Exp 10 — Decoder Q_dec eigenvalue spectrum (all 10 latent dims)\n"
                 "Blue = positive (generative), Red = negative", fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp10_dec_spectrum.png")

    # ---- Figure 2: encoder Q_in top-20 ----
    fig2, axes2 = plt.subplots(2, n, figsize=(2.5 * n, 6),
                                gridspec_kw={"hspace": 0.35})
    for ci, c in enumerate(classes):
        v  = enc_vals_tr[c]
        vr = enc_vals_rn[c]
        x_k = range(len(v))

        colors  = ["steelblue" if val > 0 else "indianred" for val in v]
        colorsr = ["steelblue" if val > 0 else "indianred" for val in vr]

        axes2[0, ci].bar(x_k, v,  color=colors,  edgecolor="white", linewidth=0.4)
        axes2[1, ci].bar(x_k, vr, color=colorsr, edgecolor="white", linewidth=0.4)
        for ax in [axes2[0, ci], axes2[1, ci]]:
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks([])
        axes2[0, ci].set_title(f"d{c}", fontsize=9)
        if ci == 0:
            axes2[0, ci].set_ylabel("Trained\neigenvalue", fontsize=8)
            axes2[1, ci].set_ylabel("Random\neigenvalue", fontsize=8)

    fig2.suptitle("Exp 10 — Encoder Q_in top-20 eigenvalues per latent direction\n"
                  "Blue = positive, Red = negative", fontsize=10, y=1.01)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp10_enc_spectrum_top20.png")


if __name__ == "__main__":
    main()
