"""
Exp 07 — Latent-Space Alignment (fixed exp03)

Exp03 measured enc↔dec alignment in pixel space, which is flawed: the encoder
eigenvec is a signed ±filter in ℝ⁷⁸⁴, while the decoder synthesis is a positive
image in [0,1]⁷⁸⁴.  The DC-offset mismatch produces near-zero cosines even when
the patterns are semantically aligned.

This experiment measures alignment properly in the 10-dimensional latent space:

  z_enc[c] = encode(synthesised_enc_image[c])
      where synthesised_enc_image[c] = Q_in top +eigvec → project to [0,1]

  z_dec[c] = top +eigvec of Q_dec for mean_c
      (already in ℝ¹⁰, no decoding needed)

  alignment[c] = cos(z_enc[c], z_dec[c])

Also computes a random-direction baseline: cos(z_enc[c], random_unit_vec).

Figure saved:
    figures/mnist/exp07_latent_alignment.png
"""

import json
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
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def _cos(a, b):
    return float(torch.dot(a.float(), b.float()) / (a.float().norm() * b.float().norm() + 1e-8))


def _enc_synthesised_image(model, c, clamp=True):
    """Return a pixel-space image from the encoder Q_in eigvec for latent dim c.

    The raw eigvec is signed (RdBu-style filter). We make it a valid image by
    normalising to [0,1] so encoding gives a meaningful latent code.
    """
    d = torch.zeros(model.d_latent); d[c] = 1.0
    Q = get_encoder_interaction_matrix(model, d)
    vals, vecs = decompose(Q)
    pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
    eigvec = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_input)
    if clamp:
        # normalise signed filter to [0,1]: shift+scale
        v = eigvec - eigvec.min()
        v = v / (v.max() + 1e-8)
    else:
        v = eigvec
    return v


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

    z_enc_codes = {}    # latent code of encoder-synthesised image
    z_dec_eigvecs = {}  # decoder top eigvec (already in latent space)
    enc_synth_imgs = {} # for visualisation

    with torch.no_grad():
        for c in classes:
            # Encoder side: Q_in top eigvec → normalise to [0,1] → encode
            img = _enc_synthesised_image(model, c, clamp=True)
            enc_synth_imgs[c] = img
            mu, _ = model.encode(img.unsqueeze(0))
            z_enc_codes[c] = mu.squeeze(0)

            # Decoder side: top +eigvec of Q_dec for mean class image
            Q_dec = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals_dec, vecs_dec = decompose(Q_dec)
            pos_dec = (vals_dec > 0).nonzero(as_tuple=True)[0]
            z_dec_eigvecs[c] = vecs_dec[pos_dec[0]] * scale if len(pos_dec) else torch.zeros(model.d_latent)

    # Alignment scores
    enc_dec_cos  = [_cos(z_enc_codes[c], z_dec_eigvecs[c]) for c in classes]

    # Random baseline: average cosine between z_enc and random unit vectors
    torch.manual_seed(42)
    rand_cos = []
    for c in classes:
        r = torch.randn(model.d_latent); r = r / r.norm()
        rand_cos.append(abs(_cos(z_enc_codes[c], r)))

    # Also measure: cosine between enc latent code and class-mean latent code
    class_lat_means = {}
    all_mu = {c: [] for c in classes}
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x)
            for i, lbl in enumerate(y.tolist()):
                all_mu[lbl].append(mu[i])
    for c in classes:
        class_lat_means[c] = torch.stack(all_mu[c]).mean(0)

    enc_classmean_cos = [_cos(z_enc_codes[c], class_lat_means[c]) for c in classes]
    dec_classmean_cos = [_cos(z_dec_eigvecs[c], class_lat_means[c]) for c in classes]

    print(f"{'Class':<8} {'enc↔dec (lat)':>15} {'enc↔classμ':>12} {'dec↔classμ':>12}")
    print("-" * 52)
    for c in classes:
        print(f"  d{c}      {enc_dec_cos[c]:>15.3f} {enc_classmean_cos[c]:>12.3f} {dec_classmean_cos[c]:>12.3f}")
    print(f"\n  Mean enc↔dec (latent):  {np.mean(enc_dec_cos):.3f}   (pixel-space exp03: -0.053)")
    print(f"  Mean |random baseline|: {np.mean(rand_cos):.3f}")
    print(f"  Mean enc↔class-mean:   {np.mean(enc_classmean_cos):.3f}")
    print(f"  Mean dec↔class-mean:   {np.mean(dec_classmean_cos):.3f}")

    with open("figures/mnist/exp07_results.json", "w") as f:
        json.dump({"enc_dec_cos_mean":       float(np.mean(enc_dec_cos)),
                   "rand_baseline_abs_mean": float(np.mean(rand_cos)),
                   "enc_classmean_cos_mean": float(np.mean(enc_classmean_cos)),
                   "dec_classmean_cos_mean": float(np.mean(dec_classmean_cos)),
                   "enc_dec_cos_per_class":  enc_dec_cos}, f, indent=2)

    # Figure
    n = len(classes)
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    x = np.arange(n)
    w = 0.28
    axes.bar(x - w, enc_dec_cos,       w, label="enc ↔ dec (latent)",   color="steelblue",  alpha=0.85)
    axes.bar(x,     enc_classmean_cos,  w, label="enc ↔ class-mean μ",   color="seagreen",   alpha=0.85)
    axes.bar(x + w, dec_classmean_cos,  w, label="dec ↔ class-mean μ",   color="darkorange", alpha=0.85)
    axes.axhline(0, color="black", linewidth=0.7)
    axes.axhline(np.mean(rand_cos), color="gray", linewidth=1.0, linestyle="--",
                 label=f"|random| baseline={np.mean(rand_cos):.3f}")
    axes.set_xticks(x); axes.set_xticklabels([f"d{c}" for c in classes])
    axes.set_ylabel("Cosine similarity", fontsize=11)
    axes.set_title(f"Latent-space alignment (corrected enc↔dec measure)\n"
                   f"mean enc↔dec = {np.mean(enc_dec_cos):.3f}  "
                   f"|  enc side: encode(Q_in eigvec)  |  dec side: Q_dec eigvec",
                   fontsize=10)
    axes.legend(fontsize=9); axes.grid(True, alpha=0.3, axis="y")

    # Save encoder-synthesised images separately
    sub_fig, sub_axes = plt.subplots(1, n, figsize=(n * 1.3, 1.5))
    for ci, c in enumerate(classes):
        sub_axes[ci].imshow(enc_synth_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        sub_axes[ci].axis("off")
        sub_axes[ci].set_title(f"d{c}", fontsize=7)
    sub_fig.suptitle("Encoder Q_in top eigvec (normalised to [0,1])", fontsize=8)
    sub_fig.tight_layout()
    sub_fig.savefig("figures/mnist/exp07_enc_synth_imgs.png", dpi=120, bbox_inches="tight")
    plt.close(sub_fig)

    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp07_latent_alignment.png")


if __name__ == "__main__":
    main()
