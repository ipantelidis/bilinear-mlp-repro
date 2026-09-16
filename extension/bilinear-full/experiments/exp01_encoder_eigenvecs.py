"""
Exp 01 — Encoder Eigenvectors: Is Class-Discriminativeness Preserved?

When the decoder is also bilinear, does the encoder's Q_in still produce
class-discriminative pixel-space eigenvectors?

Measures the cross-class cosine similarity of top +eigvecs from Q_in
(one per latent unit e_k, same protocol as bilinear-encoder exp03).

Key comparison:
  BilinearVAE (encoder only):  mean ≈ 0.232
  FullBilinearVAE (this):      ?

If the full model is higher, the bilinear decoder is collapsing the encoder
representations.  If similar, the encoder retains its discriminative structure
despite the bilinear decoder constraint.

Figures saved:
    figures/mnist/exp01_encoder_eigenvecs.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import get_encoder_interaction_matrix, decompose
from visualize import similarity_heatmap, save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def main():
    model = FullBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

    # Build cross-class similarity matrix: direction e_k for each class k
    n = model.d_latent
    top_vecs = {}
    with torch.no_grad():
        for k in range(n):
            d = torch.zeros(n); d[k] = 1.0
            Q = get_encoder_interaction_matrix(model, d)
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            top_vecs[k] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(Q.shape[0])

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            va, vb = top_vecs[i], top_vecs[j]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))

    off_diag = mat[mat < 1.0]
    print(f"Encoder cross-class similarity:")
    print(f"  mean={off_diag.mean():.3f}  min={off_diag.min():.3f}  max={off_diag.max():.3f}")
    print(f"  (BilinearVAE encoder-only: 0.232)")

    # Also show decoded eigenvectors for each class
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    from analysis import mean_lat_norm
    scale = mean_lat_norm(model, loader)

    fig, axes = plt.subplots(2, n, figsize=(18, 4),
                              gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    with torch.no_grad():
        for k in range(n):
            # Top +eigvec as image (784-dim, directly visualisable)
            v = top_vecs[k]
            axes[0, k].imshow(v.view(28, 28).numpy(), cmap="RdBu_r", aspect="equal")
            axes[0, k].set_title(f"e_{k}", fontsize=9); axes[0, k].axis("off")

            # Decoded version: what image does the decoder generate for this latent direction?
            v_lat = torch.zeros(n); v_lat[k] = scale
            img = model.decode(v_lat.unsqueeze(0)).squeeze(0)
            axes[1, k].imshow(img.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes[1, k].axis("off")

    axes[0, 0].set_ylabel("Enc eigvec\n(pixel space)", fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("Decoded\ne_k", fontsize=8, labelpad=4)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    lbls = [f"e_{k}" for k in range(n)]
    im = similarity_heatmap(ax2, mat, lbls,
                             title=f"Exp 01 — Encoder cross-class (full bilinear)\n"
                                   f"mean={off_diag.mean():.3f}  (encoder-only: 0.232)")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()

    fig.suptitle(f"Exp 01 — Encoder eigenvectors (full bilinear)  mean cos={off_diag.mean():.3f}",
                 fontsize=11, y=1.02)
    save_fig(fig,  "figures/mnist/exp01_encoder_eigenvecs.png")
    save_fig(fig2, "figures/mnist/exp01_encoder_crossclass.png")


if __name__ == "__main__":
    main()
