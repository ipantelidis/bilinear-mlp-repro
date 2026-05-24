"""
Exp 07 — Decoded Negative Eigenvectors (Suppressor Map)

The top negative eigenvector of Q_dec for each class is the latent direction
the decoder most strongly avoids when generating class c.  Decoding it shows
the "anti-pattern" — what the decoder produces when steered suppressively.

Also builds a cross-suppression matrix: does the suppressor of class c look
like the generator of class d?  Compared in pixel space via cosine similarity.

Figures saved:
    figures/mnist/exp07_suppressor_images.png
    figures/mnist/exp07_cross_suppression.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import similarity_heatmap, save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def main():
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

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

    pos_imgs, neg_imgs = {}, {}
    with torch.no_grad():
        for c in range(10):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)

            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            neg_idx = (vals < 0).nonzero(as_tuple=True)[0]

            v_pos = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            v_neg = vecs[neg_idx[0]] * scale if len(neg_idx) else torch.zeros(model.d_latent)

            pos_imgs[c] = model.decode(v_pos.unsqueeze(0)).squeeze(0).detach()
            neg_imgs[c] = model.decode(v_neg.unsqueeze(0)).squeeze(0).detach()

    # Figure 1: side-by-side +eig vs −eig decoded images
    fig1, axes1 = plt.subplots(3, 10, figsize=(18, 5.5),
                                gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    for c in range(10):
        axes1[0, c].imshow(mean_imgs[c].view(28,28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes1[0, c].set_title(f"d{c}", fontsize=9); axes1[0, c].axis("off")
        axes1[1, c].imshow(pos_imgs[c].view(28,28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes1[1, c].axis("off")
        axes1[2, c].imshow(neg_imgs[c].view(28,28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes1[2, c].axis("off")
    axes1[0, 0].set_ylabel("Mean image",      fontsize=8, labelpad=4)
    axes1[1, 0].set_ylabel("Generator (+eig)", fontsize=8, labelpad=4)
    axes1[2, 0].set_ylabel("Suppressor (−eig)", fontsize=8, labelpad=4)
    fig1.suptitle("Exp 07 — Decoded negative eigenvectors (suppressor maps)",
                  fontsize=11, y=1.01)
    save_fig(fig1, "figures/mnist/exp07_suppressor_images.png")

    # Figure 2: cross-suppression matrix (cos between suppressor_c and generator_d)
    n   = 10
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            va, vb = neg_imgs[i], pos_imgs[j]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im = similarity_heatmap(ax2, mat, [f"d{c}" for c in range(n)],
                             title="Exp 07 — Cross-suppression: cos(suppressor_c, generator_d)",
                             vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp07_cross_suppression.png")


if __name__ == "__main__":
    main()
