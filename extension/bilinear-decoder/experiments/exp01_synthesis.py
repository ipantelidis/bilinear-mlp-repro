"""
Exp 01 — Analytical Image Synthesis

For each digit class, set p* = class mean image, eigendecompose Q_dec, and
decode the top positive eigenvector.  Shows what the decoder "wants" to generate
for each class based purely on its weights — no input data is passed through
the decoder.

Figures saved:
    figures/mnist/exp01_synthesis.png
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
OUT  = "figures/mnist/exp01_synthesis.png"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def main():
    model = DecBilinearVAE()
    load_checkpoint(model, CKPT)
    model.eval()

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

    fig, axes = plt.subplots(3, 10, figsize=(18, 5.5),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    with torch.no_grad():
        for c in range(10):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)

            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            neg_idx = (vals < 0).nonzero(as_tuple=True)[0]
            v_pos = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            v_neg = vecs[neg_idx[0]] * scale if len(neg_idx) else torch.zeros(model.d_latent)

            img_pos = model.decode(v_pos.unsqueeze(0)).squeeze(0)
            img_neg = model.decode(v_neg.unsqueeze(0)).squeeze(0)

            axes[0, c].imshow(mean_imgs[c].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes[0, c].set_title(f"d{c}", fontsize=9)
            axes[0, c].axis("off")

            axes[1, c].imshow(img_pos.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes[1, c].set_xlabel(f"λ={vals[pos_idx[0]]:.1f}" if len(pos_idx) else "none", fontsize=7)
            axes[1, c].axis("off")

            axes[2, c].imshow(img_neg.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            axes[2, c].set_xlabel(f"λ={vals[neg_idx[0]]:.1f}" if len(neg_idx) else "none", fontsize=7)
            axes[2, c].axis("off")

    axes[0, 0].set_ylabel("Mean image",   fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("Decoded +eig", fontsize=8, labelpad=4)
    axes[2, 0].set_ylabel("Decoded −eig", fontsize=8, labelpad=4)
    fig.suptitle("Exp 01 — Analytical image synthesis  (p* = class mean, decode top eigvec)",
                 fontsize=11, y=1.01)
    save_fig(fig, OUT)


if __name__ == "__main__":
    main()
