"""
Exp 02 — Pixel Generative Fields

For a 10×10 spatial grid of pixel locations: set p* = e_i (canonical pixel
basis vector) and decode the top positive eigenvector of Q_dec.
Shows which pixel locations drive simple vs. complex latent responses.

Two panels:
  (a) Grid of decoded images — the "generative field" of each output pixel.
  (b) Effective-rank heatmap — how many latent dimensions each pixel mobilises.

Figures saved:
    figures/mnist/exp02_pixel_fields_grid.png
    figures/mnist/exp02_pixel_fields_rank.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"

GRID_ROWS, GRID_COLS = 10, 10   # sample every (28/10) ≈ 3 pixels
PIXEL_ROWS = np.linspace(0, 27, GRID_ROWS, dtype=int)
PIXEL_COLS = np.linspace(0, 27, GRID_COLS, dtype=int)


def _eff_rank(vals):
    p = vals.abs(); p = p / p.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))


def main():
    model = DecBilinearVAE()
    load_checkpoint(model, CKPT)
    model.eval()

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)
    scale = mean_lat_norm(model, loader)

    decoded_grid = np.zeros((GRID_ROWS, GRID_COLS, 28, 28))
    rank_map     = np.zeros((GRID_ROWS, GRID_COLS))

    with torch.no_grad():
        for ri, row in enumerate(PIXEL_ROWS):
            for ci, col in enumerate(PIXEL_COLS):
                pixel_idx = int(row * 28 + col)
                p_star = torch.zeros(784)
                p_star[pixel_idx] = 1.0

                Q = get_decoder_interaction_matrix(model, p_star)
                vals, vecs = decompose(Q)

                pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
                v = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
                img = model.decode(v.unsqueeze(0)).squeeze(0)
                decoded_grid[ri, ci] = img.view(28, 28).numpy()
                rank_map[ri, ci]     = _eff_rank(vals)

    # Panel (a): decoded images
    fig1, axes1 = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 1.8, GRID_ROWS * 1.8),
                                gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    for ri in range(GRID_ROWS):
        for ci in range(GRID_COLS):
            axes1[ri, ci].imshow(decoded_grid[ri, ci], cmap="gray_r", vmin=0, vmax=1)
            axes1[ri, ci].axis("off")
    fig1.suptitle("Exp 02a — Pixel generative fields  (decode top +eigvec for each pixel direction)",
                  fontsize=11, y=1.01)
    save_fig(fig1, "figures/mnist/exp02_pixel_fields_grid.png")

    # Panel (b): effective rank heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(rank_map, cmap="viridis", vmin=1, vmax=model.d_latent, aspect="equal")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("Exp 02b — Pixel generative fields: effective rank of Q_dec", fontsize=10)
    ax2.set_xlabel("Column (sample)", fontsize=9)
    ax2.set_ylabel("Row (sample)",    fontsize=9)
    save_fig(fig2, "figures/mnist/exp02_pixel_fields_rank.png")


if __name__ == "__main__":
    main()
