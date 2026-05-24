"""
visualize.py — Shared plotting helpers for the bilinear decoder experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


MNIST_CLASSES   = list(range(10))
FMNIST_NAMES    = {0:'T-shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat',
                   5:'Sandal',  6:'Shirt',   7:'Sneaker',  8:'Bag',   9:'Boot'}


def show_image(ax, img_tensor, title="", xlabel="", cmap="gray_r",
               vmin=0, vmax=1, title_fontsize=8, xlabel_fontsize=7):
    ax.imshow(img_tensor.view(28, 28).numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.axis("off")


def similarity_heatmap(ax, mat, labels, title="", vmin=0, vmax=1, cmap="YlOrRd",
                       annotate=True, annotate_fontsize=7):
    n = len(labels)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=annotate_fontsize,
                        color="white" if mat[i,j] > 0.6 else "black")
    return im


def save_fig(fig, path, dpi=130):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
