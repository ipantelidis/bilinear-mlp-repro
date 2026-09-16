"""visualize.py — Shared plotting helpers for bilinear-full experiments."""
import os
import numpy as np
import matplotlib.pyplot as plt


def similarity_heatmap(ax, mat, labels, title="", vmin=0, vmax=1,
                       cmap="YlOrRd", annotate=True, fontsize=7):
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
                        fontsize=fontsize,
                        color="white" if mat[i,j] > 0.6 else "black")
    return im


def save_fig(fig, path, dpi=130):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")
