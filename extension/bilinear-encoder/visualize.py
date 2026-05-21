"""
visualize.py — Shared plotting helpers.

All functions save a figure to disk and return nothing.
The `show` flag is False by default (suitable for headless servers).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Path, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvector grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvector_grid(
    images:     list,          # list of (28,28) numpy arrays
    titles:     list[str],
    row_labels: list[str],
    suptitle:   str,
    out_path:   Path,
    cmap:       str = "RdBu_r",
    show:       bool = False,
) -> None:
    """
    Generic grid where each column is one item and rows are variants.

    images: list of lists — images[row][col] is a (28,28) array.
    """
    n_rows = len(images)
    n_cols = len(images[0])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.6 * n_cols, 1.6 * n_rows),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )
    if n_rows == 1:
        axes = [axes]

    for row, row_imgs in enumerate(images):
        for col, img in enumerate(row_imgs):
            ax = axes[row][col]
            if img is not None:
                vmax = max(abs(img).max(), 1e-8)
                ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.axis("off")
        if row_labels:
            axes[row][0].set_ylabel(row_labels[row], fontsize=8,
                                    rotation=90, labelpad=4, va="center")

    for col, title in enumerate(titles):
        axes[0][col].set_title(title, fontsize=8)

    fig.suptitle(suptitle, fontsize=11, y=1.01)
    _save(fig, out_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    matrix:     np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title:      str,
    out_path:   Path,
    vmin:       float = 0.0,
    vmax:       float = 1.0,
    cmap:       str   = "YlOrRd",
    show:       bool  = False,
) -> None:
    """Square annotated heatmap."""
    n = len(row_labels)
    fig, ax = plt.subplots(figsize=(0.65 * n + 1.5, 0.65 * n + 1.2))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    threshold = vmin + 0.6 * (vmax - vmin)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    _save(fig, out_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Line plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_lines(
    curves:     dict,          # {label: array-like of y values}
    x_values:   list,
    xlabel:     str,
    ylabel:     str,
    title:      str,
    out_path:   Path,
    hlines:     list[tuple] = None,   # [(y, color, linestyle, label), ...]
    show:       bool = False,
) -> None:
    """Generic multi-line plot with optional horizontal reference lines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, (label, y) in enumerate(curves.items()):
        ax.plot(x_values, y, "o-", markersize=4, linewidth=1.5,
                color=cmap(i % 10), label=str(label))

    if hlines:
        for (y, color, ls, lbl) in hlines:
            ax.axhline(y, color=color, linestyle=ls, linewidth=0.9, label=lbl)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path, show)
