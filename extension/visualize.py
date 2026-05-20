"""
visualize.py — Plotting functions for the bilinear VAE extension.

All functions save figures to a given output directory and optionally
display them.  No side-effects on the model or data.

Functions
─────────
    plot_reconstructions        Original vs. reconstructed images.
    plot_eigenspectrum          Eigenvalue spectrum + top eigenvectors (flat models).
    plot_eigenspectrum_conv     Same but for ConvBilinearVAE — eigenvectors shown
                                as spatial feature maps (128×4×4, mean over channels).
    plot_spectra_grid           Eigenvalue spectra across all class-mean directions.
    plot_interpolation          Eigenvectors along a latent interpolation path.
    plot_latent_pca             2D PCA of latent μ vectors, coloured by class.
    plot_training_curves        Train/test loss curves from training history.
    plot_reconstructions_color  Reconstructions for RGB images (CIFAR-10).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from analysis import analyze_direction, compute_class_means


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstructions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstructions(model, loader, device: str = "cpu",
                         n: int = 10, out_path: str = "figures/reconstructions.png",
                         show: bool = False):
    """Show n original images (top row) and their reconstructions (bottom row)."""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    recon, _, _ = model(x)

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3))
    for i in range(n):
        axes[0, i].imshow(x[i].cpu().view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original",     fontsize=9, rotation=90, labelpad=4)
    axes[1, 0].set_ylabel("Reconstruct.", fontsize=9, rotation=90, labelpad=4)
    fig.suptitle("Reconstructions", fontsize=11)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue spectrum + eigenvectors for one latent direction
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenspectrum(eigenvalues: torch.Tensor, eigenvectors: torch.Tensor,
                       title: str = "", n_eig: int = 4, n_spectrum: int = 30,
                       out_path: str = "figures/eigenspectrum.png", show: bool = False):
    """
    Plot eigenvalue spectrum (bar chart) and top positive/negative eigenvectors.

    Args:
        eigenvalues  : 1-D tensor sorted by |λ| descending
        eigenvectors : 2-D tensor, each row is an eigenvector (d_input,)
        n_eig        : number of eigenvectors to visualise per row
        n_spectrum   : number of eigenvalues shown in the bar chart
    """
    pos_mask = eigenvalues > 0
    neg_mask = eigenvalues < 0
    pos_idx  = torch.where(pos_mask)[0]
    neg_idx  = torch.where(neg_mask)[0]

    fig, axes = plt.subplots(2, n_eig + 1, figsize=(3 * (n_eig + 1), 6))

    # Eigenvalue bar charts (left column)
    pos_vals = eigenvalues[pos_idx[:n_spectrum]].numpy()
    axes[0, 0].barh(range(len(pos_vals)), pos_vals, color="steelblue")
    axes[0, 0].set_title("+ eigenvalues", fontsize=9)
    axes[0, 0].invert_yaxis()

    neg_vals = eigenvalues[neg_idx[:n_spectrum]].numpy()
    axes[1, 0].barh(range(len(neg_vals)), neg_vals, color="indianred")
    axes[1, 0].set_title("− eigenvalues", fontsize=9)
    axes[1, 0].invert_yaxis()

    # Eigenvector images
    vmax = eigenvectors[:max(len(pos_idx), len(neg_idx))].abs().max().item() + 1e-8
    for i in range(n_eig):
        for row, idx_set in enumerate([pos_idx, neg_idx]):
            ax = axes[row, i + 1]
            if i < len(idx_set):
                img = eigenvectors[idx_set[i]].view(28, 28).numpy()
                ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue spectra across all class-mean directions
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectra_grid(model, loader, device: str = "cpu", n_show: int = 30,
                      out_path: str = "figures/spectra_grid.png", show: bool = False):
    """
    For each digit class, plot the eigenvalue spectrum of the class-mean direction.
    Useful for comparing low-rank structure across classes.
    """
    class_means = compute_class_means(model, loader, device)
    n_classes   = len(class_means)

    fig, axes = plt.subplots(2, n_classes // 2, figsize=(2.5 * (n_classes // 2), 5))
    axes = axes.flatten()

    for ax, (label, direction) in zip(axes, class_means.items()):
        vals, _ = analyze_direction(model, direction)
        ax.plot(vals[:n_show].numpy(), "o-", markersize=3, linewidth=1, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"digit {label}", fontsize=9)
        ax.set_xlabel("rank", fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle("Eigenvalue spectra — class mean encodings", fontsize=11)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# Latent-space interpolation
# ─────────────────────────────────────────────────────────────────────────────

def plot_interpolation(model, class_means: dict, digit_a: int, digit_b: int,
                       n_steps: int = 7, n_eig: int = 4,
                       out_path: str = "figures/interpolation.png", show: bool = False):
    """
    Interpolate between two class-mean directions and show top eigenvectors.

    α = 0 → digit_a,  α = 1 → digit_b.
    Each column is one interpolation step; each row is one eigenvector rank.
    """
    alphas = np.linspace(0, 1, n_steps)

    fig, axes = plt.subplots(n_eig, n_steps, figsize=(2.2 * n_steps, 2.2 * n_eig))

    for col, alpha in enumerate(alphas):
        # Linearly blend the two class-mean directions
        direction = (1 - alpha) * class_means[digit_a] + alpha * class_means[digit_b]
        vals, vecs = analyze_direction(model, direction)

        pos_idx = torch.where(vals > 0)[0]
        vmax    = vecs[:n_eig].abs().max().item() + 1e-8

        for row in range(n_eig):
            ax = axes[row, col]
            if row < len(pos_idx):
                img = vecs[pos_idx[row]].view(28, 28).numpy()
                ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"α={alpha:.2f}", fontsize=8)

    fig.suptitle(f"Interpolation: digit {digit_a} → digit {digit_b}  "
                 f"(top {n_eig} eigenvectors)", fontsize=11)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# Latent PCA
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_latent_pca(model, loader, device: str = "cpu",
                    out_path: str = "figures/latent_pca.png", show: bool = False):
    """
    Plot a 2D PCA of latent μ vectors coloured by class label.
    Useful for visualising class separation in latent space.
    """
    model.eval()
    zs, ys = [], []

    for x, y in loader:
        mu, _ = model.encode(x.to(device))
        zs.append(mu.cpu().numpy())
        ys.append(y.numpy())

    z_all = np.concatenate(zs)
    y_all = np.concatenate(ys)

    z_2d = PCA(n_components=2).fit_transform(z_all)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_all,
                         cmap="tab10", s=3, alpha=0.5)
    fig.colorbar(scatter, ax=ax, label="digit class")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("Latent space (μ, PCA)", fontsize=11)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: list, out_path: str = "figures/training_curves.png",
                         show: bool = False):
    """
    Plot train and test ELBO loss over epochs from the history returned by train().
    """
    epochs     = [h["epoch"]       for h in history]
    train_loss = [h["train_total"] for h in history]
    test_loss  = [h["test_total"]  for h in history]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, train_loss, label="train", linewidth=1.5)
    ax.plot(epochs, test_loss,  label="test",  linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("ELBO loss (per sample)")
    ax.set_title("Training curves"); ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# β sweep — effective rank and PVE vs β
# ─────────────────────────────────────────────────────────────────────────────

def plot_beta_sweep(sweep_results: dict,
                    out_path: str = "figures/beta_sweep.png",
                    show: bool = False):
    """
    Plot how spectral metrics change as β increases in the β-bilinear VAE.

    Args:
        sweep_results : dict {beta_value → metrics_dict}
                        metrics_dict must contain effective_rank, pve_1, pve_5, pve_10
    """
    betas  = sorted(sweep_results.keys())
    eff_r  = [sweep_results[b]["effective_rank"] for b in betas]
    pve1   = [sweep_results[b]["pve_1"]          for b in betas]
    pve5   = [sweep_results[b]["pve_5"]          for b in betas]
    pve10  = [sweep_results[b]["pve_10"]         for b in betas]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A — effective rank vs β
    axes[0].plot(betas, eff_r, "o-", color="#1f77b4", linewidth=2, markersize=7)
    axes[0].set_xlabel("β (KL weight)", fontsize=11)
    axes[0].set_ylabel("Effective rank", fontsize=11)
    axes[0].set_title("Effective rank vs β", fontsize=11)
    axes[0].set_xticks(betas)
    axes[0].grid(True, alpha=0.3)

    # Panel B — PVE@k vs β
    axes[1].plot(betas, pve1,  "o-", label="PVE@1",  linewidth=2, markersize=6)
    axes[1].plot(betas, pve5,  "s-", label="PVE@5",  linewidth=2, markersize=6)
    axes[1].plot(betas, pve10, "^-", label="PVE@10", linewidth=2, markersize=6)
    axes[1].set_xlabel("β (KL weight)", fontsize=11)
    axes[1].set_ylabel("Proportion of variance explained", fontsize=11)
    axes[1].set_title("PVE@k vs β", fontsize=11)
    axes[1].set_xticks(betas)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("β-Bilinear VAE: effect of KL weight on interaction structure",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# Eigenfeature consistency across seeds
# ─────────────────────────────────────────────────────────────────────────────

def plot_consistency(consistency_result: dict,
                     out_path: str = "figures/consistency.png",
                     show: bool = False):
    """
    Bar chart of per-class eigenfeature cosine similarity across seeds.

    Args:
        consistency_result : dict returned by metrics.eigenfeature_consistency()
                             must have keys "mean" and "per_class"
    """
    per_class = consistency_result["per_class"]
    mean_sim  = consistency_result["mean"]

    labels = sorted(per_class.keys())
    sims   = [per_class[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(l) for l in labels], sims, color="steelblue", alpha=0.8)
    ax.axhline(mean_sim, color="indianred", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_sim:.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit class", fontsize=11)
    ax.set_ylabel("Mean pairwise cosine similarity", fontsize=11)
    ax.set_title("Eigenfeature consistency across training seeds\n"
                 "(BilinearVAE encoder, top positive eigenvector per class)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# ConvBilinearVAE — feature-space eigenvector visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenspectrum_conv(eigenvalues: torch.Tensor, eigenvectors: torch.Tensor,
                            feat_shape: tuple, title: str = "",
                            n_eig: int = 4, n_spectrum: int = 30,
                            out_path: str = "figures/eigenspectrum_conv.png",
                            show: bool = False):
    """
    Plot eigenvalue spectrum and top eigenvectors for a ConvBilinearVAE.

    Because the conv encoder is non-linear, eigenvectors live in the CNN feature
    space rather than pixel space. We visualise them by:
      1. Reshaping each eigenvector to feat_shape (e.g. 128×4×4)
      2. Taking the L2 norm across channels to get a (4×4) spatial activation map
      3. Upsampling the map to a larger size for visibility

    This shows which spatial regions of the feature map each eigenvector activates,
    analogous to how pixel-space eigenvectors show which image regions matter.

    Args:
        eigenvalues  : 1-D tensor, sorted by |λ| descending
        eigenvectors : 2-D tensor, each row is a d_feat-dimensional eigenvector
        feat_shape   : tuple (C, H, W) to reshape eigenvectors into, e.g. (128, 4, 4)
        n_eig        : number of eigenvectors to show per row
        n_spectrum   : number of eigenvalues shown in the bar chart
    """
    import torch.nn.functional as F

    pos_mask = eigenvalues > 0
    neg_mask = eigenvalues < 0
    pos_idx  = torch.where(pos_mask)[0]
    neg_idx  = torch.where(neg_mask)[0]

    fig, axes = plt.subplots(2, n_eig + 1, figsize=(3 * (n_eig + 1), 6))

    # Eigenvalue bar charts (left column)
    pos_vals = eigenvalues[pos_idx[:n_spectrum]].numpy()
    axes[0, 0].barh(range(len(pos_vals)), pos_vals, color="steelblue")
    axes[0, 0].set_title("+ eigenvalues", fontsize=9)
    axes[0, 0].invert_yaxis()

    neg_vals = eigenvalues[neg_idx[:n_spectrum]].numpy()
    axes[1, 0].barh(range(len(neg_vals)), neg_vals, color="indianred")
    axes[1, 0].set_title("- eigenvalues", fontsize=9)
    axes[1, 0].invert_yaxis()

    def vec_to_spatial_map(vec):
        """
        Convert a feature-space eigenvector to a 2D spatial activation map.
        Steps: reshape to (C, H, W) → L2 norm over channels → (H, W)
        The resulting map shows which spatial positions are most activated.
        """
        feat = vec.reshape(feat_shape)                        # (C, H, W)
        spatial = feat.norm(dim=0)                            # (H, W)
        # Upsample for better visibility
        spatial = F.interpolate(
            spatial.unsqueeze(0).unsqueeze(0),
            size=(64, 64), mode="bilinear", align_corners=False
        ).squeeze()
        return spatial.numpy()

    # Eigenvector spatial maps
    for i in range(n_eig):
        for row, idx_set in enumerate([pos_idx, neg_idx]):
            ax = axes[row, i + 1]
            if i < len(idx_set):
                smap = vec_to_spatial_map(eigenvectors[idx_set[i]])
                ax.imshow(smap, cmap="hot")
            ax.axis("off")

    # Add a note explaining what's shown
    fig.text(0.5, 0.01,
             "Spatial activation maps: L2 norm over channels of each feature-space eigenvector",
             ha="center", fontsize=8, color="gray")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, Path(out_path), show)


# ─────────────────────────────────────────────────────────────────────────────
# RGB reconstructions  (for CIFAR-10)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstructions_color(model, loader, device: str = "cpu",
                               n: int = 8,
                               out_path: str = "figures/reconstructions_color.png",
                               show: bool = False):
    """
    Show n RGB originals (top row) and their reconstructions (bottom row).
    Used for CIFAR-10 where images are (3, 32, 32) tensors.
    """
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    recon, _, _ = model(x)

    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 3.8))
    for i in range(n):
        # permute (C, H, W) → (H, W, C) for imshow
        axes[0, i].imshow(x[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original",     fontsize=9, rotation=90, labelpad=4)
    axes[1, 0].set_ylabel("Reconstruct.", fontsize=9, rotation=90, labelpad=4)
    fig.suptitle("CIFAR-10 Reconstructions", fontsize=11)
    fig.tight_layout()
    _save(fig, Path(out_path), show)
