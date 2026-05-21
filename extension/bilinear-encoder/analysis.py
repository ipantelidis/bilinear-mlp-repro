"""
analysis.py — Weight-based eigendecomposition for the BilinearVAE encoder.

Core idea (Pearce et al., ICLR 2025):
    Because the encoder is bilinear, the activation along any latent direction
    μ* is a quadratic form in the input:

        f_{μ*}(x) = x^T Q x

    where Q is built entirely from the encoder weight matrices.
    Eigendecomposing Q reveals which pixel patterns activate or suppress μ*.

Algorithm — given target direction μ* ∈ R^{d_latent}:
    1. u = P_μ^T μ*                     project to hidden space
    2. Q_embed = ½(W_L^T diag(u) W_R    build interaction matrix
                 + W_R^T diag(u) W_L)
    3. Q = E^T Q_embed E                 project to pixel space
    4. Q = Σ_k λ_k v_k v_k^T           eigendecompose

    Positive eigenvectors  (λ > 0): patterns that activate μ*
    Negative eigenvectors  (λ < 0): patterns that suppress μ*

Public API:
    interaction_matrix     Build Q for a given latent direction
    decompose              Eigendecompose a symmetric matrix, sort by |λ|
    class_means            Average encoder output μ per class label
    spectrum_stats         Effective rank, PVE@k
"""

import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Interaction matrix
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def interaction_matrix(model, direction: torch.Tensor) -> torch.Tensor:
    """
    Build Q ∈ R^{d_input × d_input} for a given latent direction.

    Args:
        model:     BilinearVAE
        direction: μ*, shape (d_latent,)

    Returns:
        Q: symmetric interaction matrix in pixel space, shape (784, 784)
    """
    direction = direction.to(next(model.parameters()).device)

    E    = model.embed.weight       # (d_embed,  d_input)
    W_L  = model.bilinear.w_l       # (d_hidden, d_embed)
    W_R  = model.bilinear.w_r       # (d_hidden, d_embed)
    P_mu = model.fc_mu.weight       # (d_latent, d_hidden)

    # Step 1 — project direction to hidden space
    u = P_mu.T @ direction           # (d_hidden,)

    # Step 2 — interaction matrix in embedding space
    # Q_embed[i,j] = Σ_a u_a · W_L[a,i] · W_R[a,j]
    Q_embed = (W_L * u[:, None]).T @ W_R   # (d_embed, d_embed)
    Q_embed = 0.5 * (Q_embed + Q_embed.T)  # symmetrise

    # Step 3 — project to pixel space
    Q = E.T @ Q_embed @ E            # (d_input, d_input)

    return Q


# ─────────────────────────────────────────────────────────────────────────────
# Eigendecomposition
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decompose(matrix: torch.Tensor):
    """
    Eigendecompose a symmetric matrix, sorted by descending |λ|.

    Args:
        matrix: symmetric (n, n) tensor

    Returns:
        eigenvalues:  (n,)    sorted by |λ| descending
        eigenvectors: (n, n)  each row is one eigenvector
    """
    vals, vecs = torch.linalg.eigh(matrix)
    order = vals.abs().argsort(descending=True)
    vals  = vals[order]
    vecs  = vecs[:, order].T       # rows → eigenvectors
    return vals.cpu(), vecs.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Class means
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def class_means(model, loader, device: str = "cpu") -> dict:
    """
    Compute the average posterior mean μ per class label.

    Args:
        model:  BilinearVAE
        loader: DataLoader yielding (x, label) pairs; x can be (N,1,28,28)
                or (N,784) — both are handled.
        device: torch device string

    Returns:
        dict {label (int) → mean μ vector (d_latent,)}
    """
    model.eval()
    buckets: dict[int, list] = {}

    for x, labels in loader:
        x = x.view(x.size(0), -1).to(device)
        mu, _ = model.encode(x)
        mu = mu.cpu()
        for i, label in enumerate(labels.tolist()):
            buckets.setdefault(label, []).append(mu[i])

    return {label: torch.stack(vecs).mean(0)
            for label, vecs in sorted(buckets.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Spectral statistics
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(eigenvalues: torch.Tensor) -> float:
    """
    Effective rank (Roy & Vetterli, 2007): exp(entropy of normalised |λ|).
    Lower → more concentrated / more interpretable.
    """
    p = eigenvalues.abs()
    p = p / p.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return float(torch.exp(entropy))


def pve(eigenvalues: torch.Tensor, k: int) -> float:
    """Proportion of variance explained by the top-k eigenvalues."""
    total = eigenvalues.abs().sum()
    return float(eigenvalues[:k].abs().sum() / total)


def spectrum_stats(eigenvalues: torch.Tensor) -> dict:
    """Return effective rank and PVE@1/5/10 in one call."""
    return {
        "effective_rank": effective_rank(eigenvalues),
        "pve_1":          pve(eigenvalues, 1),
        "pve_5":          pve(eigenvalues, 5),
        "pve_10":         pve(eigenvalues, 10),
    }
