"""
analysis.py — Weight-based eigendecomposition for the bilinear VAE encoder.

The eigendecomposition method is from Pearce et al. (2025); its application
to VAE encoders is our extension (Kefallinou et al., 2025).
    Because the encoder is bilinear, its output along any direction μ* can be
    written as a quadratic form in the input:

        f_{μ*}(x) = x^T Q x

    where the interaction matrix Q is built entirely from the model weights —
    no forward pass needed.  Eigendecomposing Q reveals which input patterns
    (pixel combinations) activate that latent direction.

Algorithm (for a BilinearVAE with embed → bilinear → fc_mu):

    1. Map μ* to hidden space:   u = P_μ^T μ*          shape: (d_hidden,)
    2. Build interaction matrix: Q = Σ_a u_a W_a: V_a:^T  shape: (d_embed, d_embed)
    3. Symmetrise:               Q ← ½(Q + Q^T)
    4. Project to input space:   Q_in = E^T Q E          shape: (d_input, d_input)
    5. Eigendecompose:           Q_in = Σ_i λ_i v_i v_i^T
    6. Sort by |λ_i| descending

Functions
─────────
    get_interaction_matrix  Build Q in input space for a given latent direction.
    decompose               Eigendecompose a symmetric matrix, sort by |λ|.
    analyze_direction       Combined: build Q and decompose.
    compute_class_means     Average encoder output μ per digit class.
    effective_rank          Scalar low-rank summary of an eigenvalue spectrum.
    pve                     Proportion of variance explained by top-k eigenvalues.
    get_interaction_matrix_conv
                            Same as get_interaction_matrix but for ConvBilinearVAE.
                            Q lives in the CNN feature space (d_feat × d_feat) because
                            the conv layers are non-linear and cannot be absorbed into Q.
    analyze_direction_conv  Combined: build conv Q and decompose.
"""

import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core: interaction matrix
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_interaction_matrix(model, direction: torch.Tensor) -> torch.Tensor:
    """
    Build the interaction matrix Q_in for a given latent direction.

    Args:
        model     : BilinearVAE — must have attributes:
                        embed.weight   (d_embed, d_input)
                        bilinear.w_l   (d_hidden, d_embed)
                        bilinear.w_r   (d_hidden, d_embed)
                        fc_mu.weight   (d_latent, d_hidden)
        direction : target latent direction μ*, shape (d_latent,)

    Returns:
        Q_in : symmetric interaction matrix in input space, shape (d_input, d_input)
    """
    direction = direction.to(next(model.parameters()).device)

    # Extract encoder weights
    E    = model.embed.weight       # (d_embed,  d_input)
    W    = model.bilinear.w_l       # (d_hidden, d_embed)
    V    = model.bilinear.w_r       # (d_hidden, d_embed)
    P_mu = model.fc_mu.weight       # (d_latent, d_hidden)

    # Step 1 — project direction to hidden space
    u = P_mu.T @ direction          # (d_hidden,)

    # Step 2 — build interaction matrix in embedding space
    # Q_embed[i,j] = Σ_a u_a · W[a,i] · V[a,j]
    # Equivalent to: Q_embed = W^T diag(u) V
    Q_embed = (W * u[:, None]).T @ V    # (d_embed, d_embed)

    # Step 3 — symmetrise (anti-symmetric part vanishes under x^T A x)
    Q_embed = 0.5 * (Q_embed + Q_embed.T)

    # Step 4 — project to pixel input space
    Q_in = E.T @ Q_embed @ E           # (d_input, d_input)

    return Q_in


# ─────────────────────────────────────────────────────────────────────────────
# Eigendecomposition
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decompose(matrix: torch.Tensor):
    """
    Eigendecompose a symmetric matrix and sort by descending |eigenvalue|.

    Args:
        matrix : symmetric matrix, shape (n, n)

    Returns:
        eigenvalues  : shape (n,), sorted by |λ| descending
        eigenvectors : shape (n, n), each row is an eigenvector
    """
    # eigh assumes symmetry and is faster + more numerically stable than eig
    vals, vecs = torch.linalg.eigh(matrix)

    # Sort by magnitude (largest absolute value first)
    order = vals.abs().argsort(descending=True)
    vals  = vals[order]
    vecs  = vecs[:, order].T      # transpose so each row is one eigenvector

    return vals.cpu(), vecs.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Combined: build and decompose
# ─────────────────────────────────────────────────────────────────────────────

def analyze_direction(model, direction: torch.Tensor):
    """
    Full pipeline: build interaction matrix for a latent direction, then
    eigendecompose it.

    Args:
        model     : BilinearVAE
        direction : latent direction μ*, shape (d_latent,)

    Returns:
        eigenvalues  : shape (d_input,), sorted by |λ| descending
        eigenvectors : shape (d_input, d_input), each row is an eigenvector
                       — reshape row i to (28, 28) for visualisation
    """
    Q_in = get_interaction_matrix(model, direction)
    return decompose(Q_in)


# ─────────────────────────────────────────────────────────────────────────────
# Class-mean encodings
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_class_means(model, loader, device: str = "cpu") -> dict:
    """
    Compute the average posterior mean μ for each class label.

    Args:
        model  : VAE (any variant) — must implement encode()
        loader : DataLoader returning (x, y) pairs
        device : torch device

    Returns:
        class_means : dict {label (int) → mean μ vector (d_latent,)}
    """
    model.eval()
    buckets = {}

    for x, labels in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        mu = mu.cpu()

        for i, label in enumerate(labels.tolist()):
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(mu[i])

    return {label: torch.stack(vecs).mean(0)
            for label, vecs in sorted(buckets.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative metrics on eigenvalue spectra
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(eigenvalues: torch.Tensor) -> float:
    """
    Effective rank of an eigenvalue spectrum (Roy & Vetterli, 2007).

    Measures how spread the spectrum is.  A spectrum concentrated on a single
    eigenvalue has effective rank 1; a flat spectrum has effective rank = n.

        ρ = exp( −Σ_i p_i log p_i )   where   p_i = |λ_i| / Σ_j |λ_j|

    Lower effective rank ↔ more interpretable (computation is more low-rank).

    Args:
        eigenvalues : 1-D tensor of eigenvalues (can be any order)

    Returns:
        effective rank as a Python float
    """
    p = eigenvalues.abs()
    p = p / p.sum()                          # normalise to a probability distribution
    # Avoid log(0) by adding a tiny epsilon
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return float(torch.exp(entropy))


def pve(eigenvalues: torch.Tensor, k: int) -> float:
    """
    Proportion of Variance Explained by the top-k eigenvalues.

        PVE@k = Σ_{i=1}^{k} |λ_i| / Σ_i |λ_i|

    Assumes eigenvalues are already sorted by |λ| descending (as returned
    by decompose()).  Higher PVE@k means the interaction is more low-rank.

    Args:
        eigenvalues : 1-D tensor, sorted by |λ| descending
        k           : number of leading eigenvalues to include

    Returns:
        PVE@k as a Python float in [0, 1]
    """
    total  = eigenvalues.abs().sum()
    top_k  = eigenvalues[:k].abs().sum()
    return float(top_k / total)


def spectrum_stats(eigenvalues: torch.Tensor) -> dict:
    """
    Compute all spectral statistics at once.

    Returns a dict with:
        effective_rank  : scalar
        pve_1, pve_5, pve_10 : proportion of variance explained
    """
    return {
        "effective_rank": effective_rank(eigenvalues),
        "pve_1":          pve(eigenvalues, 1),
        "pve_5":          pve(eigenvalues, 5),
        "pve_10":         pve(eigenvalues, 10),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ConvBilinearVAE analysis  (feature-space eigendecomposition)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_interaction_matrix_conv(model, direction: torch.Tensor) -> torch.Tensor:
    """
    Build the interaction matrix Q_feat for a ConvBilinearVAE.

    Unlike the flat BilinearVAE, the convolutional encoder has non-linear
    ReLU activations that cannot be absorbed into a single linear matrix E.
    Therefore Q is computed in the CNN feature space (d_feat × d_feat = 2048×2048)
    rather than the pixel space.

    The interpretation is:
        f_{μ*}(h) = h^T Q_feat h
    where h = flatten(conv_encoder(x)) is the feature vector.
    Eigenvectors of Q_feat can be reshaped to (128, 4, 4) spatial feature maps.

    Args:
        model     : ConvBilinearVAE — must have:
                        bilinear.w_l   (d_hidden, d_feat)
                        bilinear.w_r   (d_hidden, d_feat)
                        fc_mu.weight   (d_latent, d_hidden)
        direction : target latent direction μ*, shape (d_latent,)

    Returns:
        Q_feat : symmetric interaction matrix in feature space,
                 shape (d_feat, d_feat) = (2048, 2048)
    """
    direction = direction.to(next(model.parameters()).device)

    W    = model.bilinear.w_l     # (d_hidden, d_feat)
    V    = model.bilinear.w_r     # (d_hidden, d_feat)
    P_mu = model.fc_mu.weight     # (d_latent, d_hidden)

    # Map direction to the bilinear hidden space
    u = P_mu.T @ direction        # (d_hidden,)

    # Build interaction matrix in feature space:
    #   Q_feat[i, j] = Σ_a u_a · W[a, i] · V[a, j]
    Q_feat = (W * u[:, None]).T @ V    # (d_feat, d_feat)

    # Symmetrise (anti-symmetric part vanishes under h^T A h)
    Q_feat = 0.5 * (Q_feat + Q_feat.T)

    return Q_feat


def analyze_direction_conv(model, direction: torch.Tensor):
    """
    Full pipeline for ConvBilinearVAE: build the feature-space interaction
    matrix and eigendecompose it.

    Args:
        model     : ConvBilinearVAE
        direction : latent direction μ*, shape (d_latent,)

    Returns:
        eigenvalues  : shape (d_feat,), sorted by |λ| descending
        eigenvectors : shape (d_feat, d_feat), each row is one eigenvector
                       reshape row i to model.feat_shape = (128, 4, 4)
                       to get a spatial feature map for visualisation
    """
    Q_feat = get_interaction_matrix_conv(model, direction)
    return decompose(Q_feat)
