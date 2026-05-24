"""
analysis.py — Weight-based eigendecomposition for the bilinear decoder VAE.

Because the decoder is bilinear, its output along any direction p* can be
written as a quadratic form in the latent code:

    p* · ŷ(z) = z^T Q_dec z

where Q_dec is built analytically from the decoder weights — no forward pass
or data needed.  Eigendecomposing Q_dec reveals which latent directions generate
vs. suppress each output pattern.

Algorithm (for DecBilinearVAE with embed_dec → bilinear_dec → fc_out):

    1. Map p* to hidden space:   u = P_out^T p*        shape: (d_hidden,)
    2. Build embed-space matrix: Q_embed = ½(W^T diag(u) V + V^T diag(u) W)
    3. Project to latent space:  Q_dec = E^T Q_embed E  shape: (d_latent, d_latent)
    4. Eigendecompose:           Q_dec = Σ_i λ_i v_i v_i^T
    5. Sort by |λ_i| descending

The resulting eigenvectors are latent directions.
Pass them through model.decoder to visualise what they generate.
"""

import torch


@torch.no_grad()
def get_decoder_interaction_matrix(model, output_direction: torch.Tensor) -> torch.Tensor:
    """
    Build Q_dec for a target output direction p*, shape (d_input,).

    Returns Q_dec: symmetric matrix, shape (d_latent, d_latent).
    """
    output_direction = output_direction.to(next(model.parameters()).device)

    E_dec = model.decoder.embed_dec.weight      # (d_embed, d_latent)
    W_dec = model.decoder.bilinear_dec.w_l      # (d_hidden, d_embed)
    V_dec = model.decoder.bilinear_dec.w_r      # (d_hidden, d_embed)
    P_out = model.decoder.fc_out.weight         # (d_input,  d_hidden)

    u       = P_out.T @ output_direction        # (d_hidden,)
    Q_embed = (W_dec * u[:, None]).T @ V_dec    # (d_embed, d_embed)
    Q_embed = 0.5 * (Q_embed + Q_embed.T)
    Q_dec   = E_dec.T @ Q_embed @ E_dec         # (d_latent, d_latent)

    return Q_dec


@torch.no_grad()
def decompose(matrix: torch.Tensor):
    """
    Eigendecompose a symmetric matrix and sort by descending |eigenvalue|.

    Returns:
        eigenvalues  : shape (n,), sorted by |λ| descending
        eigenvectors : shape (n, n), each row is one eigenvector
    """
    vals, vecs = torch.linalg.eigh(matrix)
    order = vals.abs().argsort(descending=True)
    vals  = vals[order]
    vecs  = vecs[:, order].T
    return vals.cpu(), vecs.cpu()


@torch.no_grad()
def compute_class_means(model, loader, device="cpu") -> dict:
    """
    Compute average posterior mean μ per class.

    Returns dict {label (int) → mean μ vector (d_latent,)}.
    """
    model.eval()
    buckets = {}
    for x, labels in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        mu = mu.cpu()
        for i, label in enumerate(labels.tolist()):
            buckets.setdefault(label, []).append(mu[i])
    return {label: torch.stack(vecs).mean(0) for label, vecs in sorted(buckets.items())}


def mean_lat_norm(model, loader, device="cpu") -> float:
    """Return mean L2 norm of encoded latent codes — used to scale eigenvectors."""
    norms = []
    with torch.no_grad():
        for x, _ in loader:
            mu, _ = model.encode(x.to(device))
            norms.append(mu.norm(dim=1).cpu())
    return torch.cat(norms).mean().item()
