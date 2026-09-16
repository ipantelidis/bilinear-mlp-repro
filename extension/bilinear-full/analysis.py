"""
analysis.py — Weight-based eigendecomposition for the full bilinear VAE.

Both the encoder and decoder are bilinear, so both quadratic forms are available:

  Encoder Q_in  (d_input × d_input = 784×784):
      For any latent direction μ*, which pixel patterns activate it?
      Q_in = E^T Q_embed E   where Q_embed = ½(W^T diag(u) V + V^T diag(u) W)

  Decoder Q_dec (d_latent × d_latent = 10×10):
      For any output direction p*, which latent directions generate it?
      Q_dec = E_dec^T Q_embed_dec E_dec

Note: the enc_skip and decoder.skip linear components are NOT captured by
these quadratic forms.  Q_in/Q_dec describe the purely bilinear sensitivity.
"""

import torch


@torch.no_grad()
def get_encoder_interaction_matrix(model, direction: torch.Tensor) -> torch.Tensor:
    """
    Build encoder Q_in for latent direction μ*, shape (d_input, d_input).

    Args:
        model     : FullBilinearVAE — needs embed, bilinear, fc_mu
        direction : latent direction μ*, shape (d_latent,)
    """
    direction = direction.to(next(model.parameters()).device)
    E    = model.embed.weight       # (d_embed,  d_input)
    W    = model.bilinear.w_l       # (d_hidden, d_embed)
    V    = model.bilinear.w_r       # (d_hidden, d_embed)
    P_mu = model.fc_mu.weight       # (d_latent, d_hidden)

    u       = P_mu.T @ direction            # (d_hidden,)
    Q_embed = (W * u[:, None]).T @ V        # (d_embed, d_embed)
    Q_embed = 0.5 * (Q_embed + Q_embed.T)
    return E.T @ Q_embed @ E               # (d_input, d_input)


@torch.no_grad()
def get_decoder_interaction_matrix(model, output_direction: torch.Tensor) -> torch.Tensor:
    """
    Build decoder Q_dec for output direction p*, shape (d_latent, d_latent).

    Args:
        model            : FullBilinearVAE — needs decoder.embed_dec, bilinear_dec, fc_out
        output_direction : target output p*, shape (d_input,)
    """
    output_direction = output_direction.to(next(model.parameters()).device)
    E_dec = model.decoder.embed_dec.weight      # (d_embed, d_latent)
    W_dec = model.decoder.bilinear_dec.w_l      # (d_hidden, d_embed)
    V_dec = model.decoder.bilinear_dec.w_r      # (d_hidden, d_embed)
    P_out = model.decoder.fc_out.weight         # (d_input,  d_hidden)

    u       = P_out.T @ output_direction        # (d_hidden,)
    Q_embed = (W_dec * u[:, None]).T @ V_dec    # (d_embed, d_embed)
    Q_embed = 0.5 * (Q_embed + Q_embed.T)
    return E_dec.T @ Q_embed @ E_dec            # (d_latent, d_latent)


@torch.no_grad()
def decompose(matrix: torch.Tensor):
    """Eigendecompose symmetric matrix, sort by |λ| descending."""
    vals, vecs = torch.linalg.eigh(matrix)
    order = vals.abs().argsort(descending=True)
    return vals[order].cpu(), vecs[:, order].T.cpu()


@torch.no_grad()
def compute_class_means(model, loader, device="cpu") -> dict:
    model.eval()
    buckets = {}
    for x, labels in loader:
        mu, _ = model.encode(x.to(device))
        for i, lbl in enumerate(labels.tolist()):
            buckets.setdefault(lbl, []).append(mu[i].cpu())
    return {lbl: torch.stack(v).mean(0) for lbl, v in sorted(buckets.items())}


@torch.no_grad()
def mean_lat_norm(model, loader, device="cpu") -> float:
    norms = []
    for x, _ in loader:
        mu, _ = model.encode(x.to(device))
        norms.append(mu.norm(dim=1).cpu())
    return torch.cat(norms).mean().item()
