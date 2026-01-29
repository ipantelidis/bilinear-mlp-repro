import torch
from encoder import VAE 

@torch.no_grad()
def bilinear_latent_decomp(encoder, head="mu"):
    """
    Paper-style decomposition for your encoder:
    For each latent dim l, build interaction matrix Q_l (hidden x hidden),
    eigendecompose, and optionally map eigenvectors to pixel space via embed.
    """
    assert head in ["mu", "logvar"]
    block = encoder.block1
    L = block.w_l          # (d_hidden, d_hidden) or (d_hidden, d_in) depending on your Bilinear impl
    R = block.w_r
    P = getattr(encoder, head).weight  # (d_latent, d_hidden)

    # Q_l in hidden space: (d_latent, d_hidden, d_hidden)
    Q = torch.einsum("lo,oi,oj->lij", P, L, R)
    Q = 0.5 * (Q + Q.transpose(-1, -2))  # symmetrize

    vals, vecs = torch.linalg.eigh(Q)    # vals, vecs: (d_latent, d_hidden), (d_latent, d_hidden, d_hidden)

    # Map eigenvectors to pixel space for visualization if you want:
    # v_hidden is length d_hidden, pixel_dir = embed.weight.T @ v_hidden
    # embed.weight is (d_hidden, d_input) in your code
    E = encoder.embed.weight  # (d_hidden, d_input)
    vecs_pix = torch.einsum("lhi,hd->lid", vecs, E)  # (d_latent, d_hidden, d_input)

    return Q, vals, vecs, vecs_pix

@torch.no_grad()
def latent_rank_metrics(vals, eps=1e-8, topk=5):
    """
    vals: (d_latent, d_hidden) eigenvalues for each latent interaction matrix.
    Returns:
      eff_rank: effective rank per latent (exp entropy of normalized |lambda|)
      energy_topk: fraction of sum(|lambda|) captured by top-k
      max_abs: max |lambda|
      l1: sum |lambda|
    """
    lam = vals.abs()
    l1 = lam.sum(dim=-1) + eps
    p = lam / l1.unsqueeze(-1)
    entropy = -(p * (p + eps).log()).sum(dim=-1)
    eff_rank = entropy.exp()

    topk_vals = torch.topk(lam, k=min(topk, lam.size(-1)), dim=-1).values
    energy_topk = topk_vals.sum(dim=-1) / l1

    max_abs = lam.max(dim=-1).values
    return {
        "eff_rank": eff_rank,
        "energy_topk": energy_topk,
        "max_abs": max_abs,
        "l1": l1,
    }

@torch.no_grad()
def encoder_weight_only_report(encoder, topk=5):
    # Decompose both heads
    Q_mu, vals_mu, vecs_mu, vecs_mu_pix = bilinear_latent_decomp(encoder, head="mu")
    Q_lv, vals_lv, vecs_lv, vecs_lv_pix = bilinear_latent_decomp(encoder, head="logvar")

    m_mu = latent_rank_metrics(vals_mu, topk=topk)
    m_lv = latent_rank_metrics(vals_lv, topk=topk)

    # Row norms of the heads are another simple "is this latent used" indicator
    mu_row_norm = encoder.mu.weight.norm(dim=1)
    lv_row_norm = encoder.logvar.weight.norm(dim=1)

    # A simple collapse score: small interaction strength and small head norm
    # (tune thresholds for your model scale)
    collapse_score = (m_mu["l1"] * mu_row_norm)

    return {
        "vals_mu": vals_mu, "vals_logvar": vals_lv,
        "metrics_mu": m_mu, "metrics_logvar": m_lv,
        "mu_row_norm": mu_row_norm, "logvar_row_norm": lv_row_norm,
        "collapse_score": collapse_score,
        "vecs_mu_pix": vecs_mu_pix, "vecs_logvar_pix": vecs_lv_pix,
    }


device = torch.device("cpu")

model = VAE().to(device)
model.load_state_dict(torch.load("encoder_vae.pt", map_location=device))
model.eval()


report = encoder_weight_only_report(model.encoder, topk=5)
print(report["metrics_mu"]["eff_rank"])
print(report["metrics_mu"]["energy_topk"])
print(report["collapse_score"])
