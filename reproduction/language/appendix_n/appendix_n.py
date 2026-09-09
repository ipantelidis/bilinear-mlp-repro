# ============================================================
# Appendix N — Language: Further Details for Feature Circuits
# Reproduces Figure 27 from Pearce et al. (2025)
#
# Panel A: Histogram of self-interactions vs cross-interactions
#          for the not-good output SAE feature (1882).
# Panel B: Eigenvalue spectrum of Q[1882] vs a random symmetric
#          matrix with the same standard deviation (dashed).
#
# Reuses the Figure 8 pipeline (same model / SAE / Q computation).
# ============================================================

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

from language.transformer import Transformer, Attention, Rotary
from sae.tracer import Tracer

if not hasattr(Transformer, "all_tied_weights_keys"):
    Transformer.all_tied_weights_keys = property(lambda self: {})


def fix_buffers(model):
    for m in model.modules():
        if isinstance(m, Attention):
            dev = next(m.parameters()).device
            m.mask = torch.tril(
                torch.ones(m.config.n_ctx, m.config.n_ctx)
            )[None, None].to(dev)
        if isinstance(m, Rotary):
            m.seq_len_cached = None


torch.set_grad_enabled(False)

LAYER    = 4
NOT_GOOD = 1882   # "not-good" output SAE feature

print("Loading model and SAEs …")
model = Transformer.from_pretrained("tdooms/ts-medium", device="cuda")
fix_buffers(model)
tracer = Tracer(model, layer=LAYER, inp=dict(name="resid-mid"))

# ============================================================
# Panel A — Histogram of self-interactions vs cross-interactions
# ============================================================
print("Computing Panel A …")
# Q projected into SAE input latent space
q_sae = tracer.q(NOT_GOOD, project=True).cpu()  # (d_features_in, d_features_in)

# Cross-interactions: lower-triangle off-diagonal (× 2 for symmetry)
idxs  = torch.tril_indices(*q_sae.shape, offset=-1)
cross = (2 * q_sae[idxs[0], idxs[1]]).numpy()

# Self-interactions: diagonal
self_int = q_sae.diagonal().numpy()

fig_a, ax_a = plt.subplots(figsize=(5, 4), dpi=150)
bins = np.linspace(-0.35, 0.45, 120)
ax_a.hist(cross,    bins=bins, label="Cross-interactions", alpha=0.7,
          color="#1f77b4", density=False)
ax_a.hist(self_int, bins=bins, label="Self-interaction",   alpha=0.7,
          color="#ff7f0e", density=False)
ax_a.set_yscale("log")
ax_a.set_xlabel("Interaction", fontsize=11)
ax_a.set_ylabel("Count",       fontsize=11)
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3)
fig_a.tight_layout()
fig_a.savefig(HERE / "fig_27a.png", dpi=150, bbox_inches="tight")
plt.close(fig_a)
print("  → fig_27a.png")

# ============================================================
# Panel B — Eigenvalue spectrum vs random matrix
# ============================================================
print("Computing Panel B …")
# Q in model hidden space for the not-good feature
q_model = tracer.q(NOT_GOOD, project=False).cpu()
q_model = 0.5 * (q_model + q_model.T)

vals_actual = torch.linalg.eigvalsh(q_model).numpy()   # ascending

# Random symmetric matrix with Gaussian entries matching the bulk std of Q.
# Use the middle 90% of eigenvalues to estimate the bulk scale (excluding
# the two large outliers), matching the paper's comparison.
d = q_model.shape[0]
vals_sorted = np.sort(np.abs(vals_actual))
bulk_cutoff  = vals_sorted[int(0.90 * d)]
bulk_vals    = vals_actual[np.abs(vals_actual) < bulk_cutoff]
bulk_std     = float(bulk_vals.std()) if len(bulk_vals) > 10 else float(q_model.std())
# For a random symmetric matrix with entry std σ, the eigenvalue range is
# approximately [-2σ√d, 2σ√d] (Wigner semicircle). Set σ so the range
# matches the bulk of the interaction matrix.
sigma = bulk_std / (2 * d ** 0.5)
torch.manual_seed(0)
R    = torch.randn(d, d) * sigma
R    = 0.5 * (R + R.T)
vals_random = torch.linalg.eigvalsh(R).numpy()

x = np.arange(len(vals_actual))

fig_b, ax_b = plt.subplots(figsize=(5, 4), dpi=150)
ax_b.plot(x, vals_actual, color="#1f77b4", linewidth=1.5,
          label="Interaction Matrix")
ax_b.plot(x, vals_random,  color="red", linewidth=1.2,
          linestyle="--", label="Random Symmetric Matrix")
ax_b.set_xlabel("Eigenvector index", fontsize=11)
ax_b.set_ylabel("Eigenvalue",         fontsize=11)
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3)
fig_b.tight_layout()
fig_b.savefig(HERE / "fig_27b.png", dpi=150, bbox_inches="tight")
plt.close(fig_b)
print("  → fig_27b.png")
print("Done.")
