"""
Exp 12 — Decoder Control: Is the Near-Universal Direction Architectural?

The central finding of the decoder analysis is that DecBilinearVAE's top
eigenvector of Q_dec is nearly identical across all digit classes (mean cosine
≈ 0.842).  This experiment asks: *why*?

Three-way comparison:
  (A) Trained DecBilinearVAE      — the finding (mean ≈ 0.842)
  (B) Random DecBilinearVAE       — untrained, random weights
      If (B) is also high: universality is structural (mathematical property
      of bilinear Q_dec).  If (B) is low: it is a learned property.
  (C) VanillaVAE                  — standard MLP decoder, no bilinear layer.
      We cannot compute Q_dec for a non-bilinear decoder, so we instead measure
      pixel-space cosine similarity between decoded class-mean latent codes.
      This tests whether a non-bilinear decoder also collapses to similar outputs.

Interpretation guide:
    (A) high, (B) high, (C) low  →  structural (bilinear architecture forces it)
    (A) high, (B) low,  (C) low  →  learned (bilinear decoder learns it during training)
    (A) high, (B) any,  (C) high →  general VAE property (not bilinear-specific)

Figure saved:
    figures/mnist/exp12_decoder_control.png
"""

import os
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, compute_class_means
from visualize import similarity_heatmap, save_fig

CKPT_TRAINED = "checkpoints/mnist/model.pt"
CKPT_VANILLA = os.path.join(os.path.dirname(__file__),
                             "../../../extension_full/checkpoints/mnist/vanilla_vae/model.pt")
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


# ── Minimal VanillaVAE matching the saved checkpoint keys ─────────────────
class _VanillaVAE(nn.Module):
    def __init__(self, d_input=784, d_enc1=256, d_enc2=512, d_latent=10, d_dec=256):
        super().__init__()
        self.enc_fc1   = nn.Linear(d_input, d_enc1)
        self.enc_fc2   = nn.Linear(d_enc1,  d_enc2)
        self.fc_mu     = nn.Linear(d_enc2,  d_latent)
        self.fc_logvar = nn.Linear(d_enc2,  d_latent)
        self.decoder   = nn.Sequential(
            nn.Linear(d_latent, d_dec), nn.ReLU(),
            nn.Linear(d_dec, d_input),  nn.Sigmoid(),
        )
        # rename to match checkpoint keys: decoder.fc1, decoder.fc2
        self.decoder[0] = nn.Linear(d_latent, d_dec)
        self.decoder[2] = nn.Linear(d_dec, d_input)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)


class VanillaVAE(nn.Module):
    """VanillaVAE matching the extension_full checkpoint key layout."""
    def __init__(self, d_input=784, d_enc1=256, d_enc2=512, d_latent=10, d_dec=256):
        super().__init__()
        self.enc_fc1   = nn.Linear(d_input, d_enc1)
        self.enc_fc2   = nn.Linear(d_enc1,  d_enc2)
        self.fc_mu     = nn.Linear(d_enc2,  d_latent)
        self.fc_logvar = nn.Linear(d_enc2,  d_latent)

        class _Dec(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(d_latent, d_dec)
                self.fc2 = nn.Linear(d_dec, d_input)
            def forward(self, z):
                return torch.sigmoid(self.fc2(F.relu(self.fc1(z))))

        self.decoder = _Dec()

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)


def _dec_crossclass_similarity(model, mean_imgs):
    """Latent-space eigvec cross-class similarity for a DecBilinearVAE."""
    classes  = sorted(mean_imgs.keys())
    top_vecs = {}
    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            top_vecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_latent)
    n   = len(classes)
    mat = np.zeros((n, n))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            va, vb = top_vecs[a], top_vecs[b]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))
    return mat


def _vanilla_decoded_similarity(vanilla_model, lat_means):
    """
    Pixel-space cosine similarity between decoded class-mean latent codes.
    This is the best proxy for cross-class generative similarity in a
    non-bilinear decoder (no analytical eigenvectors available).
    """
    classes    = sorted(lat_means.keys())
    dec_images = {}
    with torch.no_grad():
        for c in classes:
            dec_images[c] = vanilla_model.decode(lat_means[c].unsqueeze(0)).squeeze(0)
    n   = len(classes)
    mat = np.zeros((n, n))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            va, vb = dec_images[a], dec_images[b]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))
    return mat


def main():
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}

    # (A) Trained DecBilinearVAE
    trained = DecBilinearVAE(); load_checkpoint(trained, CKPT_TRAINED); trained.eval()
    mat_trained = _dec_crossclass_similarity(trained, mean_imgs)

    # (B) Random (untrained) DecBilinearVAE
    random_model = DecBilinearVAE(); random_model.eval()
    mat_random   = _dec_crossclass_similarity(random_model, mean_imgs)

    # (C) Trained VanillaVAE — decoded class-mean latent similarity
    vanilla = VanillaVAE()
    ckpt_v  = torch.load(CKPT_VANILLA, map_location="cpu", weights_only=True)
    vanilla.load_state_dict(ckpt_v["model_state"]); vanilla.eval()
    lat_means_trained = compute_class_means(trained, loader)
    lat_means_vanilla = compute_class_means(vanilla, loader)
    mat_vanilla_dec   = _vanilla_decoded_similarity(vanilla, lat_means_vanilla)
    # Also: decode DecBilinearVAE class-mean latents (fair pixel-space baseline)
    mat_dec_means     = _vanilla_decoded_similarity(trained, lat_means_trained)

    def _off_diag_mean(m):
        return m[m < 1.0].mean()

    print(f"\n(A) Trained DecBilinearVAE  (eigvec cosine):          mean={_off_diag_mean(mat_trained):.3f}")
    print(f"(B) Random  DecBilinearVAE  (eigvec cosine):          mean={_off_diag_mean(mat_random):.3f}")
    print(f"(C) VanillaVAE             (decoded mean-lat cosine): mean={_off_diag_mean(mat_vanilla_dec):.3f}")
    print(f"    DecBilinearVAE         (decoded mean-lat cosine): mean={_off_diag_mean(mat_dec_means):.3f}")

    # Interpretation
    trained_mean = _off_diag_mean(mat_trained)
    random_mean  = _off_diag_mean(mat_random)
    vanilla_mean = _off_diag_mean(mat_vanilla_dec)
    dec_mean_lat = _off_diag_mean(mat_dec_means)

    print("\nInterpretation:")
    if random_mean > 0.7:
        print("  → High similarity in random model: universality is STRUCTURAL (bilinear Q_dec geometry)")
    else:
        print("  → Low similarity in random model: universality is LEARNED during training")
    if vanilla_mean > 0.7:
        print("  → High similarity in VanillaVAE: this is a general VAE decoder property")
    else:
        print("  → Low similarity in VanillaVAE: specific to bilinear decoder")

    # ── Figure ────────────────────────────────────────────────────────────
    lbls = [f"d{c}" for c in range(10)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    panels = [
        (mat_trained,  axes[0, 0],
         f"(A) Trained DecBilinearVAE\neigvec cosine  mean={trained_mean:.3f}"),
        (mat_random,   axes[0, 1],
         f"(B) Random DecBilinearVAE\neigvec cosine  mean={random_mean:.3f}"),
        (mat_dec_means, axes[1, 0],
         f"DecBilinearVAE decoded mean-latents\npixel cosine  mean={dec_mean_lat:.3f}"),
        (mat_vanilla_dec, axes[1, 1],
         f"(C) VanillaVAE decoded mean-latents\npixel cosine  mean={vanilla_mean:.3f}"),
    ]

    for mat, ax, title in panels:
        im = similarity_heatmap(ax, mat, lbls, title=title, vmin=0, vmax=1,
                                annotate_fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Exp 12 — Is the near-universal decoder direction architectural or learned?\n"
                 "(A) trained vs (B) random DecBilinearVAE vs (C) VanillaVAE",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp12_decoder_control.png")


if __name__ == "__main__":
    main()
