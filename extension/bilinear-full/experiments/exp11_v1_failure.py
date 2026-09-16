"""
Exp 11 — V1 Failure: Even-Symmetry Collapse

The pure bilinear VAE (v1, no skip connections) suffers from a structural flaw:
    decode(z) ≡ decode(-z)   ∀ z

Proof: bilinear(embed_dec(-z)) = (W_dec @ embed_dec(-z)) ⊙ (V_dec @ embed_dec(-z))
                                = (-W_dec @ embed_dec(z)) ⊙ (-V_dec @ embed_dec(z))
                                = (W_dec @ embed_dec(z)) ⊙ (V_dec @ embed_dec(z))
                                = bilinear(embed_dec(z))

Consequence: the decoder can't distinguish z from -z. During training, the encoder
has no incentive to use the sign of z, collapsing the latent space.

This experiment:
  1. Verifies the f(z)=f(-z) identity numerically
  2. Compares test ELBO: V1 (~287) vs V2/current (~129)
  3. Shows reconstruction quality difference
  4. Shows the degenerate latent structure of V1

V1 checkpoint: extension/checkpoints/mnist/full_bilinear_vae/model.pt
V2 checkpoint: checkpoints/mnist/model.pt (our current model)

Figure saved:
    figures/mnist/exp11_v1_failure.png
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE, BilinearLayer
from train    import load_checkpoint
from visualize import save_fig

V1_CKPT = "/home/v25/ippa6201/bilinear-mlp-repro/extension/checkpoints/mnist/full_bilinear_vae/model.pt"
V2_CKPT = "checkpoints/mnist/model.pt"
DATA    = "/home/v25/ippa6201/bilinear-mlp-repro/data"


class _FullBilinearVAE_v1(nn.Module):
    """Pure bilinear v1 — no skip connections. Suffers from f(z)=f(-z)."""
    def __init__(self, d_input=784, d_embed=256, d_hidden=512, d_latent=10):
        super().__init__()
        self.d_input  = d_input
        self.d_latent = d_latent
        self.embed    = nn.Linear(d_input,  d_embed,  bias=False)
        self.bilinear = BilinearLayer(d_embed, d_hidden)
        self.fc_mu    = nn.Linear(d_hidden, d_latent, bias=False)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        class _Dec(nn.Module):
            def __init__(s):
                super().__init__()
                s.embed_dec    = nn.Linear(d_latent, d_embed,  bias=False)
                s.bilinear_dec = BilinearLayer(d_embed, d_hidden)
                s.fc_out       = nn.Linear(d_hidden, d_input, bias=False)
            def forward(s, z):
                return torch.sigmoid(s.fc_out(s.bilinear_dec(s.embed_dec(z))))

        self.decoder = _Dec()

    def encode(self, x):
        return self.fc_mu(self.bilinear(self.embed(x))), \
               self.fc_logvar(self.bilinear(self.embed(x)))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(mu), mu, logvar



def _clean(ax):
    """Hide ticks and frame but keep axis labels (axis('off') erases labels)."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    v1 = _FullBilinearVAE_v1()
    torch.load(V1_CKPT, map_location="cpu", weights_only=True)
    ckpt1 = torch.load(V1_CKPT, map_location="cpu", weights_only=True)
    v1.load_state_dict(ckpt1["model_state"]); v1.eval()
    print(f"V1 loaded (epoch {ckpt1['epoch']})")

    v2 = FullBilinearVAE(); load_checkpoint(v2, V2_CKPT); v2.eval()

    # 1. Verify f(z) = f(-z) for V1
    torch.manual_seed(0)
    z_test = torch.randn(100, 10)
    with torch.no_grad():
        d_pos = v1.decode(z_test)
        d_neg = v1.decode(-z_test)
    max_diff = (d_pos - d_neg).abs().max().item()
    print(f"\nV1 symmetry check  max |f(z)-f(-z)| = {max_diff:.6f}  (should be ~0)")

    with torch.no_grad():
        d2_pos = v2.decode(z_test)
        d2_neg = v2.decode(-z_test)
    max_diff_v2 = (d2_pos - d2_neg).abs().max().item()
    print(f"V2 symmetry check  max |f(z)-f(-z)| = {max_diff_v2:.6f}  (should be >0)")

    # 2. Test ELBO comparison using saved history
    v1_hist = ckpt1.get("history", [])
    v2_hist = torch.load(V2_CKPT, map_location="cpu", weights_only=True).get("history", [])

    # 3. Get reconstructions and latent codes
    real_imgs, v1_recons, v2_recons = [], [], []
    v1_mus, v2_mus, labels_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            r1, mu1, _ = v1(x)
            r2, mu2, _ = v2(x)
            real_imgs.append(x); v1_recons.append(r1); v2_recons.append(r2)
            v1_mus.append(mu1); v2_mus.append(mu2); labels_list.append(y)
    real_imgs = torch.cat(real_imgs)
    v1_recons = torch.cat(v1_recons); v2_recons = torch.cat(v2_recons)
    v1_mus = torch.cat(v1_mus); v2_mus = torch.cat(v2_mus)
    labels = torch.cat(labels_list)

    v1_mse = ((real_imgs - v1_recons)**2).mean().item()
    v2_mse = ((real_imgs - v2_recons)**2).mean().item()
    print(f"\nReconstruction MSE: V1={v1_mse:.4f}  V2={v2_mse:.4f}")

    # Latent std per dim
    v1_std = v1_mus.std(0).numpy()
    v2_std = v2_mus.std(0).numpy()
    print(f"Latent std (V1): {v1_std.round(3)}")
    print(f"Latent std (V2): {v2_std.round(3)}")
    print(f"V1 symmetry verified: f(z)=f(-z), max diff={max_diff:.2e}")

    # Pick 5 examples per set for display
    idx = [torch.where(labels == c)[0][0].item() for c in range(10)]
    imgs_show   = real_imgs[idx].view(-1, 28, 28).numpy()
    v1_r_show   = v1_recons[idx].view(-1, 28, 28).numpy()
    v2_r_show   = v2_recons[idx].view(-1, 28, 28).numpy()
    # Decode(z) and Decode(-z) for V1
    with torch.no_grad():
        z_idx  = v1_mus[idx]
        d_pos_show = v1.decode(z_idx).view(-1, 28, 28).numpy()
        d_neg_show = v1.decode(-z_idx).view(-1, 28, 28).numpy()

    # Figure
    n = 10
    fig, axes = plt.subplots(5, n, figsize=(2.0 * n, 10),
                              gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    row_labels = ["real", "V1 recon", "V2 recon", "V1: f(z)", "V1: f(−z)\n≡f(z)"]
    rows = [imgs_show, v1_r_show, v2_r_show, d_pos_show, d_neg_show]

    for row_i, (row_data, rlbl) in enumerate(zip(rows, row_labels)):
        for col_i in range(n):
            axes[row_i, col_i].imshow(row_data[col_i], cmap="gray_r", vmin=0, vmax=1)
            _clean(axes[row_i, col_i])
            if row_i == 0:
                axes[row_i, col_i].set_title(f"d{col_i}", fontsize=9)
        axes[row_i, 0].set_ylabel(rlbl, fontsize=8, labelpad=4)

    fig.suptitle(f"Exp 11 — V1 failure: f(z)≡f(−z)\n"
                 f"V1 max|f(z)−f(−z)|={max_diff:.2e}  |  "
                 f"Recon MSE: V1={v1_mse:.4f}  V2={v2_mse:.4f}",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp11_v1_failure.png")

    # Figure 2: latent std comparison + training curves
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    x_k = range(10)
    axes2[0].bar([k - 0.2 for k in x_k], v1_std, 0.38, label="V1 (no skip)", color="indianred", alpha=0.85)
    axes2[0].bar([k + 0.2 for k in x_k], v2_std, 0.38, label="V2 (with skip)", color="steelblue", alpha=0.85)
    axes2[0].set_xlabel("Latent dimension"); axes2[0].set_ylabel("std across test set")
    axes2[0].set_title("Latent dimension usage\n(std ≈ 0 means dim is collapsed)", fontsize=10)
    axes2[0].legend(); axes2[0].grid(True, alpha=0.3, axis="y")

    if v1_hist:
        ep1 = [h["epoch"] for h in v1_hist]
        te1 = [h.get("test_total", h.get("test_total", None)) for h in v1_hist]
        te1 = [t for t in te1 if t is not None]
        if te1:
            axes2[1].plot(ep1[:len(te1)], te1, "o-", label="V1 test ELBO", color="indianred")
    if v2_hist:
        ep2 = [h["epoch"] for h in v2_hist]
        te2 = [h.get("test_total", None) for h in v2_hist]
        te2 = [t for t in te2 if t is not None]
        if te2:
            axes2[1].plot(ep2[:len(te2)], te2, "s-", label="V2 test ELBO", color="steelblue")
    axes2[1].set_xlabel("Epoch"); axes2[1].set_ylabel("Test ELBO (lower=better)")
    axes2[1].set_title("Training curves: V1 vs V2", fontsize=10)
    axes2[1].legend(); axes2[1].grid(True, alpha=0.3)

    fig2.suptitle("Exp 11 — V1 collapse diagnosis", fontsize=10, y=1.02)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp11_v1_diagnosis.png")


if __name__ == "__main__":
    main()
