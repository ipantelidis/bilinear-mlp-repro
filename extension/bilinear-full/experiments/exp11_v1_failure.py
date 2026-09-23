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

    # Pick 6 examples for display (diverse digits with a clear v1/v2 contrast)
    show = [0, 3, 4, 5, 7, 8]
    idx = [torch.where(labels == c)[0][0].item() for c in show]
    imgs_show   = real_imgs[idx].view(-1, 28, 28).numpy()
    v1_r_show   = v1_recons[idx].view(-1, 28, 28).numpy()
    v2_r_show   = v2_recons[idx].view(-1, 28, 28).numpy()
    # Decode(z) and Decode(-z) for V1
    with torch.no_grad():
        z_idx  = v1_mus[idx]
        d_pos_show = v1.decode(z_idx).view(-1, 28, 28).numpy()
        d_neg_show = v1.decode(-z_idx).view(-1, 28, 28).numpy()

    # Figure: 5 image rows in two visual groups — reconstruction quality
    # (input / v1 / v2) on top, the even-symmetry identity (f(z) vs f(-z))
    # boxed below.  All panels come straight from the saved checkpoints.
    n = len(show)
    fig = plt.figure(figsize=(6.3, 4.35))
    gs = fig.add_gridspec(6, n, height_ratios=[1, 1, 1, 0.28, 1, 1],
                          hspace=0.06, wspace=0.05,
                          left=0.225, right=0.985, top=0.955, bottom=0.075)

    row_specs = [
        (0, imgs_show,  "input"),
        (1, v1_r_show,  f"v1 recon\nMSE {v1_mse:.3f}"),
        (2, v2_r_show,  f"v2 recon (+skip)\nMSE {v2_mse:.3f}"),
        (4, d_pos_show, "v1 decode(z)"),
        (5, d_neg_show, "v1 decode(−z)"),
    ]
    axd = {}
    for gr, row_data, rlbl in row_specs:
        for col_i in range(n):
            ax = fig.add_subplot(gs[gr, col_i])
            ax.imshow(row_data[col_i], cmap="gray_r", vmin=0, vmax=1)
            _clean(ax)
            if gr == 0:
                ax.set_title(f"{show[col_i]}", fontsize=8, pad=2)
            if col_i == 0:
                ax.text(-0.18, 0.5, rlbl, transform=ax.transAxes,
                        ha="right", va="center", fontsize=8.5,
                        fontweight="bold", linespacing=1.35)
            axd[(gr, col_i)] = ax

    # Box the last two rows: they are pixel-identical by the even-symmetry
    # identity of the pure bilinear decoder.
    p_tl = axd[(4, 0)].get_position()
    p_br = axd[(5, n - 1)].get_position()
    pad_x, pad_y = 0.010, 0.014
    x0, x1 = p_tl.x0 - pad_x, p_br.x1 + pad_x
    y0, y1 = p_br.y0 - pad_y, p_tl.y1 + pad_y
    fig.add_artist(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 transform=fig.transFigure, fill=False,
                                 edgecolor="firebrick", linewidth=1.2,
                                 zorder=5))
    diff_str = "0" if max_diff == 0 else f"{max_diff:.1e}"
    fig.text(0.5 * (x0 + x1), y0 - 0.012,
             f"pixel-identical: max |f(z) − f(−z)| = {diff_str}",
             ha="center", va="top", fontsize=8.5, fontweight="bold",
             color="firebrick")
    save_fig(fig, "figures/mnist/exp11_v1_failure.png", dpi=300)

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
