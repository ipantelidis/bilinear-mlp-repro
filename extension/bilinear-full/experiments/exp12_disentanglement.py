"""
Exp 12 — Disentanglement: Latent-Class Correlation

For each model variant, measure how cleanly each latent dimension encodes
class identity.

Metrics:
  1. Correlation heatmap: |ρ(z_k, class_one_hot_c)| for all k,c — (10×10)
  2. R² per latent dim for predicting class (variance explained by linear classifier)
  3. MIG proxy: for each class, (best latent R² − second best R²) / H(class)
     Higher MIG → more disentangled

Models compared:
  - VanillaVAE    (extension/vanilla-vae/checkpoints/mnist/model.pt)
  - BilinearVAE   (extension/bilinear-encoder/checkpoints/mnist/model.pt)
  - DecBilinearVAE (extension/bilinear-decoder/checkpoints/mnist/model.pt)
  - FullBilinearVAE (checkpoints/mnist/model.pt)

Figure saved:
    figures/mnist/exp12_disentanglement.png
"""

import os, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from visualize import save_fig

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(_HERE, "..", "..", "..", "data")
ENC_DIR  = os.path.join(_HERE, "..", "..", "bilinear-encoder")
DEC_DIR  = os.path.join(_HERE, "..", "..", "bilinear-decoder")
CKPT_VANILLA = os.path.join(os.path.dirname(__file__),
                            "../../vanilla-vae/checkpoints/mnist/model.pt")


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── VanillaVAE matching the vanilla-vae checkpoint key layout ──────────────
# (same inline definition as exp05; encoder mirrors DecBilinearVAE's encoder,
#  decoder is a plain MLP: z(10) → Linear(256) → ReLU → Linear(784) → Sigmoid)
class VanillaVAE(nn.Module):
    def __init__(self, d_input=784, d_enc1=256, d_enc2=512, d_latent=10, d_dec=256):
        super().__init__()
        self.d_latent  = d_latent
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
                return torch.sigmoid(self.fc2(Fn.relu(self.fc1(z))))

        self.decoder = _Dec()

    def encode(self, x):
        h = Fn.relu(self.enc_fc1(x))
        h = Fn.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def load_vanilla(path):
    model = VanillaVAE()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {path}")
    return model


def _get_latents_labels(model, loader, device="cpu"):
    mus, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x.to(device))
            mus.append(mu.cpu()); lbls.append(y)
    return torch.cat(mus).numpy(), torch.cat(lbls).numpy()


def _correlation_matrix(mus, labels, n_classes=10):
    """Compute |ρ(z_k, 1[y==c])| for all k,c → (d_latent, n_classes)."""
    d = mus.shape[1]
    corr = np.zeros((d, n_classes))
    for c in range(n_classes):
        y_c = (labels == c).astype(float)
        for k in range(d):
            z_k = mus[:, k]
            r = np.corrcoef(z_k, y_c)[0, 1]
            corr[k, c] = abs(r)
    return corr


def _mig_proxy(corr):
    """
    MIG proxy: for each class c, gap between best and second-best latent correlation.
    Returns mean over classes.
    """
    gaps = []
    for c in range(corr.shape[1]):
        col = np.sort(corr[:, c])[::-1]
        gaps.append(col[0] - col[1] if len(col) > 1 else col[0])
    return float(np.mean(gaps))


def _effective_rank(mus):
    """Effective dimensionality of latent distribution."""
    mu_c = mus - mus.mean(0)
    cov  = (mu_c.T @ mu_c) / (len(mus) - 1)
    eigs = np.linalg.eigvalsh(cov).clip(min=0)
    eigs = eigs[::-1]
    p    = eigs / (eigs.sum() + 1e-10)
    return float(np.exp(-(p * np.log(p + 1e-10)).sum()))


def main():
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    models = {}

    try:
        models["VanillaVAE"] = load_vanilla(CKPT_VANILLA)
    except Exception as e:
        print(f"  VanillaVAE not loaded: {e}")

    try:
        enc_mod = _load_module("enc_models", os.path.join(ENC_DIR, "models.py"))
        enc_tr  = _load_module("enc_train",  os.path.join(ENC_DIR, "train.py"))
        enc = enc_mod.BilinearVAE()
        enc_tr.load_checkpoint(enc, os.path.join(ENC_DIR, "checkpoints/mnist/model.pt"))
        models["BilinearVAE\n(enc)"] = enc
    except Exception as e:
        print(f"  BilinearVAE not loaded: {e}")

    try:
        dec_mod = _load_module("dec_models", os.path.join(DEC_DIR, "models.py"))
        dec_tr  = _load_module("dec_train",  os.path.join(DEC_DIR, "train.py"))
        dec = dec_mod.DecBilinearVAE()
        dec_tr.load_checkpoint(dec, os.path.join(DEC_DIR, "checkpoints/mnist/model.pt"))
        models["DecBilinearVAE\n(dec)"] = dec
    except Exception as e:
        print(f"  DecBilinearVAE not loaded: {e}")

    full = FullBilinearVAE(); load_checkpoint(full, "checkpoints/mnist/model.pt")
    models["FullBilinearVAE\n(enc+dec)"] = full

    # Compute metrics
    print(f"\n{'Model':<26}  {'MIG proxy':>10}  {'eff rank':>10}  {'max corr':>10}")
    print("-" * 62)

    all_corr = {}
    mig_scores = {}
    eff_ranks  = {}

    for name, model in models.items():
        model.eval()
        mus, labels = _get_latents_labels(model, loader)
        corr = _correlation_matrix(mus, labels)
        mig  = _mig_proxy(corr)
        er   = _effective_rank(mus)
        all_corr[name] = corr
        mig_scores[name] = mig
        eff_ranks[name]  = er
        label = name.replace("\n", " ")
        print(f"  {label:<24}  {mig:>10.4f}  {er:>10.2f}  {corr.max():>10.4f}")

    # Figure: correlation heatmaps side-by-side
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4.5))
    if n_models == 1: axes = [axes]

    for ax, (name, corr) in zip(axes, all_corr.items()):
        im = ax.imshow(corr, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
        ax.set_xticks(range(10)); ax.set_xticklabels([f"d{c}" for c in range(10)], fontsize=8)
        ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels([f"z{k}" for k in range(corr.shape[0])], fontsize=8)
        ax.set_xlabel("class", fontsize=9); ax.set_ylabel("latent dim", fontsize=9)
        mig = mig_scores[name]; er = eff_ranks[name]
        ax.set_title(f"{name.replace(chr(10), ' ')}\nMIG={mig:.3f}  eff_rank={er:.2f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Exp 12 — Disentanglement: |ρ(z_k, 1[y==c])|", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp12_disentanglement.png")

    # Figure 2: MIG proxy bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    names    = list(mig_scores.keys())
    vals     = [mig_scores[n] for n in names]
    colors   = ["steelblue", "indianred", "seagreen", "darkorchid"][:len(names)]
    ax2.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace("\n", "\n") for n in names], fontsize=9)
    ax2.set_ylabel("MIG proxy (class-latent correlation gap)", fontsize=10)
    ax2.set_title("Exp 12 — Disentanglement score: higher = more disentangled", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp12_mig_proxy.png")


if __name__ == "__main__":
    main()
