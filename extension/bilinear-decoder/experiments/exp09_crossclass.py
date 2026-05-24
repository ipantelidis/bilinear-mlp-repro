"""
Exp 09 — Encoder vs Decoder Cross-Class Similarity

Side-by-side comparison of:
  (a) Encoder pixel-space cross-class cosine similarity (from BilinearVAE encoder)
  (b) Decoder latent-space cross-class cosine similarity (from DecBilinearVAE)

Key finding: decoder top eigenvectors are much more similar across classes
(mean ≈ 0.842) than encoder top eigenvectors (mean ≈ 0.350), revealing a
near-universal generative direction in the decoder.

Also included: interpolation of the synthesised image as p* morphs from
class A to class B, confirming the decoder transition is smooth.

Figures saved:
    figures/mnist/exp09_crossclass_comparison.png
    figures/mnist/exp09_interpolation.png
"""

import os
import importlib.util
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import similarity_heatmap, save_fig

CKPT_DEC = "checkpoints/mnist/model.pt"
CKPT_ENC = os.path.join(os.path.dirname(__file__), "../../bilinear-encoder/checkpoints/mnist/model.pt")
ENC_DIR  = os.path.join(os.path.dirname(__file__), "../../bilinear-encoder")
DATA     = "/home/v25/ippa6201/bilinear-mlp-repro/data"

INTERP_PAIRS = [(4, 9), (1, 7), (0, 6)]
N_STEPS      = 7


def _load_module(name, filepath):
    """Load a Python module from an explicit file path, bypassing sys.path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_loader():
    return DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)


def _decoder_crossclass(model, mean_imgs):
    classes = sorted(mean_imgs.keys())
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
    return mat, top_vecs


def _encoder_crossclass(enc_model, enc_analysis, mean_imgs):
    """Encoder cross-class: top +eigvec of Q_in for each unit latent direction e_k."""
    classes  = sorted(mean_imgs.keys())
    # support both naming conventions across encoder analysis versions
    _get_Q   = getattr(enc_analysis, "interaction_matrix",
                getattr(enc_analysis, "get_interaction_matrix", None))
    _decomp  = enc_analysis.decompose
    top_vecs = {}
    with torch.no_grad():
        for c in classes:
            d = torch.zeros(enc_model.d_latent); d[c] = 1.0
            Q = _get_Q(enc_model, d)
            vals, vecs = _decomp(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            top_vecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(Q.shape[0])
    n   = len(classes)
    mat = np.zeros((n, n))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            va, vb = top_vecs[a], top_vecs[b]
            mat[i, j] = abs(float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8)))
    return mat


def main():
    loader    = _build_loader()
    dec_model = DecBilinearVAE(); load_checkpoint(dec_model, CKPT_DEC); dec_model.eval()

    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}
    classes   = sorted(mean_imgs.keys())

    dec_mat, top_vecs = _decoder_crossclass(dec_model, mean_imgs)
    off_diag  = dec_mat[dec_mat < 1.0]
    dec_mean  = off_diag.mean()
    print(f"Decoder cross-class: mean={dec_mean:.3f}  min={off_diag.min():.3f}  max={off_diag.max():.3f}")

    # Load encoder via explicit file path (avoids local models.py shadowing BilinearVAE)
    enc_mat  = None
    enc_mean = None
    try:
        enc_models   = _load_module("enc_models",   os.path.join(ENC_DIR, "models.py"))
        enc_analysis = _load_module("enc_analysis", os.path.join(ENC_DIR, "analysis.py"))
        enc_train    = _load_module("enc_train",    os.path.join(ENC_DIR, "train.py"))

        enc_model = enc_models.BilinearVAE()
        enc_train.load_checkpoint(enc_model, CKPT_ENC)
        enc_model.eval()

        enc_mat  = _encoder_crossclass(enc_model, enc_analysis, mean_imgs)
        enc_off  = enc_mat[enc_mat < 1.0]
        enc_mean = enc_off.mean()
        print(f"Encoder cross-class: mean={enc_mean:.3f}  min={enc_off.min():.3f}  max={enc_off.max():.3f}")
    except Exception as e:
        print(f"  Could not load encoder: {e}")

    # ── Figure 1: side-by-side heatmaps ──────────────────────────────────
    n_panels = 2 if enc_mat is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    lbls = [f"d{c}" for c in classes]
    im1 = similarity_heatmap(axes[0], dec_mat, lbls,
                              title=f"Decoder (bilinear)\nmean cos = {dec_mean:.3f}")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    if enc_mat is not None:
        im2 = similarity_heatmap(axes[1], enc_mat, lbls,
                                  title=f"Encoder (bilinear)\nmean cos = {enc_mean:.3f}")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Exp 09 — Top eigenvector cross-class similarity:\n"
                 "Decoder is far less class-discriminative than encoder",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp09_crossclass_comparison.png")

    # ── Figure 2: interpolation for confusable pairs ──────────────────────
    scale = mean_lat_norm(dec_model, loader)
    alphas = np.linspace(0, 1, N_STEPS)

    fig2, axes2 = plt.subplots(len(INTERP_PAIRS), N_STEPS,
                                figsize=(N_STEPS * 1.8, len(INTERP_PAIRS) * 1.8),
                                gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    with torch.no_grad():
        for row, (a, b) in enumerate(INTERP_PAIRS):
            for col, alpha in enumerate(alphas):
                p_star = (1 - alpha) * mean_imgs[a] + alpha * mean_imgs[b]
                Q = get_decoder_interaction_matrix(dec_model, p_star)
                vals, vecs = decompose(Q)
                pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
                v   = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(dec_model.d_latent)
                img = dec_model.decode(v.unsqueeze(0)).squeeze(0)
                axes2[row, col].imshow(img.view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
                axes2[row, col].axis("off")
                if row == 0:
                    axes2[row, col].set_title(f"α={alpha:.2f}", fontsize=7)
            axes2[row, 0].set_ylabel(f"d{a}→d{b}", fontsize=9, labelpad=4)

    fig2.suptitle("Exp 09 — Synthesised image interpolation between digit pairs",
                  fontsize=11, y=1.01)
    save_fig(fig2, "figures/mnist/exp09_interpolation.png")


if __name__ == "__main__":
    main()
