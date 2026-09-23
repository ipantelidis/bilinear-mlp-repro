"""
Exp 09 — Encoder vs Decoder Cross-Class Similarity

Three-way comparison of top-eigenvector cross-class |cos| similarity:
  (a) Decoder, raw class-mean targets p* = mean_c        (DecBilinearVAE)
  (b) Decoder, centered targets p* = mean_c − global mean (exp13 protocol)
  (c) Encoder, class-mean latent encodings μ* = mean_c(μ) (BilinearVAE)

Key numbers (MNIST): decoder raw 0.842 ≫ decoder centered 0.429 ≳ encoder
0.350. The decoder's "near-universal generative direction" is mostly an
artefact of raw-target overlap (see exp13/exp14); with centered targets the
decoder is nearly as class-discriminative as the encoder.

Note: earlier versions of this experiment computed the encoder side with
latent BASIS directions μ* = e_k (mean 0.232) but labeled them d0..d9 as if
they were digit classes. The encoder side now uses class-mean latent
encodings, which is the like-for-like comparison.

Also included: interpolation of the synthesised image as p* morphs from
class A to class B, confirming the decoder transition is smooth.

Outputs:
    figures/mnist/exp09_crossclass_comparison.png
    figures/mnist/exp09_interpolation.png
    figures/exp09_results.json
"""

import os
import json
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
DATA     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")

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


def _encoder_crossclass(enc_model, enc_analysis, class_mu):
    """Encoder cross-class: top +eigvec of Q for each class-mean latent μ_c."""
    classes  = sorted(class_mu.keys())
    # support both naming conventions across encoder analysis versions
    _get_Q   = getattr(enc_analysis, "interaction_matrix",
                getattr(enc_analysis, "get_interaction_matrix", None))
    _decomp  = enc_analysis.decompose
    top_vecs = {}
    with torch.no_grad():
        for c in classes:
            Q = _get_Q(enc_model, class_mu[c])
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

    def _offdiag(mat):
        off = mat[~np.eye(mat.shape[0], dtype=bool)]
        return float(off.mean()), float(off.min()), float(off.max())

    dec_mat, top_vecs = _decoder_crossclass(dec_model, mean_imgs)
    dec_mean, dec_min, dec_max = _offdiag(dec_mat)
    print(f"Decoder raw targets:      mean={dec_mean:.3f}  min={dec_min:.3f}  max={dec_max:.3f}")

    global_mean  = torch.stack([mean_imgs[c] for c in classes]).mean(0)
    centered     = {c: mean_imgs[c] - global_mean for c in classes}
    cen_mat, _   = _decoder_crossclass(dec_model, centered)
    cen_mean, cen_min, cen_max = _offdiag(cen_mat)
    print(f"Decoder centered targets: mean={cen_mean:.3f}  min={cen_min:.3f}  max={cen_max:.3f}")

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

        class_mu = enc_analysis.class_means(enc_model, loader)
        enc_mat  = _encoder_crossclass(enc_model, enc_analysis, class_mu)
        enc_mean, enc_min, enc_max = _offdiag(enc_mat)
        print(f"Encoder class-mean μ:     mean={enc_mean:.3f}  min={enc_min:.3f}  max={enc_max:.3f}")
    except Exception as e:
        print(f"  Could not load encoder: {e}")

    results = {"decoder_raw":      {"mean": dec_mean, "min": dec_min, "max": dec_max},
               "decoder_centered": {"mean": cen_mean, "min": cen_min, "max": cen_max}}
    if enc_mat is not None:
        results["encoder_class_mean_mu"] = {"mean": enc_mean, "min": enc_min, "max": enc_max}
    with open("figures/exp09_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved figures/exp09_results.json")

    # ── Figure 1: three-way heatmaps ─────────────────────────────────────
    panels = [(dec_mat, f"Decoder, raw targets p*=mean_c\nmean |cos| = {dec_mean:.3f}"),
              (cen_mat, f"Decoder, centered targets (exp13)\nmean |cos| = {cen_mean:.3f}")]
    if enc_mat is not None:
        panels.append((enc_mat, f"Encoder, class-mean μ targets\nmean |cos| = {enc_mean:.3f}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]

    lbls = [f"d{c}" for c in classes]
    for ax, (mat, title) in zip(axes, panels):
        im = similarity_heatmap(ax, mat, lbls, title=title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Exp 09 — Top eigenvector cross-class similarity:\n"
                 "the raw-target decoder overlap is mostly a target artefact — "
                 "centered decoder is near encoder level",
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
