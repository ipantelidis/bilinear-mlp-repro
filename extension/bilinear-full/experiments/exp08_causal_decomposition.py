"""
Exp 08 — Causal Accuracy Decomposition

FullBilinearVAE achieves 6/10 causal accuracy (exp04) vs DecBilinearVAE's 3/10.
This experiment isolates *why* it improves by swapping components:

  Condition A — DecBilinearVAE decoder + DecBilinearVAE encoder:   baseline (3/10)
  Condition B — DecBilinearVAE decoder + FullBilinearVAE encoder:  encoder contribution
  Condition C — FullBilinearVAE decoder + FullBilinearVAE encoder: full model (6/10)

For conditions B and C we use the same synthesised images (from DecBilinearVAE decoder)
but re-encode them with the FullBilinearVAE encoder.  If B > A, the bilinear encoder
alone is responsible.  If C > B, joint training of both sides adds further gain.

Procedure:
  1. Load DecBilinearVAE (from ../bilinear-decoder/checkpoints/mnist/model.pt)
  2. Load FullBilinearVAE (from checkpoints/mnist/model.pt)
  3. Generate 10 synth images using DecBilinearVAE decoder
  4. Compute class-mean latent codes for both models
  5. Encode each synth image with:
       (a) DecBilinearVAE encoder → nearest DecBilinear class mean
       (b) FullBilinearVAE encoder → nearest Full class mean
  6. Condition C: regenerate with FullBilinearVAE decoder, re-encode, nearest Full mean

Figure saved:
    figures/mnist/exp08_causal_decomposition.png
"""

import os
import importlib.util
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, decompose, mean_lat_norm)
from visualize import save_fig

DATA    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
DEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bilinear-decoder")


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _class_means(model, loader, device="cpu"):
    buckets = {}
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x.to(device))
            for i, lbl in enumerate(y.tolist()):
                buckets.setdefault(lbl, []).append(mu[i].cpu())
    return {c: torch.stack(v).mean(0) for c, v in buckets.items()}


def _nearest(mu, class_means):
    dists = {c: (mu - m).norm().item() for c, m in class_means.items()}
    return min(dists, key=dists.get)


def _synth_dec_images(model, mean_imgs, scale):
    """Synthesise one image per class using decoder quadratic eigvec."""
    synth = {}
    with torch.no_grad():
        for c in sorted(mean_imgs.keys()):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            z = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            synth[c] = model.decode(z.unsqueeze(0)).squeeze(0).detach()
    return synth


def _run_causal_encode(encode_fn, class_means, synth_imgs):
    """Encode synth images and find nearest class mean."""
    results = []
    classes = sorted(synth_imgs.keys())
    with torch.no_grad():
        for c in classes:
            mu, _ = encode_fn(synth_imgs[c].unsqueeze(0))
            mu    = mu.squeeze(0)
            nearest = _nearest(mu, class_means)
            cos = float(torch.cosine_similarity(mu.unsqueeze(0),
                                                 class_means[c].unsqueeze(0)))
            results.append({"class": c, "nearest": nearest,
                             "correct": nearest == c, "cos": cos})
    return results


def main():
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    # Load pixel-space class means
    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}

    # --- Load DecBilinearVAE ---
    dec_models_mod = _load_module("dec_models", os.path.join(DEC_DIR, "models.py"))
    dec_train_mod  = _load_module("dec_train",  os.path.join(DEC_DIR, "train.py"))
    dec_model = dec_models_mod.DecBilinearVAE()
    dec_train_mod.load_checkpoint(dec_model, os.path.join(DEC_DIR, "checkpoints/mnist/model.pt"))
    dec_model.eval()

    # --- Load FullBilinearVAE ---
    full_model = FullBilinearVAE(); load_checkpoint(full_model, "checkpoints/mnist/model.pt")
    full_model.eval()

    # Scales and class means
    dec_scale  = mean_lat_norm(dec_model,  loader)
    full_scale = mean_lat_norm(full_model, loader)

    dec_class_means  = _class_means(dec_model,  loader)
    full_class_means = _class_means(full_model, loader)

    # Condition A: dec decoder + dec encoder (baseline)
    dec_synth = _synth_dec_images(dec_model, mean_imgs, dec_scale)
    res_A = _run_causal_encode(dec_model.encode, dec_class_means, dec_synth)

    # Condition B: dec decoder + full encoder
    res_B = _run_causal_encode(full_model.encode, full_class_means, dec_synth)

    # Condition C: full decoder + full encoder
    full_synth = _synth_dec_images(full_model, mean_imgs, full_scale)
    res_C = _run_causal_encode(full_model.encode, full_class_means, full_synth)

    n_A = sum(r["correct"] for r in res_A)
    n_B = sum(r["correct"] for r in res_B)
    n_C = sum(r["correct"] for r in res_C)
    classes = sorted(mean_imgs.keys())

    print(f"\nCausal accuracy decomposition:")
    print(f"  Condition A (dec dec + dec enc):  {n_A}/10")
    print(f"  Condition B (dec dec + full enc): {n_B}/10   ← encoder contribution")
    print(f"  Condition C (full dec + full enc):{n_C}/10   ← full model (exp04)")
    print(f"\n  {'Class':<8} {'Cond A':>8} {'Cond B':>8} {'Cond C':>8}")
    print("  " + "-" * 34)
    for i, c in enumerate(classes):
        def fmt(r): return "✓" if r["correct"] else f"→{r['nearest']}"
        print(f"  d{c}      {fmt(res_A[i]):>8} {fmt(res_B[i]):>8} {fmt(res_C[i]):>8}")

    # Figure
    n   = len(classes)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.5),
                              gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    def fill_row(results, synth_imgs_dict, row_img, row_lbl, label):
        cs = sorted(synth_imgs_dict.keys())
        for col, c in enumerate(cs):
            axes[row_img, col].imshow(synth_imgs_dict[c].view(28, 28).numpy(),
                                      cmap="gray_r", vmin=0, vmax=1)
            axes[row_img, col].axis("off")
            if row_img == 0:
                axes[row_img, col].set_title(f"d{c}", fontsize=9)
            r = results[col]
            tick  = "✓" if r["correct"] else f"→{r['nearest']}"
            color = "darkgreen" if r["correct"] else "firebrick"
            axes[row_lbl, col].text(0.5, 0.65, tick, ha="center", va="center",
                fontsize=12, color=color, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].text(0.5, 0.2, f"cos={r['cos']:.2f}", ha="center",
                va="center", fontsize=7, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].axis("off")
        axes[row_img, 0].set_ylabel(f"{label}\nsynth", fontsize=8, labelpad=4)
        nc = sum(r["correct"] for r in results)
        axes[row_lbl, 0].set_ylabel(f"({nc}/10)", fontsize=8, labelpad=4)

    # Rows 0-1: Conditions A & B share the same dec_synth images
    fill_row(res_A, dec_synth,  0, 1, "A: dec+dec")
    # For condition B: same images, different labels
    for col, (c, r) in enumerate(zip(classes, res_B)):
        tick  = "✓" if r["correct"] else f"→{r['nearest']}"
        color = "darkgreen" if r["correct"] else "firebrick"
        axes[1, col].text(0.5, 0.35, f"B:{tick}", ha="center", va="center",
            fontsize=10, color=color, transform=axes[1, col].transAxes)

    # Rows 2-3: Condition C
    fill_row(res_C, full_synth, 2, 3, "C: full+full")

    fig.add_artist(plt.Line2D([0.02, 0.98], [0.505, 0.505],
                               transform=fig.transFigure, color="gray",
                               linestyle="--", linewidth=0.9))
    fig.suptitle(f"Exp 08 — Causal accuracy decomposition\n"
                 f"A (dec+dec): {n_A}/10   B (dec+full-enc): {n_B}/10   C (full): {n_C}/10",
                 fontsize=11, y=1.01)
    save_fig(fig, "figures/mnist/exp08_causal_decomposition.png")


if __name__ == "__main__":
    main()
