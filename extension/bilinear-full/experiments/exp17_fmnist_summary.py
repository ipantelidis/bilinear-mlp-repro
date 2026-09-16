"""
Exp 17 — FashionMNIST Key Metrics Summary

Runs the four key experiments on FashionMNIST FullBilinearVAE to test
whether findings generalise beyond MNIST:

  (A) Near-universal direction (exp02 analogue):
      mean cross-class cos of Q_dec top eigvecs

  (B) Causal generation accuracy (exp04 analogue):
      how many of 10 classes are correctly retrieved?

  (C) Latent alignment (exp07 analogue):
      mean enc↔dec cos in latent space

  (D) Cross-seed consistency (exp06 analogue):
      decoder synthesis and encoder eigvec consistency across 5 seeds

Requires:
    checkpoints/fashion_mnist/model.pt
    checkpoints/fashion_mnist/seeds/seed0..4.pt

Run train_fmnist.py first if these don't exist.

Figure saved:
    figures/fashion_mnist/exp17_fmnist_summary.png
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, get_encoder_interaction_matrix,
                      decompose, mean_lat_norm, compute_class_means)
from visualize import save_fig

CKPT  = "checkpoints/fashion_mnist/model.pt"
SEEDS = [0, 1, 2, 3, 4]
DATA  = "/home/v25/ippa6201/bilinear-mlp-repro/data"

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def _cos(a, b):
    return float(torch.dot(a.float(), b.float()) / (a.float().norm() * b.float().norm() + 1e-8))


def _near_universal_cos(model, mean_imgs, classes):
    eigvecs = {}
    for c in classes:
        Q = get_decoder_interaction_matrix(model, mean_imgs[c])
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        eigvecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_latent)
    pairs = list(combinations(classes, 2))
    return float(np.mean([abs(_cos(eigvecs[a], eigvecs[b])) for a, b in pairs])), eigvecs


def _causal_accuracy(model, mean_imgs, class_means, scale, classes):
    correct = 0; results = []
    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            z = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            x_synth = model.decode(z.unsqueeze(0)).squeeze(0)
            mu_back, _ = model.encode(x_synth.unsqueeze(0))
            mu_back = mu_back.squeeze(0)
            dists   = {lbl: (mu_back - m).norm().item() for lbl, m in class_means.items()}
            nearest = min(dists, key=dists.get)
            ok = nearest == c; correct += int(ok)
            results.append({"class": c, "nearest": nearest, "correct": ok,
                             "synth": x_synth.detach()})
    return correct, results


def _latent_alignment(model, mean_imgs, scale, classes):
    enc_codes = {}; dec_eigvecs = {}
    with torch.no_grad():
        for c in classes:
            d = torch.zeros(model.d_latent); d[c] = 1.0
            Q_enc = get_encoder_interaction_matrix(model, d)
            vals_e, vecs_e = decompose(Q_enc)
            pos_e = (vals_e > 0).nonzero(as_tuple=True)[0]
            eigvec_img = vecs_e[pos_e[0]] if len(pos_e) else torch.zeros(model.d_input)
            v = eigvec_img - eigvec_img.min()
            v = v / (v.max() + 1e-8)
            mu, _ = model.encode(v.unsqueeze(0))
            enc_codes[c] = mu.squeeze(0)
            Q_dec = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals_d, vecs_d = decompose(Q_dec)
            pos_d = (vals_d > 0).nonzero(as_tuple=True)[0]
            dec_eigvecs[c] = vecs_d[pos_d[0]] * scale if len(pos_d) else torch.zeros(model.d_latent)
    return float(np.mean([_cos(enc_codes[c], dec_eigvecs[c]) for c in classes]))


def main():
    if not Path(CKPT).exists():
        print(f"Checkpoint not found: {CKPT}")
        print("Run: python train_fmnist.py")
        return

    model = FullBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

    loader = DataLoader(
        datasets.FashionMNIST(DATA, train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs   = {c: torch.stack(v).mean(0) for c, v in buckets.items()}
    class_means = compute_class_means(model, loader)
    scale       = mean_lat_norm(model, loader)
    classes     = sorted(mean_imgs.keys())

    # (A) Near-universal cos
    nuc, dec_eigvecs = _near_universal_cos(model, mean_imgs, classes)
    print(f"\n(A) Near-universal cos: {nuc:.4f}  (MNIST: 0.808, DecBilinear FMNIST: 0.910)")

    # (B) Causal accuracy
    n_correct, causal_results = _causal_accuracy(model, mean_imgs, class_means, scale, classes)
    print(f"(B) Causal accuracy:    {n_correct}/10  (MNIST: 6/10)")
    for r in causal_results:
        tick = "✓" if r["correct"] else f"→{r['nearest']}"
        print(f"    {CLASS_NAMES[r['class']]}: {tick}")

    # (C) Latent alignment
    align = _latent_alignment(model, mean_imgs, scale, classes)
    print(f"(C) Latent alignment:   {align:.4f}  (MNIST: 0.007)")

    # (D) Cross-seed consistency — requires seed checkpoints
    seed_ckpts = [f"checkpoints/fashion_mnist/seeds/seed{s}.pt" for s in SEEDS]
    if all(Path(p).exists() for p in seed_ckpts):
        seed_models = []
        for s in SEEDS:
            m = FullBilinearVAE(); load_checkpoint(m, seed_ckpts[s]); m.eval()
            seed_models.append(m)

        dec_cos_pairs = []; enc_cos_pairs = []
        for ma, mb in combinations(range(len(SEEDS)), 2):
            for c in classes:
                # Decoder synth cosine
                def get_synth(mod, c):
                    Q = get_decoder_interaction_matrix(mod, mean_imgs[c])
                    vals, vecs = decompose(Q)
                    pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
                    z = vecs[pos_idx[0]] * mean_lat_norm(mod, loader) if len(pos_idx) else torch.zeros(mod.d_latent)
                    return mod.decode(z.unsqueeze(0)).squeeze(0).detach()

                img_a = get_synth(seed_models[ma], c)
                img_b = get_synth(seed_models[mb], c)
                dec_cos_pairs.append(abs(_cos(img_a, img_b)))

                # Encoder eigvec cosine
                def get_enc_eigvec(mod, c):
                    d = torch.zeros(mod.d_latent); d[c] = 1.0
                    Q = get_encoder_interaction_matrix(mod, d)
                    vals, vecs = decompose(Q)
                    pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
                    return vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(mod.d_input)

                ev_a = get_enc_eigvec(seed_models[ma], c)
                ev_b = get_enc_eigvec(seed_models[mb], c)
                enc_cos_pairs.append(abs(_cos(ev_a, ev_b)))

        dec_cons = float(np.mean(dec_cos_pairs))
        enc_cons = float(np.mean(enc_cos_pairs))
        print(f"(D) Decoder seed consistency: {dec_cons:.4f}  (MNIST: 0.941)")
        print(f"    Encoder seed consistency: {enc_cons:.4f}  (MNIST: 0.326)")
    else:
        dec_cons = enc_cons = float("nan")
        print(f"(D) Seed checkpoints not found, skipping consistency")

    Path("figures/fashion_mnist").mkdir(parents=True, exist_ok=True)
    with open("figures/fashion_mnist/exp17_results.json", "w") as f:
        json.dump({"near_universal_cos": nuc,
                   "causal_accuracy": n_correct,
                   "causal_per_class": [{"class": r["class"],
                                         "name": CLASS_NAMES[r["class"]],
                                         "nearest": r["nearest"],
                                         "correct": bool(r["correct"])}
                                        for r in causal_results],
                   "latent_alignment": align,
                   "decoder_seed_consistency": dec_cons,
                   "encoder_seed_consistency": enc_cons}, f, indent=2)

    # Figure
    fig = plt.figure(figsize=(16, 10))

    # Panel A: near-universal cos heatmap
    ax_a = fig.add_axes([0.04, 0.55, 0.20, 0.38])
    pairs = list(combinations(classes, 2))
    pair_cos = np.zeros((len(classes), len(pairs)))
    for ci, c in enumerate(classes):
        for pi, (a, b) in enumerate(pairs):
            def gv(mod, c2):
                Q = get_decoder_interaction_matrix(mod, mean_imgs[c2])
                vals, vecs = decompose(Q)
                pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
                return vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(mod.d_latent)
            # use dec_eigvecs already computed
            pass
    # Simpler: just show the scalar NUC
    ax_a.text(0.5, 0.5, f"Near-universal\ncos\n\n{nuc:.4f}\n\n(MNIST: 0.808)\n(Dec-only FMNIST: 0.910)",
              ha="center", va="center", fontsize=14, transform=ax_a.transAxes,
              bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.6))
    ax_a.axis("off"); ax_a.set_title("(A) Near-universal direction", fontsize=10)

    # Panel B: causal accuracy
    ax_b = fig.add_axes([0.28, 0.55, 0.20, 0.38])
    ax_b.text(0.5, 0.6, f"Causal accuracy\n\n{n_correct}/10",
              ha="center", va="center", fontsize=16, transform=ax_b.transAxes,
              bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6))
    ax_b.text(0.5, 0.15, "(MNIST: 6/10)", ha="center", va="center",
              fontsize=10, transform=ax_b.transAxes, color="gray")
    ax_b.axis("off"); ax_b.set_title("(B) Causal accuracy", fontsize=10)

    # Panel C: alignment
    ax_c = fig.add_axes([0.52, 0.55, 0.20, 0.38])
    ax_c.text(0.5, 0.5, f"Latent enc↔dec\nalignment\n\n{align:.4f}\n\n(MNIST: 0.007)",
              ha="center", va="center", fontsize=13, transform=ax_c.transAxes,
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))
    ax_c.axis("off"); ax_c.set_title("(C) Latent alignment", fontsize=10)

    # Panel D: consistency
    ax_d = fig.add_axes([0.76, 0.55, 0.20, 0.38])
    txt = (f"Decoder cons: {dec_cons:.4f}\nEncoder cons: {enc_cons:.4f}"
           if not np.isnan(dec_cons) else "Seed checkpoints\nnot available")
    ax_d.text(0.5, 0.5, f"Cross-seed consistency\n\n{txt}\n\n(MNIST dec: 0.941)\n(MNIST enc: 0.326)",
              ha="center", va="center", fontsize=11, transform=ax_d.transAxes,
              bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.6))
    ax_d.axis("off"); ax_d.set_title("(D) Seed consistency", fontsize=10)

    # Lower panel: synthesised images
    ax_imgs = fig.add_axes([0.04, 0.05, 0.92, 0.40])
    ax_imgs.axis("off")
    ax_imgs.set_title("Causal synthesis: Q_dec top eigvec decoded (FashionMNIST)", fontsize=10, pad=4)
    for ci, r in enumerate(causal_results):
        inner = fig.add_axes([0.04 + ci * 0.091, 0.07, 0.082, 0.32])
        inner.imshow(r["synth"].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
        inner.axis("off")
        tick  = "✓" if r["correct"] else f"→{r['nearest']}"
        color = "darkgreen" if r["correct"] else "firebrick"
        inner.set_title(f"{CLASS_NAMES[r['class']][:6]}\n{tick}", fontsize=7, color=color)

    fig.suptitle(f"Exp 17 — FashionMNIST FullBilinearVAE summary\n"
                 f"NUC={nuc:.3f}  causal={n_correct}/10  align={align:.3f}",
                 fontsize=11, y=1.01)

    Path("figures/fashion_mnist").mkdir(parents=True, exist_ok=True)
    save_fig(fig, "figures/fashion_mnist/exp17_fmnist_summary.png")


if __name__ == "__main__":
    main()
