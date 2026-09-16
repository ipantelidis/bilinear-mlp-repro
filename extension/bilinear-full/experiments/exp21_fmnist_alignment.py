"""
Exp 21 — Fashion-MNIST alignment anomaly: fluke or real?

Exp17 measured the corrected latent enc↔dec alignment (exp07 measure) at
0.424 on the single Fashion-MNIST main checkpoint — well above the ~0.25
|random| level, unlike MNIST's clean null (exp07/exp15/exp19: 0.03 ± 0.13).

This experiment settles whether that anomaly survives seed spread:

    z_enc[c] = encode( Q_in top +eigvec for latent dim c, normalised to [0,1] )
    z_dec[c] = Q_dec(mean_img_c) top +eigvec · scale
    metric   = mean_c cos(z_enc[c], z_dec[c])

computed for the FMNIST main checkpoint + all 5 seeds, with an
FMNIST-specific empirical |random| baseline (500 unit-vector draws against
the same z_enc codes, mean ± std of the 10-class mean), and per-class values.

If the effect survives across seeds it is a genuine dataset-dependent
positive worth a paper paragraph; if not, it is recorded as a fluke of the
single checkpoint.  Either way the per-class panel shows which classes
carry it.

Outputs:
    figures/fashion_mnist/exp21_fmnist_alignment.png
    figures/fashion_mnist/exp21_results.json
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models    import FullBilinearVAE
from train     import load_checkpoint
from analysis  import (get_encoder_interaction_matrix, get_decoder_interaction_matrix,
                       decompose, mean_lat_norm)
from visualize import save_fig

DATA   = "/home/v25/ippa6201/bilinear-mlp-repro/data"
CKPTS  = [("main", "checkpoints/fashion_mnist/model.pt")] + \
         [(f"seed{s}", f"checkpoints/fashion_mnist/seeds/seed{s}.pt") for s in range(5)]
N_RAND = 500
MNIST_V2_MEAN = 0.026   # exp19 group mean, for reference
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def corrected_alignment(model, mean_imgs, scale, classes):
    """exp07/exp19 measure; returns per-class cosines and the z_enc codes."""
    per_class, z_encs = [], []
    with torch.no_grad():
        for c in classes:
            d = torch.zeros(model.d_latent); d[c] = 1.0
            vals_e, vecs_e = decompose(get_encoder_interaction_matrix(model, d))
            pos_e = (vals_e > 0).nonzero(as_tuple=True)[0]
            ev = vecs_e[pos_e[0]] if len(pos_e) else torch.zeros(model.d_input)
            v = ev - ev.min(); v = v / (v.max() + 1e-8)
            mu, _ = model.encode(v.unsqueeze(0))
            z_enc = mu.squeeze(0); z_encs.append(z_enc)
            vals_d, vecs_d = decompose(get_decoder_interaction_matrix(model, mean_imgs[c]))
            pos_d = (vals_d > 0).nonzero(as_tuple=True)[0]
            z_dec = vecs_d[pos_d[0]] * scale if len(pos_d) else torch.zeros(model.d_latent)
            per_class.append(_cos(z_enc, z_dec))
    return per_class, z_encs


def random_baseline(z_encs, d_latent, n=N_RAND, seed=42):
    """Distribution over draws of mean_c |cos(z_enc[c], random unit)|."""
    g = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(n):
        vals = []
        for z in z_encs:
            r = torch.randn(d_latent, generator=g); r = r / r.norm()
            vals.append(abs(_cos(z, r)))
        draws.append(float(np.mean(vals)))
    return float(np.mean(draws)), float(np.std(draws))


def main():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    loader = DataLoader(datasets.FashionMNIST(DATA, train=False, download=False,
                                              transform=tfm),
                        batch_size=512, shuffle=False)
    buckets = {}
    for x, y in loader:
        for i, l in enumerate(y.tolist()):
            buckets.setdefault(l, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in sorted(buckets.items())}
    classes = sorted(mean_imgs)

    rows = []
    print(f"{'ckpt':<8}{'align':>8}{'|rand| baseline':>18}")
    print("-" * 36)
    for name, path in CKPTS:
        if not Path(path).exists():
            print(f"{name}: {path} missing — skipped"); continue
        model = FullBilinearVAE(); load_checkpoint(model, path); model.eval()
        scale = mean_lat_norm(model, loader)
        per_class, z_encs = corrected_alignment(model, mean_imgs, scale, classes)
        rb_m, rb_s = random_baseline(z_encs, model.d_latent)
        rows.append({"ckpt": name, "alignment": float(np.mean(per_class)),
                     "alignment_per_class": per_class,
                     "rand_baseline_mean": rb_m, "rand_baseline_std": rb_s})
        print(f"{name:<8}{np.mean(per_class):>8.3f}{rb_m:>12.3f}±{rb_s:.3f}")

    aligns = [r["alignment"] for r in rows]
    mean_a, std_a = float(np.mean(aligns)), float(np.std(aligns))
    rb_pool_m = float(np.mean([r["rand_baseline_mean"] for r in rows]))
    rb_pool_s = float(np.mean([r["rand_baseline_std"] for r in rows]))
    per_class_mat = np.array([r["alignment_per_class"] for r in rows])  # (n_ckpt, 10)
    class_means = per_class_mat.mean(0)

    survives = mean_a - std_a > rb_pool_m
    print(f"\nFMNIST alignment across {len(rows)} ckpts: {mean_a:.3f} ± {std_a:.3f}")
    print(f"|random| baseline (pooled):  {rb_pool_m:.3f} ± {rb_pool_s:.3f}")
    print(f"MNIST v2 reference (exp19):  {MNIST_V2_MEAN:.3f}")
    print(f"Per-class means: " +
          "  ".join(f"{CLASS_NAMES[c][:7]}:{class_means[c]:+.2f}" for c in classes))
    print(f"\nAnomaly survives seed spread (mean−std > |random| mean): {survives}")

    with open("figures/fashion_mnist/exp21_results.json", "w") as f:
        json.dump({"per_ckpt": rows,
                   "alignment_mean": mean_a, "alignment_std": std_a,
                   "rand_baseline_pooled_mean": rb_pool_m,
                   "rand_baseline_pooled_std": rb_pool_s,
                   "per_class_mean": {CLASS_NAMES[c]: float(class_means[c])
                                      for c in classes},
                   "mnist_v2_reference_mean": MNIST_V2_MEAN,
                   "anomaly_survives": bool(survives)}, f, indent=2)

    # ── Figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    names = [r["ckpt"] for r in rows]
    x = np.arange(len(rows))

    axes[0].axhspan(rb_pool_m - rb_pool_s, rb_pool_m + rb_pool_s,
                    color="gray", alpha=0.25,
                    label=f"|random| baseline {rb_pool_m:.2f}±{rb_pool_s:.2f}")
    axes[0].axhline(rb_pool_m, color="gray", ls="--", lw=1)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].axhline(MNIST_V2_MEAN, color="steelblue", ls=":", lw=1.5,
                    label=f"MNIST v2 mean ({MNIST_V2_MEAN:.2f}, exp19)")
    axes[0].scatter(x, aligns, s=80, color="darkorange", zorder=3, label="FMNIST ckpts")
    axes[0].axhline(mean_a, color="darkorange", lw=1.2,
                    label=f"FMNIST mean {mean_a:.2f}±{std_a:.2f}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=45, fontsize=8)
    axes[0].set_ylabel("Corrected latent enc↔dec cos")
    axes[0].set_ylim(-0.5, 0.7)
    axes[0].set_title("Per-checkpoint alignment vs |random| band", fontsize=10)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3, axis="y")

    im = axes[1].imshow(per_class_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(rows))); axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_title("Per-class alignment cos (rows = checkpoints)", fontsize=10)
    for i in range(len(rows)):
        for j in range(10):
            v = per_class_mat[i, j]
            axes[1].text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(v) > 0.6 else "black")
    plt.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)

    verdict = "survives seed spread" if survives else "does not survive seed spread"
    fig.suptitle(f"Exp 21 — Fashion-MNIST alignment anomaly {verdict}: "
                 f"{mean_a:.2f} ± {std_a:.2f} vs |random| {rb_pool_m:.2f} ± {rb_pool_s:.2f}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/fashion_mnist/exp21_fmnist_alignment.png")


if __name__ == "__main__":
    main()
