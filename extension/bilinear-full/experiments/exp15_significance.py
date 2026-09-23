"""
Exp 15 — Statistical Significance (Bootstrap over 5 seeds)

All key metrics from the extension are computed per-seed and reported as
mean ± std, with a bootstrap 95% CI.  This converts point estimates into
proper statistical claims.

Metrics tested:
  1. Near-universal cos (decoder): expected ~0.808 ± ?
  2. Encoder cross-class cos:      expected ~0.364 ± ?
  3. Causal accuracy (out of 10):  expected ~6 ± ?
  4. Latent alignment (enc↔dec):   expected ~0.007 ± ?
  5. Decoder seed consistency:     pairwise between seeds

Requires: checkpoints/mnist/seeds/seed0..4.pt

Figure saved:
    figures/mnist/exp15_significance.png
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, get_encoder_interaction_matrix,
                      decompose, mean_lat_norm, compute_class_means)
from visualize import save_fig

SEEDS = [0, 1, 2, 3, 4]
DATA  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
N_BOOTSTRAP = 2000


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
    return float(np.mean([abs(_cos(eigvecs[a], eigvecs[b])) for a, b in pairs]))


def _enc_cross_class_cos(model, classes):
    eigvecs = {}
    for c in classes:
        d = torch.zeros(model.d_latent); d[c] = 1.0
        Q = get_encoder_interaction_matrix(model, d)
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        eigvecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_input)
    pairs = list(combinations(classes, 2))
    return float(np.mean([abs(_cos(eigvecs[a], eigvecs[b])) for a, b in pairs]))


def _causal_accuracy(model, mean_imgs, class_means, scale, classes):
    correct = 0
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
            correct += int(nearest == c)
    return correct


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

    cos_vals = [_cos(enc_codes[c], dec_eigvecs[c]) for c in classes]
    return float(np.mean(cos_vals))


def _bootstrap_ci(values, n=N_BOOTSTRAP, alpha=0.05):
    values = np.array(values)
    rng    = np.random.default_rng(42)
    boot   = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(values.std()), lo, hi


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
    classes   = sorted(mean_imgs.keys())

    metrics = {k: [] for k in ["dec_univ", "enc_cross", "causal_acc", "alignment"]}

    for s in SEEDS:
        model = FullBilinearVAE()
        load_checkpoint(model, f"checkpoints/mnist/seeds/seed{s}.pt")
        model.eval()
        scale = mean_lat_norm(model, loader)
        class_means = compute_class_means(model, loader)

        metrics["dec_univ"].append(_near_universal_cos(model, mean_imgs, classes))
        metrics["enc_cross"].append(_enc_cross_class_cos(model, classes))
        metrics["causal_acc"].append(_causal_accuracy(model, mean_imgs, class_means, scale, classes))
        metrics["alignment"].append(_latent_alignment(model, mean_imgs, scale, classes))
        print(f"  seed{s}: dec_univ={metrics['dec_univ'][-1]:.4f}  "
              f"enc_cross={metrics['enc_cross'][-1]:.4f}  "
              f"causal={metrics['causal_acc'][-1]}/10  "
              f"align={metrics['alignment'][-1]:.4f}")

    print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8}  {'95% CI':>20}")
    print("-" * 60)
    ci_results = {}
    for k, vals in metrics.items():
        mean, std, lo, hi = _bootstrap_ci(vals)
        ci_results[k] = (mean, std, lo, hi)
        print(f"  {k:<18}  {mean:>8.4f}  {std:>8.4f}  [{lo:.4f}, {hi:.4f}]")

    with open("figures/mnist/exp15_results.json", "w") as f:
        json.dump({k: {"per_seed": metrics[k], "mean": ci_results[k][0],
                       "std": ci_results[k][1],
                       "ci95": [ci_results[k][2], ci_results[k][3]]}
                   for k in metrics}, f, indent=2)

    # Figure
    metric_labels = {
        "dec_univ":   "Decoder near-universal\ncross-class cos",
        "enc_cross":  "Encoder cross-class\ncos (discriminative)",
        "causal_acc": "Causal accuracy\n(out of 10)",
        "alignment":  "Latent enc↔dec\nalignment cos",
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["seagreen", "darkorchid", "steelblue", "indianred"]

    for ax, (k, col) in zip(axes, zip(ci_results.keys(), colors)):
        vals = metrics[k]
        mean, std, lo, hi = ci_results[k]
        ax.scatter(range(len(vals)), vals, color=col, s=80, zorder=3, label="seeds")
        ax.axhline(mean, color="black", linewidth=1.5, label=f"mean={mean:.3f}")
        ax.axhspan(lo, hi, color=col, alpha=0.15, label="95% CI")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"s{s}" for s in SEEDS], fontsize=9)
        ax.set_title(metric_labels[k], fontsize=9)
        ax.text(0.5, 0.02, f"mean={mean:.3f} ± {std:.3f}", ha="center", va="bottom",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 15 — Statistical significance: 5-seed bootstrap 95% CIs\n"
                 "(FullBilinearVAE on MNIST)", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp15_significance.png")


if __name__ == "__main__":
    main()
